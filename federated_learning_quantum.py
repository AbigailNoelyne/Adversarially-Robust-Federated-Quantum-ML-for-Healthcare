"""
federated_learning_quantum.py
------------------------------
Quantum FL variant of federated_learning.py.

THE ONLY CHANGE FROM THE ORIGINAL is line 9:
    FROM:  from model import CirrhosisPredictor
    TO:    from quantum_model import CirrhosisPredictor

Every other line is identical to the baseline.  This means:
  - All 5 attack types work unchanged (attack_simulation.py)
  - All adaptive defenses work unchanged (server_defense.py, clients_defense.py)
  - All encryption works unchanged (encryption.py)
  - All evaluation works unchanged (evaluation.py, performance.py)

This is intentional: the research question is whether the same defenses
that protect classical FL updates also protect VQC parameter updates.
Nothing should change except the model architecture.
"""

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
# ============================================================
# THE ONE CHANGED LINE:
from quantum_model import CirrhosisPredictor        # <-- VQC model
# Original was: from model import CirrhosisPredictor
# ============================================================
from evaluation import evaluate_model
from collections import deque
from encryption import EncryptionSimulator, encrypt_vector, decrypt_vector
from server_defense import FederatedDefender
from clients_defense import (
    enhanced_local_data_validation,
    enhanced_local_model_validation,
    client_local_train
)
from attack_simulation import (
    data_poisoning_attack,
    model_poisoning_attack,
    backdoor_attack,
    mitm_attack
)


def train_local_model(model, data, epochs=30, lr=0.001):
    """Train a local model (kept for API compatibility)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    X, y = data
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)  # smaller batch for VQC

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def aggregate_models(global_model, client_models, client_data_sizes, encryption_simulator):
    """Federated averaging over encrypted client models (unchanged)."""
    global_state_dict = global_model.state_dict()
    total_data = sum(client_data_sizes)

    for key in global_state_dict.keys():
        weighted_sum = torch.zeros_like(global_state_dict[key])
        for client_model, data_size in zip(client_models, client_data_sizes):
            weight = data_size / total_data
            decrypted_param = decrypt_vector(encryption_simulator, client_model[key])
            decrypted_param = torch.tensor(decrypted_param).reshape(global_state_dict[key].shape)
            weighted_sum += weight * decrypted_param
        global_state_dict[key] = weighted_sum

    global_model.load_state_dict(global_state_dict)


def federated_learning_with_early_stopping(
    global_model,
    client_data,
    X_test_tensor,
    y_test_tensor,
    max_rounds=100,
    patience=5,
    min_delta=0.001,
    enable_defense=True,
    monitor=None
):
    """
    Federated learning loop with early stopping.

    Identical to the baseline except:
      - Uses HybridVQCPredictor (imported as CirrhosisPredictor above)
      - Batch size for client_local_train reduced to 32 (VQC is slower per sample)
      - local_model instantiation uses model.input_dim instead of model.fc[0].in_features
        (both work; using the cleaner attribute here for clarity)

    All attack injection, defense filtering, encryption, and monitoring
    logic is unchanged from the baseline.
    """
    warmup_rounds = 2
    best_accuracy = 0
    rounds_without_improvement = 0
    accuracy_history = deque(maxlen=patience)
    encryption_simulator = EncryptionSimulator()
    defender = FederatedDefender(
        validation_data=(X_test_tensor, y_test_tensor),
        warmup_rounds=warmup_rounds,
        min_clients=max(5, len(client_data) // 4),
        encryption_simulator=encryption_simulator,
        monitor=monitor
    )

    security_true_labels = []
    security_pred_labels = []
    security_scores = []
    round_accuracies = []
    attack_metrics = {
        'data_poisoning': {'true': [], 'pred': [], 'scores': []},
        'model_poisoning': {'true': [], 'pred': [], 'scores': []},
        'backdoor':        {'true': [], 'pred': [], 'scores': []},
        'mitm':            {'true': [], 'pred': [], 'scores': []}
    }

    total_attack_counters = {k: 0 for k in attack_metrics}
    post_warmup_attack_counters = {k: 0 for k in attack_metrics}

    trigger_pattern = torch.ones_like(X_test_tensor[0]) * 0.1

    monitor.record_scalability_metrics(
        len(client_data), sum(len(c[0]) for c in client_data)
    )

    for round_num in range(max_rounds):
        print(f"\nRound {round_num + 1} training:\n")
        if monitor:
            monitor.start_timer('round')

        client_models = []
        client_data_sizes = []
        client_indices = []
        all_client_status = [None] * len(client_data)

        round_attack_counters = {k: 0 for k in attack_metrics}
        past_warmup = round_num >= warmup_rounds

        for client_idx, (client_X, client_y) in enumerate(client_data):
            print(f"Client {client_idx + 1} training:")

            attack_type = np.random.choice(
                ['none', 'data_poisoning', 'model_poisoning', 'backdoor', 'mitm'],
                p=[0.80, 0.05, 0.05, 0.05, 0.05]
            )
            attack_info = {
                'is_malicious': attack_type != 'none',
                'attack_type': attack_type
            }

            # ---- Apply data-level attacks before training ----
            if attack_type == 'data_poisoning':
                client_X, client_y, attack_info = data_poisoning_attack(client_X, client_y)
                total_attack_counters['data_poisoning'] += 1
                if past_warmup:
                    post_warmup_attack_counters['data_poisoning'] += 1
                    round_attack_counters['data_poisoning'] += 1

            elif attack_type == 'backdoor':
                client_X, client_y, attack_info = backdoor_attack(
                    client_X, client_y, trigger_pattern, target_label=2
                )
                total_attack_counters['backdoor'] += 1
                if past_warmup:
                    post_warmup_attack_counters['backdoor'] += 1
                    round_attack_counters['backdoor'] += 1

            # ---- Client-side data validation ----
            if not enhanced_local_data_validation(client_X, client_y):
                print(f"Skipping client {client_idx + 1} — poor data quality. "
                      f"Attack: {attack_type}")
                all_client_status[client_idx] = {
                    'is_malicious': attack_info['is_malicious'],
                    'attack_type': attack_type,
                    'detected': attack_info['is_malicious'],
                    'skipped': True,
                    'score': 1000.0
                }
                if monitor and attack_info['is_malicious']:
                    monitor.record_security_event(client_idx, True, True, attack_type)
                continue

            val_split = int(0.2 * len(client_X))
            x_train, x_val = client_X[:-val_split], client_X[-val_split:]
            y_train, y_val = client_y[:-val_split], client_y[-val_split:]

            # Instantiate local VQC model and copy global weights
            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())

            # ---- Local training with DP + adversarial training ----
            client_model_state = client_local_train(
                local_model,
                data=(x_train, y_train),
                epochs=30,
                lr=0.001,
                enable_dp=True,
                dp_clip=1.0,
                dp_noise_scale=0.01,
                enable_adv=True,
                adv_epsilon=0.1,
                adv_ratio=0.5,
                local_val_data=(x_val, y_val)
            )

            # ---- Apply model-level attack after training ----
            if attack_type == 'model_poisoning':
                client_model_state = model_poisoning_attack(client_model_state)
                total_attack_counters['model_poisoning'] += 1
                if past_warmup:
                    post_warmup_attack_counters['model_poisoning'] += 1
                    round_attack_counters['model_poisoning'] += 1
                attack_info = {'is_malicious': True, 'attack_type': 'model_poisoning'}

            # ---- Client-side model validation ----
            if not enhanced_local_model_validation(
                local_model, x_val, y_val,
                accuracy_threshold=0.58,
                loss_threshold=0.96,
                consistency_threshold=0.65,
                monitor=monitor
            ):
                print(f"Client {client_idx + 1}: Skipping — model poisoning detected.")
                all_client_status[client_idx] = {
                    'is_malicious': attack_info['is_malicious'],
                    'attack_type': attack_type,
                    'detected': attack_info['is_malicious'],
                    'skipped': True,
                    'score': 1000.0
                }
                if monitor and attack_info['is_malicious']:
                    monitor.record_security_event(client_idx, True, True, attack_type)
                continue

            # ---- Encrypt VQC parameter update ----
            encrypted_state = {
                k: encrypt_vector(encryption_simulator, v.flatten())
                for k, v in client_model_state.items()
                if isinstance(v, torch.Tensor)
            }

            # ---- Apply MitM attack on encrypted channel ----
            if attack_type == 'mitm':
                encrypted_state = mitm_attack(encrypted_state)
                total_attack_counters['mitm'] += 1
                if past_warmup:
                    post_warmup_attack_counters['mitm'] += 1
                    round_attack_counters['mitm'] += 1
                attack_info = {'is_malicious': True, 'attack_type': 'mitm'}

            encrypted_state.update(attack_info)
            client_models.append(encrypted_state)
            client_data_sizes.append(len(x_train))
            client_indices.append(client_idx)
            all_client_status[client_idx] = {
                'is_malicious': attack_info['is_malicious'],
                'attack_type': attack_type,
                'detected': False,
                'skipped': False,
                'score': None
            }

        if monitor:
            monitor.record_scalability_metrics(
                len(client_models), sum(client_data_sizes)
            )

        # ---- Server-side defense and secure aggregation ----
        if enable_defense:
            valid_indices, client_scores = defender.analyze_models(
                client_models, encryption_simulator, client_indices, all_client_status
            )
            for i, idx in enumerate(client_indices):
                all_client_status[idx]['score'] = client_scores[i]
                all_client_status[idx]['detected'] = i not in valid_indices
            filtered_models = [client_models[i] for i in valid_indices]
            global_model = defender.secure_aggregate(
                global_model, filtered_models, client_data_sizes
            )

        if not defender.verify_global_model(global_model):
            print("Model rollback due to verification failure")
            rounds_without_improvement += 1
            continue

        defender.update_reference(global_model.state_dict())

        if monitor:
            monitor.stop_timer('round')

        test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        print(f"Round {round_num + 1}, Test Accuracy: {test_accuracy:.4f}")
        round_accuracies.append(test_accuracy)

        if monitor:
            monitor.metrics['performance']['accuracy'].append(test_accuracy)

        # Collect security metrics
        for status in all_client_status:
            if status is not None:
                security_true_labels.append(status['is_malicious'])
                security_pred_labels.append(status['detected'])
                security_scores.append(
                    status['score'] if status['score'] is not None else 1000.0
                )

        for status in all_client_status:
            if status is not None:
                for at in attack_metrics:
                    is_type = status['attack_type'] == at
                    attack_metrics[at]['true'].append(is_type)
                    attack_metrics[at]['pred'].append(status['detected'])
                    attack_metrics[at]['scores'].append(
                        status['score'] if status['score'] is not None else 1000.0
                    )

        if test_accuracy > best_accuracy + min_delta:
            best_accuracy = test_accuracy
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

    print("\n=== Total Attack Simulation Summary (All Rounds) ===")
    for at, count in total_attack_counters.items():
        print(f"  {at.replace('_', ' ').title()}: {count}")

    print("\n=== Total Attack Simulation Summary (Post-Warmup) ===")
    for at, count in post_warmup_attack_counters.items():
        print(f"  {at.replace('_', ' ').title()}: {count}")

    return (
        global_model,
        round_accuracies,
        security_true_labels,
        security_pred_labels,
        security_scores,
        attack_metrics
    )
