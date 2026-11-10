import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model import CirrhosisPredictor
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
    """
    Train a local model using the provided data.

    This function trains the given model using CrossEntropyLoss and AdamW optimizer.
    It uses mini-batch training with a batch size of 64.

    Args:
        model (nn.Module): The neural network model to be trained.
        data (tuple): A tuple containing training data (X, y), where X is the input features
                      and y is the corresponding labels.
        epochs (int, optional): The number of training epochs. Defaults to 30.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.

    Returns:
        dict: The state dictionary of the trained model.

    Note:
        - The function uses a weight decay of 1e-5 for regularization.
        - The data is shuffled before each epoch.
        - The model is set to training mode before the training loop.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    X, y = data
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def aggregate_models(global_model, client_models, client_data_sizes, encryption_simulator):
    """
    Aggregate multiple client models into a global model using federated averaging.

    This function performs a weighted average of the client models' parameters,
    where the weights are proportional to the size of each client's dataset.
    The client models' parameters are assumed to be encrypted and are decrypted
    before aggregation.

    Args:
        global_model (nn.Module): The global model to be updated with aggregated parameters.
        client_models (list): A list of state dictionaries from client models.
        client_data_sizes (list): A list of integers representing the data size for each client.
        encryption_simulator (object): An object with a decrypt_vector method for decrypting model parameters.

    Returns:
        None. The global_model is updated in-place.

    Note:
        - The function assumes that all client models have the same architecture as the global model.
        - Client model parameters are expected to be encrypted and are decrypted using the provided encryption_simulator.
        - The aggregation is done layer-wise and parameter-wise.
        - The global model's state dict is directly updated with the aggregated parameters.
    """
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
    max_rounds = 100,
    patience=5,
    min_delta=0.001,
    enable_defense=True,
    monitor=None
):
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

    # Initialize lists for security metrics
    security_true_labels = []  # True malicious status
    security_pred_labels = []  # Detection status
    security_scores = []       # Anomaly scores
    round_accuracies = []
    attack_metrics = {
        'data_poisoning': {'true': [], 'pred': [], 'scores': []},
        'model_poisoning': {'true': [], 'pred': [], 'scores': []},
        'backdoor': {'true': [], 'pred': [], 'scores': []},
        'mitm': {'true': [], 'pred': [], 'scores': []}
    }

    # Initialize attack counters
    total_attack_counters = {
        'data_poisoning': 0,
        'model_poisoning': 0,
        'backdoor': 0,
        'mitm': 0
    }
    post_warmup_attack_counters = {
        'data_poisoning': 0,
        'model_poisoning': 0,
        'backdoor': 0,
        'mitm': 0
    }
    
    trigger_pattern = torch.ones_like(X_test_tensor[0]) * 0.1
    
    monitor.record_scalability_metrics(len(client_data), sum(len(c[0]) for c in client_data))
    
    # Add list to store accuracies per round
    round_accuracies = []

    for round in range(max_rounds):
        print(f"\nRound {round + 1} training:\n")
        if monitor:
            monitor.start_timer('round')

        client_models = []
        client_data_sizes = []
        client_indices = []
        all_client_status = [None] * len(client_data)
        
        round_attack_counters = {
            'data_poisoning': 0,
            'model_poisoning': 0,
            'backdoor': 0,
            'mitm': 0
        }
        
        past_warmup = round >= warmup_rounds
        
        for client_idx, (client_X, client_y) in enumerate(client_data):
            print(f"Client {client_idx + 1} training:")
            attack_type = np.random.choice(
                ['none', 'data_poisoning', 'model_poisoning', 'backdoor', 'mitm'],
                p=[0.8, 0.05, 0.05, 0.05, 0.05]
            )
            attack_info = {'is_malicious': attack_type != 'none', 'attack_type': attack_type}
            
            # Count attacks and apply them
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
            # else:
            #     attack_info = {'is_malicious': False, 'attack_type': 'none'}

            if not enhanced_local_data_validation(client_X, client_y):
                print(f"Skipping client {client_idx + 1} due to poor data quality. Attack applied: {attack_type}")
                all_client_status[client_idx] = {
                    'is_malicious': attack_info['is_malicious'],
                    'attack_type': attack_type,
                    'detected': attack_info['is_malicious'],
                    'skipped': True,
                    'score': 1000.0
                }
                if monitor and attack_info['is_malicious']:  # Record event explicitly
                    monitor.record_security_event(client_idx, True, True, attack_type)
                continue
            
            val_split = int(0.2 * len(client_X))
            x_train, x_val = client_X[:-val_split], client_X[-val_split:]
            y_train, y_val = client_y[:-val_split], client_y[-val_split:]
            
            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())
            
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
            
            if attack_type == 'model_poisoning':
                client_model_state = model_poisoning_attack(client_model_state)
                total_attack_counters['model_poisoning'] += 1
                if past_warmup:
                    post_warmup_attack_counters['model_poisoning'] += 1
                    round_attack_counters['model_poisoning'] += 1
                attack_info = {'is_malicious': True, 'attack_type': 'model_poisoning'}

            if not enhanced_local_model_validation(
                local_model, x_val, y_val,
                accuracy_threshold=0.58,
                loss_threshold=0.96,
                consistency_threshold=0.65,
                monitor=monitor
            ):
                print(f"Client {client_idx + 1}: Skipping due to model poisoning detection.")
                all_client_status[client_idx] = {
                    'is_malicious': attack_info['is_malicious'],
                    'attack_type': attack_type,
                    'detected': attack_info['is_malicious'],
                    'skipped': True,
                    'score': 1000.0
                }
                if monitor and attack_info['is_malicious']:  # Record event explicitly
                    monitor.record_security_event(client_idx, True, True, attack_type)
                continue
            
            encrypted_state = {
                k: encrypt_vector(encryption_simulator, v.flatten())
                for k, v in client_model_state.items()
                if isinstance(v, torch.Tensor)
            }
            
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
            monitor.record_scalability_metrics(len(client_models), sum(client_data_sizes))

        if enable_defense:
            valid_indices, client_scores = defender.analyze_models(client_models, encryption_simulator, client_indices, all_client_status)
            for i, idx in enumerate(client_indices):
                all_client_status[idx]['score'] = client_scores[i]
                all_client_status[idx]['detected'] = i not in valid_indices
            # filtered_models = defender.analyze_models(client_models, encryption_simulator, client_indices, all_client_status)
            filtered_models = [client_models[i] for i in valid_indices]
            global_model = defender.secure_aggregate(global_model, filtered_models, client_data_sizes)
        
        if not defender.verify_global_model(global_model):
            print("Model rollback due to verification failure")
            rounds_without_improvement += 1
            continue
        
        defender.update_reference(global_model.state_dict())

        if monitor:
            monitor.stop_timer('round')
        
        test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        print(f"Round {round+1}, Test Accuracy: {test_accuracy:.4f}")
        round_accuracies.append(test_accuracy)  # Store accuracy

        if monitor:
            monitor.metrics['performance']['accuracy'].append(test_accuracy)

        # Collect security metrics
        for status in all_client_status:
            if status is not None:
                security_true_labels.append(status['is_malicious'])
                security_pred_labels.append(status['detected'])
                security_scores.append(status['score'] if status['score'] is not None else 1000.0)
        
        # Collect attack-specific metrics, including benign samples
        for status in all_client_status:
            if status is not None:
                for attack_type in attack_metrics:
                    is_malicious_for_type = status['attack_type'] == attack_type
                    attack_metrics[attack_type]['true'].append(is_malicious_for_type)
                    attack_metrics[attack_type]['pred'].append(status['detected'])
                    attack_metrics[attack_type]['scores'].append(status['score'] if status['score'] is not None else 1000.0)

        if test_accuracy > best_accuracy + min_delta:
            best_accuracy = test_accuracy
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        
        if rounds_without_improvement >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break
    
    print("\n=== Total Attack Simulation Summary (All Rounds) ===")
    for attack_type, count in total_attack_counters.items():
        print(f"{attack_type.replace('_', ' ').title()}: {count} instances")
    
    print("\n=== Total Attack Simulation Summary (Post-Warmup Rounds) ===")
    for attack_type, count in post_warmup_attack_counters.items():
        print(f"{attack_type.replace('_', ' ').title()}: {count} instances")
    
    return global_model, round_accuracies, security_true_labels, security_pred_labels, security_scores, attack_metrics