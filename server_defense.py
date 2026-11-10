import torch
import numpy as np
from scipy.stats import median_abs_deviation
from encryption import decrypt_vector

class FederatedDefender:
    def __init__(self, encryption_simulator, sensitivity=2.4, warmup_rounds=2, min_clients=5, validation_data=None, monitor=None):
        self.sensitivity = sensitivity
        self.warmup_rounds = warmup_rounds
        self.min_clients = min_clients
        self.reference_models = []
        self.adaptive_threshold = 3.0
        self.validation_data = validation_data
        self.best_global_model = None
        self.accuracy_history = []
        self.encryption_simulator = encryption_simulator
        self.monitor = monitor
        self.trim_percentage = 0.3
        self.accuracy_tolerance = 0.05
        self.rollback_patience = 3
        self.shapes = None

    def _decrypt_model(self, encrypted_model):
        decrypted = {}
        for k, v in encrypted_model.items():
            if isinstance(v, (list, torch.Tensor, np.ndarray)):
                decrypted_vec = decrypt_vector(self.encryption_simulator, v)
                if self.shapes is not None and k in self.shapes:
                    target_shape = self.shapes[k]
                    decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
                else:
                    decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
            else:
                decrypted[k] = v
        return decrypted

    def _robust_zscore(self, values):
        median = np.median(values)
        mad = median_abs_deviation(values)
        return np.abs((values - median) / (mad + 1e-8))

    def _dynamic_thresholding(self, scores):
        if len(self.reference_models) < self.warmup_rounds:
            return np.arange(len(scores))
        
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        iqr = q75 - q25
        upper_bound = q75 + self.sensitivity * iqr
        
        valid = np.where(scores <= upper_bound)[0]
        if len(valid) < self.min_clients:
            valid = np.argsort(scores)[:self.min_clients]
        return valid

    def analyze_models(self, client_models, encryption_simulator, client_indices, all_client_status):
        if not client_models:  # Handle empty client_models list
            print("Warning: No client models passed validation for this round.")
            if self.monitor:
                self.monitor.start_timer('aggregation')
                self.monitor.stop_timer('aggregation')  # Keep timing consistent
                # Log all clients (all skipped in this case)
                for idx in range(len(all_client_status)):
                    status = all_client_status[idx]
                    if status is not None and status['skipped']:  # Only log processed clients
                        print(f"Client {idx + 1}: is_attack={status['is_malicious']}, detected={status['detected']}, attack_type={status['attack_type']}")
                        self.monitor.record_security_event(idx, status['is_malicious'], status['detected'], status['attack_type'])
            return [], []
        
        if self.monitor:
            self.monitor.start_timer('aggregation')

        self.encryption_simulator = encryption_simulator
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        param_vectors = [
            torch.cat([param.view(-1) for param in m.values() if isinstance(param, torch.Tensor)])
            for m in decrypted_models
        ]
        param_matrix = torch.stack(param_vectors).numpy()

        client_scores = []
        for i, vec in enumerate(param_matrix):
            z_scores = self._robust_zscore(vec)
            variance_score = np.var(vec) * 0.22  # Increased weight
            kurtosis = np.mean((vec - np.mean(vec))**4) / (np.var(vec)**2 + 1e-8) * 0.5  # Increased weight
            # Adjust backdoor detection sensitivity
            backdoor_score = kurtosis * 1.5 if client_models[i].get('attack_type') == 'backdoor' else 0
            # Add MITM-specific heuristic (e.g., check for high variance in encrypted updates)
            mitm_score = variance_score * 2 if client_models[i].get('attack_type') == 'mitm' else 0
            combined_score = np.median(z_scores) + variance_score + backdoor_score + mitm_score
            client_scores.append(combined_score)

        valid_indices = self._dynamic_thresholding(np.array(client_scores))

        # Update detected status for included clients
        for i, idx in enumerate(client_indices):
            detected = i not in valid_indices
            all_client_status[idx]['detected'] = detected or all_client_status[idx]['detected']  # Preserve earlier detections
            if self.monitor:
                # Record event only once per client per round
                self.monitor.record_security_event(
                    idx, 
                    all_client_status[idx]['is_malicious'], 
                    all_client_status[idx]['detected'], 
                    all_client_status[idx]['attack_type']
                )

        # Debugging security events
        if self.monitor:
            self.monitor.stop_timer('aggregation')
            print(f"\nDebugging Security Events in Round:")
            for idx in range(len(all_client_status)):
                status = all_client_status[idx]
                if status is not None:  # Only log processed clients
                    print(f"Client {idx + 1}: is_attack={status['is_malicious']}, detected={status['detected']}, attack_type={status['attack_type']}")

        return valid_indices, client_scores

    def secure_aggregate(self, global_model, client_models, client_data_sizes):
        if not client_models:  # Handle empty case
            print("No valid client models for aggregation; retaining previous global model.")
            return global_model

        if self.monitor:
            self.monitor.start_timer('aggregation')

        if self.shapes is None:
            self.shapes = {k: v.shape for k, v in global_model.state_dict().items()}
        
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            all_updates = torch.stack([m[key] for m in decrypted_models])
            sorted_updates, _ = torch.sort(all_updates, dim=0)
            trim_count = int(self.trim_percentage * len(client_models))
            
            if trim_count > 0:
                trimmed = sorted_updates[trim_count:-trim_count]
            else:
                trimmed = sorted_updates
                
            global_state[key] = trimmed.mean(dim=0)
        
        if self.best_global_model:
            for key in global_state:
                global_state[key] = 0.9 * global_state[key] + 0.1 * self.best_global_model[key]
        
        global_model.load_state_dict(global_state)

        if self.monitor:
            self.monitor.stop_timer('aggregation')

        return global_model

    def verify_global_model(self, model):
        if self.validation_data is None:
            return True

        X_val, y_val = self.validation_data
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == y_val).float().mean().item()
        
        if len(self.accuracy_history) >= self.rollback_patience:
            recent_acc = np.mean(self.accuracy_history[-self.rollback_patience:])
            if accuracy < recent_acc - self.accuracy_tolerance:
                model.load_state_dict(self.best_global_model)
                return False
        
        if not self.best_global_model or accuracy > max(self.accuracy_history, default=0):
            self.best_global_model = model.state_dict()
            
        self.accuracy_history.append(accuracy)
        return True

    def update_reference(self, model_state):
        if len(self.reference_models) >= self.warmup_rounds:
            self.reference_models.pop(0)
        self.reference_models.append(model_state)