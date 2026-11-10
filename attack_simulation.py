import torch
import random
import numpy as np

def data_poisoning_attack(X, y, poison_ratio=0.3):
    """
    Simulate a data poisoning attack by flipping labels for a portion of the data.
    
    Args:
    X (torch.Tensor): Input features
    y (torch.Tensor): Labels
    poison_ratio (float): Ratio of data to be poisoned
    
    Returns:
    tuple: Poisoned X and y
    """
    num_samples = len(y)
    num_poisoned = int(num_samples * poison_ratio)
    poisoned_indices = random.sample(range(num_samples), num_poisoned)
    
    y_poisoned = y.clone()
    for idx in poisoned_indices:
        y_poisoned[idx] = (y_poisoned[idx] + 1) % 3  # Assuming 3 classes
    
    return X, y_poisoned, {'is_malicious': True}

def model_poisoning_attack(model_state_dict, attack_strength=2.0):
    poisoned_state_dict = {}
    for key, param in model_state_dict.items():
        noise = torch.randn_like(param) * attack_strength
        poisoned_state_dict[key] = param + noise
    poisoned_state_dict['is_malicious'] = True
    return poisoned_state_dict

def backdoor_attack(X, y, trigger_pattern, target_label, backdoor_ratio=0.1):
    """
    Simulate a backdoor attack by inserting a trigger pattern into a portion of the data.
    
    Args:
    X (torch.Tensor): Input features
    y (torch.Tensor): Labels
    trigger_pattern (torch.Tensor): The backdoor trigger pattern
    target_label (int): The target label for backdoored samples
    backdoor_ratio (float): Ratio of data to be backdoored
    
    Returns:
    tuple: Backdoored X and y
    """
    num_samples = len(y)
    num_backdoored = int(num_samples * backdoor_ratio)
    backdoored_indices = random.sample(range(num_samples), num_backdoored)
    
    X_backdoored = X.clone()
    y_backdoored = y.clone()
    
    for idx in backdoored_indices:
        X_backdoored[idx] += trigger_pattern
        y_backdoored[idx] = target_label
    attack_info = {'is_malicious': True, 'attack_type': 'backdoor'}
    return X_backdoored, y_backdoored, attack_info

# In attack_simulation.py
def mitm_attack(model_update, attack_strength=2.0):
    modified_update = {}
    for key, param in model_update.items():
        if isinstance(param, dict) and 'data' in param:  # Handle encrypted vector dictionary
            noise = [np.random.randn() * attack_strength for _ in param['data']]
            modified_update[key] = {
                'data': [p + n for p, n in zip(param['data'], noise)],
                'hash': param['hash'],  # Keep original hash to simulate tampering
                'malicious': True
            }
        elif isinstance(param, torch.Tensor):
            noise = torch.randn_like(param) * attack_strength
            modified_update[key] = param + noise
        else:
            modified_update[key] = param
    modified_update['is_malicious'] = True
    return modified_update