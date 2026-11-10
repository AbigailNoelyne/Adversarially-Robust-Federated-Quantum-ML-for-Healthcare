import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

######################################################################
# Differential Privacy Helper Functions
######################################################################

def clip_gradients(model, max_norm):
    """
    Clips the gradients of the model parameters to a maximum norm.
    Returns the total norm before clipping.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    return total_norm

def add_dp_noise(model, noise_scale):
    """
    Adds Gaussian noise scaled by noise_scale to each gradient.
    This simulates differential privacy by perturbing model updates.
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.data.add_(noise)

######################################################################
# Adversarial Training Functions
######################################################################

def generate_adversarial_examples(model, loss_fn, x, y, epsilon=0.1):
    """
    Generate adversarial examples for a batch using the Fast Gradient Sign Method (FGSM).
    Assumes input x is normalized (e.g., in [0,1]). The attack perturbs
    a subset of the inputs by an amount 'epsilon'.
    """
    # Clone and set requires_grad to compute gradient w.r.t. input
    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = loss_fn(outputs, y)
    model.zero_grad()
    loss.backward()
    # FGSM update: add epsilon * sign(gradient)
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    # Clamp the adversarial examples to ensure valid input range; adjust if needed.
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

######################################################################
# Local Data and Model Validation
######################################################################

def local_data_validation(x, y):
    """
    Perform basic validation on local data.
    Checks if the data contains NaN values.
    """
    if torch.isnan(x).any() or torch.isnan(y).any():
        print("Error: Local dataset contains NaN values.")
        return False
    print("Local data validation passed.")
    return True

def local_model_validation(model, x_val, y_val, accuracy_threshold=0.5):
    """
    Validates the trained model on local validation data.
    Returns True if accuracy is above the threshold.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(x_val)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_val).float().mean().item()
    print(f"Local model validation accuracy: {accuracy:.4f}")
    if accuracy < accuracy_threshold:
        print("Warning: Local model accuracy is below the target threshold.")
        return False
    return True

######################################################################
# Integrated Client Training Function
######################################################################

def client_local_train(model, data, epochs=30, lr=0.001,
                       enable_dp=False, dp_clip=1.0, dp_noise_scale=0.01,
                       enable_adv=False, adv_epsilon=0.1, adv_ratio=0.5,
                       local_val_data=None):
    """
    Extended local training routine that integrates:
      - Differential Privacy: Clipping gradients and adding noise.
      - Adversarial Training: Augmenting a portion of each batch with adversarial examples.
      - Local Model Validation: Optionally validating the model on a held-out validation set.
      
    Parameters:
      model          : PyTorch model to be trained.
      data           : Tuple (X, y) with training data.
      epochs         : Number of training epochs.
      lr             : Learning rate.
      enable_dp      : Enable differential privacy updates.
      dp_clip        : Maximum norm for gradient clipping.
      dp_noise_scale : Standard deviation of the DP noise.
      enable_adv     : Whether to perform adversarial training.
      adv_epsilon    : Perturbation magnitude for adversarial examples.
      adv_ratio      : Fraction of each batch to be replaced with adversarial examples.
      local_val_data : Tuple (x_val, y_val) for local model validation post-training.
      
    Returns:
      Updated model state_dict.
    """
    X, y = data
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Adversarial training step: mix in adversarial loss for a fraction of the batch
            if enable_adv:
                batch_size = batch_X.size(0)
                num_adv = int(adv_ratio * batch_size)
                if num_adv > 0:
                    # Generate adversarial examples on a subset of the batch
                    adv_X = generate_adversarial_examples(model, criterion,
                                                          batch_X[:num_adv],
                                                          batch_y[:num_adv],
                                                          epsilon=adv_epsilon)
                    adv_outputs = model(adv_X)
                    adv_loss = criterion(adv_outputs, batch_y[:num_adv])
                    # Combine the original and adversarial losses equally
                    loss = 0.5 * loss + 0.5 * adv_loss
            
            loss.backward()
            
            # Apply differential privacy mechanisms if enabled
            if enable_dp:
                clip_gradients(model, dp_clip)
                add_dp_noise(model, dp_noise_scale)
            
            optimizer.step()
            
    # If a local validation split is provided, run a simple model validation check.
    if local_val_data:
        x_val, y_val = local_val_data
        validation_passed = local_model_validation(model, x_val, y_val)
        if not validation_passed:
            print("Local model validation failed. Consider additional training or parameter tuning.")
    
    return model.state_dict()

def enhanced_local_model_validation(model, x_val, y_val,
                                    accuracy_threshold=0.56,
                                    loss_threshold=0.9,
                                    adv_epsilon=0.1,
                                    consistency_threshold=0.65,
                                    monitor=None
                                    ):
    """
    Enhanced model validation to detect potential model poisoning.
    
    Validates that:
      - The model achieves a minimum accuracy on clean validation data.
      - The validation loss is within an acceptable range.
      - The model exhibits consistent predictions when evaluated on adversarial examples.
    
    Parameters:
      model               : PyTorch model (e.g. an instance of CirrhosisPredictor).
      x_val, y_val        : Validation data.
      accuracy_threshold  : Minimum acceptable accuracy.
      loss_threshold      : Maximum acceptable loss.
      adv_epsilon         : Perturbation magnitude for adversarial example generation.
      consistency_threshold: Minimum fraction of matching predictions on original and adversarial data.
    
    Returns:
      True if the model passes validation; False otherwise.
    """
    if monitor:
        monitor.start_timer('validation')

    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        outputs = model(x_val)
        loss = criterion(outputs, y_val)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_val).float().mean().item()

    adv_examples = fgsm_attack(model, x_val, y_val, adv_epsilon)
    with torch.no_grad():
        adv_outputs = model(adv_examples)
        _, adv_preds = torch.max(adv_outputs, 1)
        consistency = (preds == adv_preds).float().mean().item()

    # Add backdoor-specific check
    trigger_pattern = torch.ones_like(x_val[0]) * 0.1  # Match simulation trigger
    backdoor_input = x_val.clone()
    backdoor_input += trigger_pattern  # Apply trigger to all validation samples
    with torch.no_grad():
        backdoor_outputs = model(backdoor_input)
        _, backdoor_preds = torch.max(backdoor_outputs, 1)
        target_label = 2  # Match backdoor_attack target_label
        backdoor_success_rate = (backdoor_preds == target_label).float().mean().item()

    # print(f"Local model validation accuracy: {accuracy:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}, loss: {loss.item():.4f}, adversarial consistency: {consistency:.4f}, backdoor_success: {backdoor_success_rate:.4f}")

    is_valid = (accuracy >= accuracy_threshold and 
                loss <= loss_threshold and 
                consistency >= consistency_threshold and 
                backdoor_success_rate < 0.5)  # Flag if >50% samples predict target label)

    if monitor:
        monitor.stop_timer('validation')

    return is_valid

def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data

def enhanced_local_data_validation(x, y, cl_range=2.9, min_label_prop=0.032):
    """
    Enhanced validation for local data to detect potential data poisoning.
    
    Parameters:
      x             : Torch tensor representing features.
      y             : Torch tensor representing labels.
      cl_range      : Multiplier for the IQR to set acceptable value bounds.
      min_label_prop: Minimum proportion of samples required for each label.
    
    Returns:
      True if the data passes validation; False otherwise.
    """
    # Check for missing values
    if torch.isnan(x).any() or torch.isnan(y).any():
        print("Validation failed: found NaN values in data.")
        return False

    # Check feature distributions using interquartile range (IQR)
    # Assume each column corresponds to one feature.
    q1 = torch.quantile(x, 0.25, dim=0)
    q3 = torch.quantile(x, 0.75, dim=0)
    iqr = q3 - q1
    lower_bound = q1 - cl_range * iqr
    upper_bound = q3 + cl_range * iqr

    # Identify values outside the acceptable range and flag if too many are found per feature.
    outlier_mask = (x < lower_bound) | (x > upper_bound)
    outlier_fraction = outlier_mask.float().mean(dim=0)
    if (outlier_fraction > 0.18).any():
        print("Validation warning: Unusual fraction of outliers detected in one or more features.")
        return False

    # Check label distribution: each label must appear at least a minimum fraction of the total samples.
    unique_labels, counts = torch.unique(y, return_counts=True)
    label_proportions = counts.float() / len(y)
    if label_proportions.min() < min_label_prop:
        print("Validation warning: Extremely imbalanced label distribution detected, potential label flipping.")
        return False

    print("Enhanced data validation passed.")
    return True