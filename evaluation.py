import torch

def evaluate_model(model, X, y):
    """
    Evaluate the model's performance on given data.
    
    Args:
    model (torch.nn.Module): The model to evaluate.
    X (torch.Tensor): Input features.
    y (torch.Tensor): True labels.
    
    Returns:
    float: Accuracy of the model on the given data.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean()
    return accuracy.item()

def calculate_metrics(model, X, y):
    """
    Calculate various performance metrics for the model.
    
    Args:
    model (torch.nn.Module): The model to evaluate.
    X (torch.Tensor): Input features.
    y (torch.Tensor): True labels.
    
    Returns:
    dict: A dictionary containing various performance metrics.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        
        # Calculate accuracy
        accuracy = (predictions == y).float().mean().item()
        
        # Calculate precision, recall, and F1-score for each class
        num_classes = outputs.shape[1]
        precision = []
        recall = []
        f1_score = []
        
        for class_idx in range(num_classes):
            true_positives = ((predictions == class_idx) & (y == class_idx)).sum().float()
            false_positives = ((predictions == class_idx) & (y != class_idx)).sum().float()
            false_negatives = ((predictions != class_idx) & (y == class_idx)).sum().float()
            
            class_precision = true_positives / (true_positives + false_positives + 1e-8)
            class_recall = true_positives / (true_positives + false_negatives + 1e-8)
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)
            
            precision.append(class_precision.item())
            recall.append(class_recall.item())
            f1_score.append(class_f1.item())
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def print_evaluation_results(metrics):
    """
    Print the evaluation results in a formatted manner.
    
    Args:
    metrics (dict): A dictionary containing various performance metrics.
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for i, (p, r, f1) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['f1_score'])):
        print(f"Class {i}:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall: {r:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print()