import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from federated_learning import federated_learning_with_early_stopping
from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
from performance import PerformanceMonitor
from data_preprocessing import split_data_among_clients, load_and_preprocess_data
from model import CirrhosisPredictor

def run_federated_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_runs=3, num_clients=20):
    all_round_accuracies = []
    run_metrics = []
    
    for run in range(num_runs):
        print(f"\n--- Test Run {run + 1} ---")
        monitor = PerformanceMonitor()
        client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)
        
        # Run federated learning
        trained_model, round_accuracies, security_true_labels, security_pred_labels, security_scores, attack_metrics = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor
        )
        
        all_round_accuracies.append(round_accuracies)
        
        # Evaluate final model
        final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
        print_evaluation_results(final_metrics)
        
        # Defense framework performance report
        print("\nDefense Framework Performance Analysis:")
        monitor.print_detailed_report()
        
        report = monitor.generate_report()
        run_metrics.append(report['performance'])
        
        # Attack-specific metrics and ROC curves
        attack_results = {}
        plt.figure(figsize=(10, 6))
        colors = ['b', 'g', 'r', 'c']
        attack_types = ['data_poisoning', 'model_poisoning', 'backdoor', 'mitm']
        
        for attack_type, color in zip(attack_types, colors):
            true = np.array(attack_metrics[attack_type]['true'])
            pred = np.array(attack_metrics[attack_type]['pred'])
            scores = np.array(attack_metrics[attack_type]['scores'])
            
            if len(true) == 0 or len(scores) == 0:
                print(f"No instances of {attack_type} detected.")
                attack_results[attack_type] = {'detection_rate': 0, 'fpr': 0, 'precision': 0, 'auc': 0}
                continue
            
            fpr, tpr, _ = roc_curve(true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{attack_type.replace("_", " ").title()} (AUC = {roc_auc:.2f})')
            
            tp = sum(true & pred)
            fp = sum(~true & pred)
            tn = sum(~true & ~pred)
            fn = sum(true & ~pred)
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            attack_results[attack_type] = {
                'detection_rate': detection_rate,
                'fpr': fpr_val,
                'precision': precision,
                'auc': roc_auc
            }
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f'ROC Curves for Attack Detection - Run {run + 1}', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f'results/attack_detection_roc_run_{run + 1}.png')
        plt.show()
        
        # Attack detection metrics table
        print(f"\n=== Attack Detection Metrics - Run {run + 1} ===")
        print(f"{'Attack Type':<15} | {'Detection Rate':<15}")
        print("-" * 55)
        for attack_type in attack_types:
            metrics = attack_results.get(attack_type, {'detection_rate': 0, 'fpr': 0, 'precision': 0})
            print(f"{attack_type.replace('_', ' ').title():<15} | {metrics['detection_rate']:<15.4f}")
        
        # Model performance metrics
        y_true = y_test_tensor.numpy()
        y_scores = trained_model(X_test_tensor).detach().numpy()
        y_pred = np.argmax(y_scores, axis=1)
        
        cm_model = confusion_matrix(y_true, y_pred)
        print("\nModel Performance Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        classes = ['1', '2', '3']
        sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title(f'Confusion Matrix - Test Run {run + 1}', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f'results/confusion_matrix_defense_fl_run_{run + 1}.png')
        plt.show()
        
        print("\nPlotting ROC Curves for Model Performance:")
        plt.figure(figsize=(10, 6))
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f'ROC Curves - All Classes (Run {run + 1})', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'results/ROC_curve_defense_fl_run_{run + 1}.png')
        plt.show()
        
        # Security metrics
        security_true = np.array(security_true_labels)
        security_pred = np.array(security_pred_labels)
        security_scores = np.array(security_scores)
        
        cm_security = confusion_matrix(security_true, security_pred)
        print("\nSecurity Metrics Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        classes = ['No', 'Yes']
        sns.heatmap(cm_security, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title(f'Confusion Matrix - Test Run {run + 1}', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f'results/cm_sec_{run + 1}.png')
        plt.show()
        
        print("\nPlotting ROC Curve for Security Metrics:")
        fpr, tpr, _ = roc_curve(security_true, security_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Security Detection (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC Curve - Security Detection', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend()
        plt.savefig(f'results/ROC_sec_{run + 1}.png')
        plt.show()
    
    # Accuracy trends across runs
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(all_round_accuracies):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'Test Run {i + 1}', marker='o')
    plt.xlabel('Federation Round', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('Accuracy Across Federation Rounds for 3 Test Runs', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/acc_trend_sec_{num_runs}.png')
    plt.show()
    
    # Latency metrics table
    print("\n=== Latency Metrics Across Runs ===")
    print(f"{'Run':<5} | {'Aggregation Latency':<20} | {'Validation Latency':<20} | {'Average Round Time':<20}")
    print("-" * 70)
    for i, metrics in enumerate(run_metrics, 1):
        agg_lat = metrics.get('avg_aggregation_latency', 0)
        val_lat = metrics.get('avg_validation_latency', 0)
        round_lat = metrics.get('avg_round_latency', 0)
        print(f"{i:<5} | {agg_lat:<20.4f} | {val_lat:<20.4f} | {round_lat:<20.4f}")
    
    avg_agg = np.mean([m.get('avg_aggregation_latency', 0) for m in run_metrics])
    avg_val = np.mean([m.get('avg_validation_latency', 0) for m in run_metrics])
    avg_round = np.mean([m.get('avg_round_latency', 0) for m in run_metrics])
    print(f"{'Avg':<5} | {avg_agg:<20.4f} | {avg_val:<20.4f} | {avg_round:<20.4f}")
    
    # Bar chart for performance metrics
    metrics_names = ['Aggregation Latency', 'Validation Latency', 'Average Round Time']
    runs = ['Run 1', 'Run 2', 'Run 3']
    data = {
        'Aggregation Latency': [max(m.get('avg_aggregation_latency', 0), 1e-6) for m in run_metrics],
        'Validation Latency': [max(m.get('avg_validation_latency', 0), 1e-6) for m in run_metrics],
        'Average Round Time': [max(m.get('avg_round_latency', 0), 1e-6) for m in run_metrics]
    }
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(runs))
    width = 0.25
    plt.bar(x - width, data['Aggregation Latency'], width, label='Aggregation Latency', color='skyblue')
    plt.bar(x, data['Validation Latency'], width, label='Validation Latency', color='lightgreen')
    plt.bar(x + width, data['Average Round Time'], width, label='Average Round Time', color='salmon')
    plt.yscale('log')
    plt.xlabel('Test Run', fontsize=16)
    plt.ylabel('Time (seconds, log scale)', fontsize=16)
    plt.title('Performance Metrics Across Test Runs', fontsize=16)
    plt.xticks(x, runs, fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png')
    plt.show()

def run_scalability_experiment(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, client_counts):
    scalability_data = []
    
    for num_clients_scale in client_counts:
        print(f"\nRunning with {num_clients_scale} clients")
        monitor = PerformanceMonitor()
        client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients_scale)
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)
        
        trained_model, round_accuracies, security_true_labels, security_pred_labels, security_scores, _ = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor
        )
        
        report = monitor.generate_report()
        performance = report['performance']
        agg_lat = performance.get('avg_aggregation_latency', 0)
        val_lat = performance.get('avg_validation_latency', 0)
        round_lat = performance.get('avg_round_latency', 0)
        
        scalability_data.append({
            'num_clients': num_clients_scale,
            'agg_latency': agg_lat,
            'val_latency': val_lat,
            'round_latency': round_lat
        })
    
    # Scalability metrics table
    print("\n=== Scalability Metrics Table ===")
    print(f"{'Clients':<10} | {'Agg Latency':<15} | {'Val Latency':<15} | {'Round Time':<15}")
    print("-" * 60)
    for data in scalability_data:
        print(f"{data['num_clients']:<10} | {data['agg_latency']:<15.4f} | {data['val_latency']:<15.4f} | {data['round_latency']:<15.4f}")
    
    # Line graph for scalability
    plt.figure(figsize=(10, 6))
    plt.plot([d['num_clients'] for d in scalability_data], [d['agg_latency'] for d in scalability_data], label='Aggregation Latency', marker='o')
    plt.plot([d['num_clients'] for d in scalability_data], [d['val_latency'] for d in scalability_data], label='Validation Latency', marker='o')
    plt.plot([d['num_clients'] for d in scalability_data], [d['round_latency'] for d in scalability_data], label='Average Round Time', marker='o')
    plt.xlabel('Number of Clients', fontsize=16)
    plt.ylabel('Time (s)', fontsize=16)
    plt.title('Scalability Metrics vs Number of Clients', fontsize=16)
    plt.yscale('log')
    plt.legend()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True)
    plt.savefig('results/scalability_metrics.png')
    plt.show()

def run_accuracy_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, file_path):
    # Experiment 1: Accuracy vs Number of Clients
    print("\n=== Experiment: Accuracy vs Number of Clients ===")
    client_counts = list(range(2, 21, 2))
    accuracies_vs_clients = []
    
    for num_clients_ce in client_counts:
        print(f"Running with {num_clients_ce} clients")
        monitor = PerformanceMonitor()
        client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients_ce)
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)
        
        trained_model, round_accuracies, _, _, _, _ = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor
        )
        
        accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
        accuracies_vs_clients.append(accuracy)
    
    print("\nAccuracy vs Number of Clients")
    print(f"{'Clients':<10} | {'Accuracy':<10}")
    print("-" * 22)
    for clients, acc in zip(client_counts, accuracies_vs_clients):
        print(f"{clients:<10} | {acc:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(client_counts, accuracies_vs_clients, marker='o', label='Accuracy')
    plt.xlabel('Number of Clients', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16) 
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('Accuracy vs Number of Clients', fontsize=16)
    plt.grid(True)
    plt.savefig('results/accuracy_vs_clients.png')
    plt.show()
    
    # Experiment 3: Accuracy vs Sample Size
    print("\n=== Experiment: Accuracy vs Sample Size ===")
    total_samples = len(X_train_tensor)
    sample_sizes = [int(total_samples * i / 10) for i in range(1, 11)]
    accuracies_vs_sample_size = []
    fixed_num_clients = 5
    
    for sample_size in sample_sizes:
        print(f"Running with sample size {sample_size}")
        monitor = PerformanceMonitor()
        indices = np.random.choice(len(X_train_tensor), sample_size, replace=False)
        X_train_subset = X_train_tensor[indices]
        y_train_subset = y_train_tensor[indices]
        
        client_data = split_data_among_clients(X_train_subset, y_train_subset, fixed_num_clients)
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)
        
        trained_model, round_accuracies, _, _, _, _ = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor
        )
        
        accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
        accuracies_vs_sample_size.append(accuracy)
    
    print("\nAccuracy vs Sample Size")
    print(f"{'Sample Size':<15} | {'Accuracy':<10}")
    print("-" * 28)
    for size, acc in zip(sample_sizes, accuracies_vs_sample_size):
        print(f"{size:<15} | {acc:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, accuracies_vs_sample_size, marker='o', label='Accuracy')
    plt.xlabel('Sample Size', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('Accuracy vs Sample Size', fontsize=13)
    plt.grid(True)
    plt.savefig('results/accuracy_vs_sample_size.png')
    plt.show()