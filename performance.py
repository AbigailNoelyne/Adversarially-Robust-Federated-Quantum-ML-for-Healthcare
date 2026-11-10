import time
import numpy as np
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'security': defaultdict(list),
            'performance': defaultdict(list),
            'scalability': defaultdict(list)
        }
        self.timers = {}
        self.attack_ground_truth = {}
        self.client_counts = []
        self.data_sizes = []
        self.round_timestamps = []
        self.detected_attacks = {  # New dictionary to count detected attacks by type
            'data_poisoning': 0,
            'model_poisoning': 0,
            'backdoor': 0,
            'mitm': 0
        }

    def start_timer(self, name):
        self.timers[name] = time.time()

    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers.pop(name)
            self.metrics['performance'][f'{name}_latency'].append(elapsed)

    def record_scalability_metrics(self, num_clients, data_size):
        self.client_counts.append(num_clients)
        self.data_sizes.append(data_size)
        self.round_timestamps.append(len(self.client_counts))

    def calculate_scalability_metrics(self):
        time_metrics = []
        client_growth = []
        
        for i, ts in enumerate(self.round_timestamps):
            if ts-1 < len(self.metrics['performance'].get('round_latency', [])):
                time_metrics.append(self.metrics['performance']['round_latency'][ts-1])
                client_growth.append(self.client_counts[i])
        
        min_length = min(len(client_growth), len(time_metrics))
        client_growth = client_growth[:min_length]
        time_metrics = time_metrics[:min_length]
        
        if len(client_growth) < 2 or len(time_metrics) < 2:
            return {
                'client_efficiency': 0,
                'data_efficiency': 0
            }
        
        try:
            client_eff = np.polyfit(client_growth, time_metrics, 1)[0]
            data_eff = np.polyfit(np.log(self.data_sizes[:min_length]), 
                                time_metrics[:min_length], 1)[0]
        except Exception as e:
            print(f"Scalability metric error: {str(e)}")
            client_eff = data_eff = 0
            
        return {
            'client_efficiency': client_eff,
            'data_efficiency': data_eff
        }
    
    def record_security_event(self, client_id, is_attack, detected, attack_type='none'):
        self.metrics['security']['true_positives'].append(is_attack and detected)
        self.metrics['security']['false_positives'].append(not is_attack and detected)
        self.metrics['security']['true_negatives'].append(not is_attack and not detected)
        self.metrics['security']['false_negatives'].append(is_attack and not detected)
        if is_attack and detected and attack_type in self.detected_attacks:  # Count detected attacks by type
            self.detected_attacks[attack_type] += 1

    def calculate_security_metrics(self):
        tp = sum(self.metrics['security']['true_positives'])
        fp = sum(self.metrics['security']['false_positives'])
        tn = sum(self.metrics['security']['true_negatives'])
        fn = sum(self.metrics['security']['false_negatives'])
        
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': detection_rate
        }
    
    def calculate_performance_metrics(self):
        # Handle empty lists gracefully, returning 0 if no data
        def safe_mean(data):
            return np.mean(data) if data else 0.0
        
        return {
            'avg_aggregation_latency': safe_mean(self.metrics['performance'].get('aggregation_latency', [])),
            'avg_validation_latency': safe_mean(self.metrics['performance'].get('validation_latency', [])),
            'avg_round_latency': safe_mean(self.metrics['performance'].get('round_latency', []))  # Changed to match timer name
        }
    
    def generate_report(self):
        return {
            'security': self.calculate_security_metrics(),
            'performance': self.calculate_performance_metrics(),
            'scalability': self.calculate_scalability_metrics(),
            'detected_attacks': self.detected_attacks
        }

    def print_detailed_report(self):
        report = self.generate_report()
        
        print("\n=== Security Metrics ===")
        print(f"Attack Detection Rate: {report['security']['detection_rate']:.2%}")
        print(f"False Positive Rate: {report['security']['false_positive_rate']:.2%}")
        print(f"Precision: {report['security']['precision']:.2%}")
        print(f"Recall: {report['security']['recall']:.2%}")
        print()
        print(f"True Positives (TP): {report['security']['tp']}")
        print(f"False Positives (FP): {report['security']['fp']}")
        print(f"True Negatives (TN): {report['security']['tn']}")
        print(f"False Negatives (FN): {report['security']['fn']}")

        print("\nDetected Attack Counts:")
        for attack_type, count in report['detected_attacks'].items():
            print(f"{attack_type.replace('_', ' ').title()}: {count}")
        
        print("\n=== Performance Metrics ===")
        print(f"Aggregation Latency: {report['performance']['avg_aggregation_latency']:.4f}s")
        print(f"Validation Latency: {report['performance']['avg_validation_latency']:.4f}s")
        print(f"Average Round Time: {report['performance']['avg_round_latency']:.4f}s")  # Updated label
        
        print("\n=== Scalability Metrics ===")
        print(f"Client Efficiency Slope: {report['scalability']['client_efficiency']:.4f}")
        print(f"Data Efficiency Slope: {report['scalability']['data_efficiency']:.4f}")