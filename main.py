from data_preprocessing import load_and_preprocess_data
from experiments import run_federated_experiments, run_scalability_experiment, run_accuracy_experiments
from pathlib import Path 

data_folder = Path(__file__).resolve().parent.parent / 'data'
FILE_PATH = data_folder / 'liver_cirrhosis.csv'

# FILE_PATH = '../data/liver_cirrhosis.csv'

def main():
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(FILE_PATH)
    run_federated_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_runs=3, num_clients=20)
    run_scalability_experiment(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, client_counts=range(2, 21, 2))
    run_accuracy_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, FILE_PATH)

if __name__ == "__main__":
    main()