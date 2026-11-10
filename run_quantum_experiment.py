"""
run_quantum_experiment.py  —  FULL VERSION WITH ALL FIGURES
-------------------------------------------------------------
Generates every figure needed for the paper:

  Fig 1  — accuracy_comparison.png         (accuracy per round, 3 conditions)
  Fig 2  — roc_comparison.png              (security detection ROC, 3 conditions)
  Fig 3  — attack_detection_bar.png        (per-attack detection rate bar chart)
  Fig 4  — confusion_matrix_runN.png       (confusion matrix per run)
  Fig 5  — roc_curves_runN.png             (per-class clinical ROC per run)
  Fig 6  — attack_detection_roc_runN.png   (per-attack ROC per run)
  Fig 7  — accuracy_per_run.png            (accuracy across rounds, all 3 runs)
  Fig 8  — performance_metrics.png         (latency bar chart per run)
  Fig 9  — scalability_metrics.png         (round time vs client count)
  Fig 10 — accuracy_vs_clients.png         (final accuracy vs client count)

Run:
    python run_quantum_experiment.py
    # Mac (prevent sleep):
    caffeinate -i python run_quantum_experiment.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH   = PROJECT_ROOT / "data" / "liver_cirrhosis.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "quantum"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from data_preprocessing import load_and_preprocess_data, split_data_among_clients
from evaluation import evaluate_model
from performance import PerformanceMonitor

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3})

COND_COLORS = {
    "Classical FL + defense (baseline)": "#2196F3",
    "Quantum FL — no defense":           "#FF9800",
    "Quantum FL + defense (proposed)":   "#4CAF50",
}

NUM_CLIENTS = 10
NUM_RUNS    = 3
MAX_ROUNDS  = 50
PATIENCE    = 5


# ── save helper ──────────────────────────────────────────────────────────────
def _save(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── core runner ───────────────────────────────────────────────────────────────
def run_experiment(condition_name, use_quantum, enable_defense,
                   X_train, y_train, X_test, y_test, run_id=0):
    print(f"\n{'='*65}\n  {condition_name}  |  Run {run_id + 1}\n{'='*65}")

    if use_quantum:
        from quantum_model import CirrhosisPredictor
        from federated_learning_quantum import federated_learning_with_early_stopping
    else:
        from model import CirrhosisPredictor
        from federated_learning import federated_learning_with_early_stopping

    monitor      = PerformanceMonitor()
    client_data  = split_data_among_clients(X_train, y_train, NUM_CLIENTS)
    global_model = CirrhosisPredictor(X_train.shape[1])

    trained_model, round_accs, sec_true, sec_pred, sec_scores, atk_metrics = \
        federated_learning_with_early_stopping(
            global_model, client_data, X_test, y_test,
            max_rounds=MAX_ROUNDS, patience=PATIENCE,
            enable_defense=enable_defense, monitor=monitor,
        )

    final_acc  = evaluate_model(trained_model, X_test, y_test)
    report     = monitor.generate_report()
    sec_report = report["security"]
    perf       = report["performance"]
    fp, tn     = sec_report["fp"], sec_report["tn"]
    fper       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  Accuracy: {final_acc:.4f}  ADR: {sec_report['detection_rate']:.4f}  "
          f"FPER: {fper:.4f}  Round: {perf['avg_round_latency']:.2f}s")

    import torch
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_test)
        probs   = outputs.numpy()
        preds   = outputs.argmax(dim=1).numpy()

    return {
        "accuracy":          final_acc,
        "adr":               sec_report["detection_rate"],
        "fper":              fper,
        "avg_round_latency": perf["avg_round_latency"],
        "avg_agg_latency":   perf["avg_aggregation_latency"],
        "avg_val_latency":   perf["avg_validation_latency"],
        "round_accuracies":  round_accs,
        "security_true":     sec_true,
        "security_pred":     sec_pred,
        "security_scores":   sec_scores,
        "attack_metrics":    atk_metrics,
        "y_true":            y_test.numpy(),
        "y_pred":            preds,
        "y_prob":            probs,
    }


# ── Fig 1 — accuracy per round (all conditions) ───────────────────────────────
def fig_accuracy_comparison(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, results), style in zip(all_results.items(), ["-o","-s","-^"]):
        accs    = [r["round_accuracies"] for r in results]
        min_len = min(len(a) for a in accs)
        arr     = np.array([a[:min_len] for a in accs])
        mean, std = arr.mean(0), arr.std(0)
        x = range(1, min_len + 1)
        col = COND_COLORS.get(name)
        ax.plot(x, mean, style, label=name, lw=2, markersize=5, color=col)
        ax.fill_between(x, mean-std, mean+std, alpha=0.12, color=col)
    ax.set_xlabel("Federation round", fontsize=13)
    ax.set_ylabel("Test accuracy", fontsize=13)
    ax.set_title("Accuracy per round: classical vs quantum FL", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.set_ylim(0, 1.0)
    fig.tight_layout(); _save(fig, "accuracy_comparison.png")


# ── Fig 2 — security detection ROC (all conditions) ──────────────────────────
def fig_roc_comparison(all_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, results in all_results.items():
        all_true   = np.concatenate([np.array(r["security_true"])  for r in results])
        all_scores = np.concatenate([np.array(r["security_scores"]) for r in results])
        col = COND_COLORS.get(name)
        if len(np.unique(all_true)) < 2:
            ax.plot([0,1],[0,1], lw=2, label=f"{name} (AUC=0.50)", color=col)
        else:
            fpr, tpr, _ = roc_curve(all_true, all_scores)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.2f})", color=col)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False positive rate", fontsize=13)
    ax.set_ylabel("True positive rate",  fontsize=13)
    ax.set_title("Security detection ROC: classical vs quantum FL", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); fig.tight_layout(); _save(fig, "roc_comparison.png")


# ── Fig 3 — per-attack detection bar chart ───────────────────────────────────
def fig_attack_detection_bar(all_results):
    atypes = ["data_poisoning","model_poisoning","backdoor","mitm"]
    labels = [a.replace("_"," ").title() for a in atypes]
    x, w   = np.arange(len(atypes)), 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, results) in enumerate(all_results.items()):
        rates = []
        for at in atypes:
            tv = np.concatenate([np.array(r["attack_metrics"][at]["true"]) for r in results])
            pv = np.concatenate([np.array(r["attack_metrics"][at]["pred"]) for r in results])
            tp, fn = np.sum(tv & pv), np.sum(tv & ~pv)
            rates.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
        ax.bar(x+(i-1)*w, rates, w, label=name, alpha=0.85, color=COND_COLORS.get(name))
    ax.set_xlabel("Attack type", fontsize=13); ax.set_ylabel("Detection rate", fontsize=13)
    ax.set_title("Per-attack detection rate: classical vs quantum FL", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0,1.05); ax.legend(fontsize=11)
    fig.tight_layout(); _save(fig, "attack_detection_bar.png")


# ── Fig 4 — confusion matrices (per run, quantum+defense) ────────────────────
def fig_confusion_matrices(qd_results):
    cnames = ["Stage 1", "Stage 2", "Stage 3"]
    for i, r in enumerate(qd_results):
        cm  = confusion_matrix(r["y_true"], r["y_pred"])
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cnames, yticklabels=cnames, ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — Quantum FL + Defense  (Run {i+1})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(); _save(fig, f"confusion_matrix_run{i+1}.png")


# ── Fig 5 — per-class clinical ROC (per run, quantum+defense) ────────────────
def fig_roc_per_class(qd_results):
    cnames = ["Stage 1","Stage 2","Stage 3"]
    cols   = ["#2196F3","#FF9800","#4CAF50"]
    for i, r in enumerate(qd_results):
        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in range(r["y_prob"].shape[1]):
            y_bin = (r["y_true"] == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, r["y_prob"][:,cls])
            ax.plot(fpr, tpr, lw=2, label=f"{cnames[cls]} (AUC={auc(fpr,tpr):.2f})", color=cols[cls])
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curves — All Cirrhosis Stages  (Run {i+1})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11); fig.tight_layout(); _save(fig, f"roc_curves_run{i+1}.png")


# ── Fig 6 — per-attack ROC (per run, quantum+defense) ────────────────────────
def fig_attack_roc_per_run(qd_results):
    atypes = ["data_poisoning","model_poisoning","backdoor","mitm"]
    alabels= ["Data Poisoning","Model Poisoning","Backdoor","MitM"]
    acols  = ["#2196F3","#4CAF50","#F44336","#00BCD4"]
    for i, r in enumerate(qd_results):
        fig, ax = plt.subplots(figsize=(8, 6))
        for at, lbl, col in zip(atypes, alabels, acols):
            tv = np.array(r["attack_metrics"][at]["true"])
            sc = np.array(r["attack_metrics"][at]["scores"])
            if len(np.unique(tv)) < 2: continue
            fpr, tpr, _ = roc_curve(tv, sc)
            ax.plot(fpr, tpr, lw=2, label=f"{lbl} (AUC={auc(fpr,tpr):.2f})", color=col)
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curves for Attack Detection  (Run {i+1})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11); fig.tight_layout()
        _save(fig, f"attack_detection_roc_run{i+1}.png")


# ── Fig 7 — accuracy across rounds all 3 runs (quantum+defense) ──────────────
def fig_accuracy_per_run(qd_results):
    cols = ["#2196F3","#FF9800","#4CAF50"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(qd_results):
        accs = r["round_accuracies"]
        ax.plot(range(1, len(accs)+1), accs, "-o",
                label=f"Run {i+1}", color=cols[i], lw=2, markersize=5)
    ax.set_xlabel("Federation Round", fontsize=13)
    ax.set_ylabel("Test Accuracy",    fontsize=13)
    ax.set_title("Accuracy Across Rounds — Quantum FL + Defense  (3 Runs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12); ax.set_ylim(0, 1.0)
    fig.tight_layout(); _save(fig, "accuracy_per_run.png")


# ── Fig 8 — performance latency (quantum+defense) ────────────────────────────
def fig_performance_metrics(qd_results):
    run_labels = [f"Run {i+1}" for i in range(len(qd_results))]
    agg  = [r["avg_agg_latency"]   for r in qd_results]
    val  = [r["avg_val_latency"]   for r in qd_results]
    rnd  = [r["avg_round_latency"] for r in qd_results]
    x, w = np.arange(len(run_labels)), 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x-w, agg, w, label="Aggregation Latency", color="#42A5F5", alpha=0.9)
    ax.bar(x,   val, w, label="Validation Latency",  color="#66BB6A", alpha=0.9)
    ax.bar(x+w, rnd, w, label="Average Round Time",  color="#EF5350", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Test Run", fontsize=13); ax.set_ylabel("Time (seconds, log scale)", fontsize=13)
    ax.set_title("Performance Metrics Across Test Runs — Quantum FL + Defense",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(run_labels, fontsize=12)
    ax.legend(fontsize=11); fig.tight_layout(); _save(fig, "performance_metrics.png")


# ── Fig 9 — scalability sweep ────────────────────────────────────────────────
def fig_scalability(X_train, y_train, X_test, y_test):
    from quantum_model import CirrhosisPredictor
    from federated_learning_quantum import federated_learning_with_early_stopping
    counts = list(range(2, 21, 2))
    agg_l, val_l, rnd_l = [], [], []
    print("\nScalability sweep...")
    for n in counts:
        print(f"  n={n}", end="", flush=True)
        mon  = PerformanceMonitor()
        cd   = split_data_among_clients(X_train, y_train, n)
        mdl  = CirrhosisPredictor(X_train.shape[1])
        federated_learning_with_early_stopping(
            mdl, cd, X_test, y_test,
            max_rounds=2, patience=99, enable_defense=True, monitor=mon,
        )
        p = mon.generate_report()["performance"]
        agg_l.append(p["avg_aggregation_latency"])
        val_l.append(p["avg_validation_latency"])
        rnd_l.append(p["avg_round_latency"])
        print(f" → {p['avg_round_latency']:.1f}s")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(counts, agg_l, "-o", lw=2, label="Aggregation Latency", color="#42A5F5")
    ax.plot(counts, val_l, "-o", lw=2, label="Validation Latency",  color="#FF9800")
    ax.plot(counts, rnd_l, "-o", lw=2, label="Average Round Time",  color="#4CAF50")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Clients", fontsize=13)
    ax.set_ylabel("Time (s)",          fontsize=13)
    ax.set_title("Scalability Metrics vs Number of Clients — Quantum FL + Defense",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); fig.tight_layout(); _save(fig, "scalability_metrics.png")


# ── Fig 10 — accuracy vs client count ────────────────────────────────────────
def fig_accuracy_vs_clients(X_train, y_train, X_test, y_test):
    from quantum_model import CirrhosisPredictor
    from federated_learning_quantum import federated_learning_with_early_stopping
    counts, accs = list(range(2, 21, 2)), []
    print("\nAccuracy-vs-clients sweep...")
    for n in counts:
        print(f"  n={n}", end="", flush=True)
        mon = PerformanceMonitor()
        cd  = split_data_among_clients(X_train, y_train, n)
        mdl = CirrhosisPredictor(X_train.shape[1])
        trained, *_ = federated_learning_with_early_stopping(
            mdl, cd, X_test, y_test,
            max_rounds=10, patience=3, enable_defense=True, monitor=mon,
        )
        acc = evaluate_model(trained, X_test, y_test)
        accs.append(acc); print(f" → {acc:.4f}")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(counts, accs, "-o", lw=2, color="#4CAF50", markersize=8)
    ax.set_xlabel("Number of Clients", fontsize=13)
    ax.set_ylabel("Accuracy",          fontsize=13)
    ax.set_title("Final Accuracy vs Number of Clients — Quantum FL + Defense",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0); fig.tight_layout(); _save(fig, "accuracy_vs_clients.png")


# ── comparison table ─────────────────────────────────────────────────────────
def print_comparison_table(all_results):
    print("\n" + "="*80)
    print(f"{'Condition':<38} {'Accuracy':>12} {'ADR':>8} {'FPER':>8} {'Round(s)':>10}")
    print("="*80)
    for name, metrics in all_results.items():
        acc  = np.mean([r["accuracy"] for r in metrics])
        adr  = np.mean([r["adr"]      for r in metrics])
        fper = np.mean([r["fper"]     for r in metrics])
        lat  = np.mean([r["avg_round_latency"] for r in metrics])
        std  = np.std( [r["accuracy"] for r in metrics])
        print(f"{name:<38} {acc:.4f}±{std:.4f}  {adr:.4f}  {fper:.4f}  {lat:>8.2f}s")
    print("="*80)
    print("ADR = Attack Detection Rate | FPER = False-Positive Exclusion Rate")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    X_train, y_train, X_test, y_test = load_and_preprocess_data(str(DATA_PATH))
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    conditions = {
        "Classical FL + defense (baseline)": {"use_quantum": False, "enable_defense": True},
        "Quantum FL — no defense":           {"use_quantum": True,  "enable_defense": False},
        "Quantum FL + defense (proposed)":   {"use_quantum": True,  "enable_defense": True},
    }

    all_results = {name: [] for name in conditions}
    for cond_name, cfg in conditions.items():
        for run_id in range(NUM_RUNS):
            result = run_experiment(
                cond_name, cfg["use_quantum"], cfg["enable_defense"],
                X_train, y_train, X_test, y_test, run_id,
            )
            all_results[cond_name].append(result)

    print_comparison_table(all_results)

    qd = all_results["Quantum FL + defense (proposed)"]

    print("\nGenerating figures...")
    fig_accuracy_comparison(all_results)   # Fig 1
    fig_roc_comparison(all_results)        # Fig 2
    fig_attack_detection_bar(all_results)  # Fig 3
    fig_confusion_matrices(qd)             # Fig 4
    fig_roc_per_class(qd)                  # Fig 5
    fig_attack_roc_per_run(qd)             # Fig 6
    fig_accuracy_per_run(qd)               # Fig 7
    fig_performance_metrics(qd)            # Fig 8

    print("\nRunning scalability sweeps (extra ~15 min)...")
    fig_scalability(X_train, y_train, X_test, y_test)        # Fig 9
    fig_accuracy_vs_clients(X_train, y_train, X_test, y_test) # Fig 10

    print(f"\nAll figures saved to: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  {f.name}")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
