# Quantum FL Paper — Setup & Running Guide

## What these files are

Three new files that extend your existing thesis codebase for the new conference paper:

| File | Purpose |
|---|---|
| `quantum_model.py` | Drop-in replacement for `model.py` — implements the hybrid VQC |
| `federated_learning_quantum.py` | Copy of `federated_learning.py` with one import changed |
| `run_quantum_experiment.py` | Runs all three conditions and produces comparison figures |

**Your existing files are not touched.** Everything in your thesis codebase stays exactly as it is.

---

## Install PennyLane

```bash
pip install pennylane pennylane-lightning
```

That is the only new dependency. PyTorch, scikit-learn, pandas, matplotlib, seaborn, scipy — all already installed from your thesis work.

---

## File layout

Place the three new files in the same folder as your existing code:

```
your_project/
├── data/
│   └── liver_cirrhosis.csv
├── quantum_model.py              ← NEW
├── federated_learning_quantum.py ← NEW
├── run_quantum_experiment.py     ← NEW
├── model.py                      (unchanged)
├── federated_learning.py         (unchanged)
├── attack_simulation.py          (unchanged)
├── clients_defense.py            (unchanged)
├── server_defense.py             (unchanged)
├── encryption.py                 (unchanged)
├── evaluation.py                 (unchanged)
├── performance.py                (unchanged)
├── data_preprocessing.py         (unchanged)
└── main.py                       (unchanged)
```

---

## Run the experiment

```bash
python run_quantum_experiment.py
```

This runs three conditions automatically:

1. **Classical FL + defense** — your thesis baseline (uses original `model.py`)
2. **Quantum FL — no defense** — VQC clients, no server-side filtering
3. **Quantum FL + defense** — VQC clients with all adaptive defenses active

Results are saved to `results/quantum/`.

---

## What the VQC architecture looks like

```
Input features (input_dim ≈ 20)
        │
        ▼
Classical encoder
  Linear(input_dim → 64) → ReLU → Dropout(0.2)
  Linear(64 → 8) → Tanh              ← compress to 8 values in (-1, 1)
        │
        ▼  multiply by π → angles in (-π, π)
        │
Variational Quantum Circuit (8 qubits)
  AngleEmbedding: RY(angle_i) on qubit i
  StronglyEntanglingLayers × 3: parameterised rotations + CNOT entanglement
  Measurement: PauliZ expectation values on all 8 qubits
        │
        ▼
Classical head
  Linear(8 → 3) → Softmax
        │
        ▼
Class probabilities (3 cirrhosis stages)
```

**Why this architecture:**
- Follows Chen & Yoo (2021) exactly: classical pre-processing → VQC → classical output
- `StronglyEntanglingLayers` provides full entanglement across all qubits
- `AngleEmbedding` is the standard encoding for tabular data
- Tanh in encoder keeps angles bounded (avoids wrapping issues)
- The VQC weight tensor shape is `(3, 8, 3)` = 72 quantum parameters

---

## Timing estimates

On a standard laptop CPU (no GPU needed, quantum simulation runs on CPU):

| Step | Time |
|---|---|
| One client training epoch (200 samples, batch 32) | ~0.4 s |
| One client full training (30 epochs) | ~12 s |
| One FL round (10 clients) | ~2 min |
| One complete experiment (20 rounds, early stopping) | ~35–40 min |
| Three conditions × 3 runs | ~6–7 hours total |

**Tip:** Run overnight. Set `MAX_ROUNDS = 20` and `NUM_RUNS = 1` first to verify everything works, then run the full experiment.

---

## Key research questions your results answer

1. Does quantum FL show higher or lower vulnerability to the 4 attack types compared to classical FL?
2. Do classical adaptive defenses (EMA trust, adaptive trimming, DP scheduler) transfer effectively to protect VQC parameter updates?
3. Does the false-positive exclusion rate (FPER) change when the model is quantum vs classical?
4. What is the latency overhead of quantum client training relative to classical?

These four questions map directly to four paper sections.

---

## Output files

After running `run_quantum_experiment.py`:

```
results/quantum/
├── accuracy_comparison.png    ← accuracy per round for all 3 conditions
├── roc_comparison.png         ← security detection ROC curves
└── attack_detection_bar.png   ← per-attack detection rate grouped bar chart
```

---

## Verifying your VQC model works before running the full experiment

```bash
python quantum_model.py
```

Expected output:
```
HybridVQCPredictor self-test
Forward pass (8 samples): ~25 ms
Output shape: torch.Size([8, 3])
fc[0].in_features: 20
load_state_dict: OK
Input grad norm: >0  (FGSM works)
All checks passed.
```

---

## Citing the quantum architecture

In your paper methods section:

> "Each federated client trains a hybrid classical-quantum model following the architecture of Chen and Yoo [1]. The model consists of a classical encoder that compresses the input features to n=8 dimensions, a variational quantum circuit (VQC) implemented using PennyLane [2] with AngleEmbedding and three layers of StronglyEntanglingLayers on 8 qubits simulated via the `default.qubit` device, and a classical softmax classification head. Quantum gradients are computed via backpropagation through the PennyLane–PyTorch interface."

References to add:
- [1] Chen, S.Y.-C.; Yoo, S. Federated Quantum Machine Learning. *Entropy* 2021, 23, 460.
- [2] Bergholm, V. et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. *arXiv* 2018, arXiv:1811.04968.
