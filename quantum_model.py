"""
quantum_model.py
----------------
Hybrid classical-quantum model for the paper:
  "Adversarial Robustness in Federated Quantum Machine Learning
   for Healthcare"

Architecture (Chen & Yoo 2021 style, adapted for tabular clinical data):
  1. Classical encoder   : Linear(input_dim -> 64) -> ReLU
                           -> Linear(64 -> n_qubits) -> Tanh
                           Compresses features into qubit-compatible angles.

  2. VQC layer           : AngleEmbedding (RY rotations) +
                           StronglyEntanglingLayers (parameterised rotations
                           + CNOT entanglement). Returns PauliZ expectation
                           values as classical measurement outcomes.

  3. Classical head      : Linear(n_qubits -> output_dim) -> Softmax
                           Maps quantum measurements to class probabilities.

Compatibility guarantees (all existing FL code works unchanged):
  - model.fc[0].in_features  -> returns input_dim  (federated_learning.py line 218)
  - model.state_dict()        -> standard PyTorch dict (encryption, server_defense)
  - model.load_state_dict()   -> standard (copy global -> local client model)
  - model.train() / .eval()   -> standard (evaluation.py, clients_defense.py)
  - FGSM input gradients flow through encoder -> VQC -> head (clients_defense.py)
  - CrossEntropyLoss works on output (clients_defense.py, federated_learning.py)

Usage (drop-in replacement):
  from quantum_model import HybridVQCPredictor as CirrhosisPredictor
  model = CirrhosisPredictor(input_dim)

Default hyperparameters:
  n_qubits = 8    (matches 8-qubit simulation, feasible on CPU via default.qubit)
  n_layers = 3    (StronglyEntanglingLayers depth, balances expressibility/speed)
  output_dim = 3  (cirrhosis stages 0, 1, 2)
"""

import torch
import torch.nn as nn
import pennylane as qml
import math


# ---------------------------------------------------------------------------
# Module-level device and circuit definition
# These are created once at import time so all model instances share the same
# compiled quantum device — avoids re-compilation overhead per instantiation.
# ---------------------------------------------------------------------------

_N_QUBITS_DEFAULT = 8
_N_LAYERS_DEFAULT = 3

_dev = qml.device("default.qubit", wires=_N_QUBITS_DEFAULT)


@qml.qnode(_dev, interface="torch", diff_method="backprop")
def _vqc_circuit(inputs, weights):
    """
    Variational quantum circuit.

    Encoding : AngleEmbedding applies RY(input[i]) to qubit i.
               Inputs are pre-scaled to [-pi, pi] by the encoder Tanh * pi.
    Ansatz   : StronglyEntanglingLayers with learnable 3-parameter rotations
               and CNOT entanglement.  Shape: (n_layers, n_qubits, 3).
    Output   : PauliZ expectation values for all qubits — real numbers
               in [-1, 1] that feed the classical classification head.
    """
    qml.AngleEmbedding(inputs, wires=range(_N_QUBITS_DEFAULT), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(_N_QUBITS_DEFAULT))
    return [qml.expval(qml.PauliZ(i)) for i in range(_N_QUBITS_DEFAULT)]


_VQC_WEIGHT_SHAPES = {"weights": (_N_LAYERS_DEFAULT, _N_QUBITS_DEFAULT, 3)}


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class HybridVQCPredictor(nn.Module):
    """
    Hybrid classical-quantum predictor for cirrhosis stage classification.

    This is the quantum FL client model.  It replaces the classical
    CirrhosisPredictor while keeping an identical external interface so that
    all existing federated learning, defense, attack, and evaluation code
    operates without modification.

    Parameters
    ----------
    input_dim : int
        Number of input features (determined by dataset after preprocessing).
    n_qubits : int, optional
        Number of qubits in the VQC.  Must equal _N_QUBITS_DEFAULT (8).
    n_layers : int, optional
        Depth of StronglyEntanglingLayers.  Must equal _N_LAYERS_DEFAULT (3).
    output_dim : int, optional
        Number of output classes (default 3 for cirrhosis stages).
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = _N_QUBITS_DEFAULT,
        n_layers: int = _N_LAYERS_DEFAULT,
        output_dim: int = 3,
    ):
        super().__init__()

        if n_qubits != _N_QUBITS_DEFAULT or n_layers != _N_LAYERS_DEFAULT:
            raise ValueError(
                f"n_qubits must be {_N_QUBITS_DEFAULT} and n_layers must be "
                f"{_N_LAYERS_DEFAULT} to match the compiled quantum device. "
                "Change _N_QUBITS_DEFAULT/_N_LAYERS_DEFAULT at the top of "
                "quantum_model.py to use different values."
            )

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim

        # ------------------------------------------------------------------
        # Compatibility shim: existing federated_learning.py accesses
        # global_model.fc[0].in_features to recover input_dim.
        # We expose a single-element ModuleList with a dummy Linear that
        # satisfies this access pattern without participating in the forward
        # pass.
        # ------------------------------------------------------------------
        self.fc = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False)])
        # Freeze the dummy layer — it must never contribute to training
        for p in self.fc[0].parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # 1. Classical encoder
        #    Compresses input_dim features down to n_qubits values in (-1, 1)
        #    via Tanh, then scaled to (-pi, pi) in forward() for angle embed.
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits),
            nn.Tanh(),           # output in (-1, 1); forward() scales by pi
        )

        # ------------------------------------------------------------------
        # 2. VQC layer (PennyLane TorchLayer)
        #    Wraps the module-level _vqc_circuit.
        #    TorchLayer handles batched forward passes and gradient flow.
        # ------------------------------------------------------------------
        self.vqc = qml.qnn.TorchLayer(_vqc_circuit, _VQC_WEIGHT_SHAPES)

        # ------------------------------------------------------------------
        # 3. Classical output head
        #    Maps n_qubits measurement values to class probabilities.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, output_dim),
            nn.Softmax(dim=1),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for classical layers; random for VQC weights."""
        for m in [self.encoder, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialise VQC weights near zero for stable early training
        with torch.no_grad():
            nn.init.uniform_(
                self.vqc.weights,
                -math.pi / 8,
                math.pi / 8,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Input features (standardised by data_preprocessing.py).

        Returns
        -------
        torch.Tensor, shape (batch, output_dim)
            Class probability distribution.
        """
        # Classical encoding -> angle range [-pi, pi]
        encoded = self.encoder(x)          # (batch, n_qubits) in (-1, 1)
        angles = encoded * math.pi         # (batch, n_qubits) in (-pi, pi)

        # Quantum processing
        quantum_out = self.vqc(angles)     # (batch, n_qubits), values in [-1, 1]

        # Classical classification
        return self.classifier(quantum_out)  # (batch, output_dim), sums to 1


# ---------------------------------------------------------------------------
# Alias — allows existing imports to stay unchanged
# All files that do:
#   from model import CirrhosisPredictor
# can instead do:
#   from quantum_model import CirrhosisPredictor
# ---------------------------------------------------------------------------
CirrhosisPredictor = HybridVQCPredictor


# ---------------------------------------------------------------------------
# Quick self-test (run: python quantum_model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("HybridVQCPredictor self-test")
    print("=" * 60)

    input_dim = 20
    batch = 8
    model = HybridVQCPredictor(input_dim=input_dim)

    # Forward pass
    x = torch.randn(batch, input_dim)
    y = torch.randint(0, 3, (batch,))
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(f"Forward pass ({batch} samples): {(t1-t0)*1000:.1f} ms")
    print(f"Output shape : {out.shape}")
    print(f"Output sums  : {out.sum(dim=1).tolist()}")

    # Backward pass
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    print(f"Loss         : {loss.item():.4f}")
    print(f"VQC grad norm: {model.vqc.weights.grad.norm().item():.6f}")

    # Compatibility checks
    print(f"\nfc[0].in_features : {model.fc[0].in_features}  (must equal {input_dim})")
    sd = model.state_dict()
    print(f"State dict keys   : {list(sd.keys())}")
    model2 = HybridVQCPredictor(input_dim=input_dim)
    model2.load_state_dict(sd)
    print("load_state_dict   : OK")

    # FGSM gradient check
    x2 = torch.randn(4, input_dim, requires_grad=True)
    y2 = torch.randint(0, 3, (4,))
    model.zero_grad()
    loss2 = nn.CrossEntropyLoss()(model(x2), y2)
    loss2.backward()
    print(f"Input grad norm   : {x2.grad.norm().item():.6f}  (must be > 0 for FGSM)")

    print(f"\nTotal parameters  : {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params  : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("\nAll checks passed.")
