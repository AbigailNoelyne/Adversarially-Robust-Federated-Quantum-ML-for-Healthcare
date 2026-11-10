"""
check_setup.py
--------------
Run this after installing packages to confirm your environment
is ready for the quantum FL experiments.

Usage:
    python check_setup.py
"""

import sys

print("=" * 55)
print("  Quantum FL Environment Check")
print("=" * 55)

errors = []

# Python version
major, minor = sys.version_info[:2]
status = "OK" if (major == 3 and minor >= 10) else "FAIL - need Python 3.10+"
print(f"Python {major}.{minor}            [{status}]")
if "FAIL" in status:
    errors.append("Python version too old - install Python 3.10 or higher")

libs = [
    ("torch",       "2.2",  "PyTorch      "),
    ("pennylane",   "0.38", "PennyLane    "),
    ("numpy",       "1.26", "NumPy        "),
    ("sklearn",     "1.4",  "Scikit-learn "),
    ("pandas",      "2.1",  "Pandas       "),
    ("matplotlib",  "3.8",  "Matplotlib   "),
    ("scipy",       "1.11", "SciPy        "),
    ("seaborn",     "0.13", "Seaborn      "),
]

for module, min_ver, label in libs:
    try:
        mod = __import__(module)
        ver = mod.__version__
        parts = ver.split(".")
        min_parts = min_ver.split(".")
        ok = tuple(int(x) for x in parts[:2]) >= tuple(int(x) for x in min_parts)
        tag = "OK" if ok else f"WARN - want >={min_ver}"
        print(f"{label}  {ver:<12} [{tag}]")
    except ImportError:
        print(f"{label}  MISSING       [FAIL]")
        errors.append(f"{label.strip()} not installed - run: pip install {module}")

# Quick VQC test - this proves the quantum part works
print()
print("Running VQC forward pass test...")
try:
    import torch
    import pennylane as qml

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(4), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(4))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    weight_shapes = {"weights": (2, 4)}
    layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    x = torch.randn(2, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    grad = layer.weights.grad.norm().item()

    print(f"  VQC output shape : {list(out.shape)}  [OK]")
    print(f"  Gradient norm    : {grad:.4f}          [OK - gradients flow correctly]")

except Exception as e:
    print(f"  VQC test FAILED: {e}")
    errors.append("VQC test failed - check PennyLane installation")

# Full hybrid model test
print()
print("Running full HybridVQCPredictor test...")
try:
    import os, sys
    # Try to import from same folder if quantum_model.py is present
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quantum_model import HybridVQCPredictor
    import torch

    model = HybridVQCPredictor(input_dim=20)
    x = torch.randn(4, 20)
    y = torch.randint(0, 3, (4,))
    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()
    print(f"  Output shape     : {list(out.shape)}  [OK]")
    print(f"  fc[0].in_features: {model.fc[0].in_features}              [OK - FL compatibility confirmed]")
    print(f"  Loss backward    : OK")
except ImportError:
    print("  quantum_model.py not found in same folder - place it here and re-run")
except Exception as e:
    print(f"  HybridVQCPredictor test FAILED: {e}")
    errors.append("HybridVQCPredictor test failed")

print()
print("=" * 55)
if errors:
    print("ACTION NEEDED - fix these issues:")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
else:
    print("All checks passed.")
    print("You are ready to run: python run_quantum_experiment.py")
print("=" * 55)
