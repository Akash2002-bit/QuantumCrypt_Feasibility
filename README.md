# QuantumCrypt_Feasibility

# Artifact for: From Theory to Hardware Reality: A Systematization of Quantum Cryptanalysis Feasibility

This repository contains the official Qiskit simulation code accompanying the research paper: **"From Theory to Hardware Reality: A Systematization of Quantum Cryptanalysis Feasibility."** This artifact empirically evaluates the depth limitations of near-term quantum hardware by simulating fundamental cryptanalytic algorithms (Shor's and Grover's) under realistic noise constraints. It derives the mathematical "impossibility gap" separating current Noisy Intermediate-Scale Quantum (NISQ) capabilities from practical cryptanalytic relevance.

## 🚀 Key Features

* **Algorithmic Implementations:** * Shor's Algorithm (Quantum Phase Estimation) instances for $N=4, 15, 21$, and $35$, incorporating Linear Nearest-Neighbour (LNN) routing overhead.
  * Grover's Algorithm instances for unstructured search spaces of $n \in \{3, 4, 5, 6\}$ qubits.
* **Dual Noise Models:**
  * **Model A (Optimistic):** Uniform depolarising noise (standard theoretical benchmark).
  * **Model B (Pessimistic/Realistic):** Composite thermal-relaxation ($T_1$/$T_2$) plus depolarising noise, parameterized for current superconducting architectures (e.g., IBM Eagle-class).
* **Impossibility Gap Analysis:** Automatically derives and reports the orders-of-magnitude gap ($>10^{11}$) between empirically observed NISQ depth limits and the logical gate requirements for RSA-2048 and AES-128.

## 📋 Requirements

The code is written in Python 3 and relies on the IBM Qiskit framework. To ensure exact reproducibility of the results presented in the paper, we recommend using the exact versions specified below.

* `python` >= 3.11.5
* `qiskit` == 2.4.0
* `qiskit-aer` == 0.17.2
* `matplotlib` >= 3.10.8
* `numpy` >= 2.4.4

### Installation
We recommend setting up a virtual environment before installing the dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv quantum_env
source quantum_env/bin/activate  # On Windows use: quantum_env\Scripts\activate

# Install dependencies
pip install qiskit==2.4.0 qiskit-aer==0.17.2 matplotlib numpy
