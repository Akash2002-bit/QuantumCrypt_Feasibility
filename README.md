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
```

## ⚙️ Usage & Reproducibility

To execute the simulations and generate the output data and figures, run the main script from your terminal:

```bash
python quantum_sim.py
```

### Reproducibility Note
The script uses a fixed random seed (`SEED = 42`) for both the transpiler (`seed_transpiler`) and the noisy simulator (`seed_simulator`). Combined with `SHOTS = 1024`, running this script will produce the **exact numerical results** and confidence intervals reported in Table II of the paper.

## 📊 Expected Outputs

The script runs in a few minutes on a standard laptop and produces two types of outputs:

### 1. Console Reporting
The script will print detailed tables directly to your terminal. These include:
* **Shor's Algorithm Results:** Circuit depths and success probabilities ($p_s$) under Ideal, Model A, and Model B conditions.
* **Grover's Algorithm Results:** Success probabilities mapped against theoretically optimal iterations ($k^*$).
* **Fidelity Decay Summary:** A side-by-side comparison of empirical results vs. theoretical decay envelopes $(1-\varepsilon)^d$.
* **Impossibility Gap Analysis:** A step-by-step numerical breakdown bridging the empirical NISQ depth collapse ($d^* \approx 10^3$) to cryptanalytic requirements (e.g., $10^{14}$ for RSA-2048).

### 2. High-Resolution Visual Artifacts
The script automatically generates and saves three publication-ready plots to the current directory:
* `shor_histogram.png`: Empirical success probability vs. depth for Shor's algorithm.
* `grover_scaling.png`: Empirical success probability vs. depth for Grover's algorithm.
* `noise_comparison.png`: A side-by-side fidelity decay overlay proving that Model B (composite thermal noise) degrades faster than standard depolarising benchmarks.

## 📖 Citation

If you use this code in your own work or wish to cite the accompanying systematization of knowledge, please use the following citation (update with actual publication details once available):

```bibtex
@inproceedings{angom2026fromtheory,
  title={From Theory to Hardware Reality: A Systematization of Quantum Cryptanalysis Feasibility},
  author={Angom, Akash and Bhattacharjee, Arup and Singh, Laiphrakpam Dolendro},
  booktitle={Proceedings of the [Conference/Workshop Name]},
  year={2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
```

***

### 📄 `requirements.txt`
If you are adding a `requirements.txt` file to your repository alongside the README, here is the updated version matching your exact system environment:

```text
qiskit==2.4.0
qiskit-aer==0.17.2
matplotlib>=3.10.8
numpy>=2.4.4
```
