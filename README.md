# Quantum Computing for Optimization & Secure Systems

**Self-Initiated Project | March 2026**  
Designed and implemented quantum circuits using Qiskit in NISQ-based environments, with a focus on quantum optimization, post-quantum cryptography (PQC), and quantum key distribution (QKD).

---

## Project Overview

This project demonstrates practical quantum computing across three domains:

| Domain | Script | Key Output |
|--------|--------|------------|
| Quantum circuits & entanglement | `bell_states_sim.py` | All 4 Bell states, measurement histograms |
| NISQ noise benchmarking | `bell_noise_benchmark.py` | Fidelity vs noise level, JSON export |
| Quantum optimization (QAOA) | `qaoa_maxcut.py` | Max-Cut on 4-node graph, approximation ratio |

Industry case studies covering IBM, RIKEN, Cleveland Clinic, Basque Quantum, Mercedes-Benz, and ExxonMobil are documented in [`docs/technical_report.docx`](docs/).

---

## Repository Structure

```
quantum-optimization-secure-systems/
├── bell_states_sim.py          # Bell state circuits + visualization
├── bell_noise_benchmark.py     # Depolarizing noise model benchmarking
├── qaoa_maxcut.py              # QAOA Max-Cut optimization
├── requirements.txt            # Python dependencies
├── docs/
│   └── technical_report.docx  # Full technical report + pilot roadmap
└── README.md
```

---

## Results Summary

### Bell States (bell_states_sim.py)
- Simulated all 4 Bell states: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
- Verified quantum entanglement: correlated and anti-correlated measurement outcomes
- Demonstrated that phase differences (|Φ+⟩ vs |Φ-⟩) are invisible to direct measurement — foundational to QKD eavesdropping detection

### Noise Benchmarking (bell_noise_benchmark.py)

| Noise Level | Single-qubit error | CX error | Avg fidelity |
|-------------|-------------------|----------|--------------|
| Ideal       | 0%                | 0%       | 1.0000       |
| Low         | 0.1%              | 0.5%     | 0.9989       |
| Medium      | 0.5%              | 2%       | 0.9945       |
| High        | 2%                | 5%       | 0.9816       |

Key finding: |Ψ-⟩ degrades fastest (deepest circuit depth) — directly relevant to circuit optimization on real hardware.

### QAOA Max-Cut (qaoa_maxcut.py)

| Metric | Value |
|--------|-------|
| Problem | Max-Cut on 4-node, 5-edge graph |
| Classical optimum | 4 edges |
| QAOA ⟨C⟩ (p=1) | 3.174 |
| Approximation ratio | 0.7935 |
| P(optimal solution) | 46.74% |
| Optimizer | COBYLA |

Optimal partition found: {0,2} vs {1,3} — bitstrings `0101` / `1010` at ~23-24% probability each.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/quantum-optimization-secure-systems.git
cd quantum-optimization-secure-systems

# Create and activate virtual environment
python -m venv qiskit_env

# Windows
qiskit_env\Scripts\activate.bat

# macOS / Linux
source qiskit_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Scripts

```bash
# 1. Bell state simulation
python bell_states_sim.py

# 2. Noise model benchmarking
python bell_noise_benchmark.py

# 3. QAOA optimization
python qaoa_maxcut.py
```

Each script saves a `.png` visualization and the noise benchmark also exports `bell_noise_results.json`.

---

## Key Concepts

**Bell States** — maximally entangled two-qubit states. Measurement of one qubit instantly determines the other, regardless of distance. Foundation of QKD protocols (BB84, E91).

**Depolarizing Noise** — models real NISQ hardware imperfections. With probability *p*, a random Pauli error (X, Y, or Z) is applied. Benchmarked here using Bhattacharyya fidelity.

**QAOA** — Quantum Approximate Optimization Algorithm (Farhi et al., 2014). Alternates between a cost unitary U_C(γ) encoding the problem and a mixer unitary U_B(β) exploring solutions. Parameters are classically optimized.

**Max-Cut** — NP-hard combinatorial optimization problem. Relevant to network partitioning, portfolio diversification, and supply chain scheduling.

---

## Industry Relevance

| Company | Quantum Application |
|---------|-------------------|
| IBM | Quantum-safe cryptography migration, Qiskit ecosystem |
| RIKEN | Quantum networking and QKD infrastructure (Japan) |
| Cleveland Clinic | Quantum-accelerated drug discovery and molecular simulation |
| Basque Quantum | Regional quantum network deployment, QKD testbeds |
| Mercedes-Benz | Supply chain optimization using quantum annealing |
| ExxonMobil | Asset scheduling and route optimization with QAOA |

---

## References

- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. *arXiv:1411.4028*
- NIST PQC Standardization (2022). Selected algorithms: CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, SPHINCS+
- IBM Quantum (2024). *Qiskit Documentation*. https://docs.quantum.ibm.com
- Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing. *Proceedings of IEEE ICCS*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
