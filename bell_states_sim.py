"""
Bell State Quantum Circuit Simulation
======================================
Simulates all 4 Bell states using Qiskit's AerSimulator (NISQ backend).
Outputs measurement counts and a bar chart for each Bell state.

Requirements:
    pip install qiskit qiskit-aer matplotlib

Run:
    python bell_states_sim.py
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ─────────────────────────────────────────────────────────────
SHOTS = 1024          # Number of measurement shots per circuit
BACKEND = AerSimulator()   # NISQ-style noisy simulator (shot-based)

# ── Bell State Circuit Builders ────────────────────────────────────────────────

def bell_phi_plus():
    """
    |Φ+⟩ = (|00⟩ + |11⟩) / √2
    Both qubits start in |0⟩. H on q0 creates superposition,
    CNOT entangles. Expected: ~50% |00⟩, ~50% |11⟩
    """
    qc = QuantumCircuit(2, 2, name="|Φ+⟩")
    qc.h(0)          # Hadamard: |0⟩ → (|0⟩ + |1⟩)/√2
    qc.cx(0, 1)      # CNOT: entangle qubits
    qc.measure([0, 1], [0, 1])
    return qc


def bell_phi_minus():
    """
    |Φ-⟩ = (|00⟩ - |11⟩) / √2
    Z gate on q0 adds a phase flip after entanglement.
    Expected: ~50% |00⟩, ~50% |11⟩ (same counts, different phase)
    """
    qc = QuantumCircuit(2, 2, name="|Φ-⟩")
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)          # Phase flip: introduces relative minus sign
    qc.measure([0, 1], [0, 1])
    return qc


def bell_psi_plus():
    """
    |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    X gate on q1 flips the target qubit, shifting correlations.
    Expected: ~50% |01⟩, ~50% |10⟩
    """
    qc = QuantumCircuit(2, 2, name="|Ψ+⟩")
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)          # Bit flip on q1: flips |00⟩↔|01⟩ and |11⟩↔|10⟩
    qc.measure([0, 1], [0, 1])
    return qc


def bell_psi_minus():
    """
    |Ψ-⟩ = (|01⟩ - |10⟩) / √2
    X on q1 (bit flip) + Z on q0 (phase flip) together.
    Expected: ~50% |01⟩, ~50% |10⟩
    """
    qc = QuantumCircuit(2, 2, name="|Ψ-⟩")
    qc.h(0)
    qc.cx(0, 1)
    qc.x(1)          # Bit flip
    qc.z(0)          # Phase flip
    qc.measure([0, 1], [0, 1])
    return qc


# ── Run Simulation ─────────────────────────────────────────────────────────────

def run_circuits(circuits):
    """Transpile and run all circuits on AerSimulator. Returns counts dict."""
    results = {}
    for qc in circuits:
        transpiled = transpile(qc, BACKEND, optimization_level=1)
        job = BACKEND.run(transpiled, shots=SHOTS)
        result = job.result()
        counts = result.get_counts()
        results[qc.name] = counts
        print(f"  {qc.name}  →  {counts}")
    return results


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_all(results):
    """Plot all 4 Bell state histograms in a 2×2 grid."""

    BELL_COLORS = {
        "|Φ+⟩": "#7C6CF5",   # purple
        "|Φ-⟩": "#5BB8F5",   # blue
        "|Ψ+⟩": "#4ECBA0",   # teal
        "|Ψ-⟩": "#F5875B",   # coral
    }

    EXPECTED = {
        "|Φ+⟩": "Expected: |00⟩ + |11⟩",
        "|Φ-⟩": "Expected: |00⟩ − |11⟩  (same dist.)",
        "|Ψ+⟩": "Expected: |01⟩ + |10⟩",
        "|Ψ-⟩": "Expected: |01⟩ − |10⟩  (same dist.)",
    }

    fig = plt.figure(figsize=(12, 8), facecolor="#0F0F14")
    fig.suptitle(
        "Bell State Measurement Outcomes  |  AerSimulator  |  "
        f"{SHOTS} shots per circuit",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)
    names = list(results.keys())

    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        counts = results[name]

        # Ensure all 4 basis states are shown (fill missing with 0)
        all_states = ["00", "01", "10", "11"]
        values = [counts.get(s, 0) for s in all_states]
        probs = [v / SHOTS for v in values]

        color = BELL_COLORS[name]
        bars = ax.bar(all_states, probs, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5, width=0.55)

        # Annotate bars with percentage
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{prob:.1%}",
                    ha="center", va="bottom", color="white",
                    fontsize=9, fontweight="bold"
                )

        # Ideal 50% reference line
        ax.axhline(0.5, color="white", linewidth=0.6,
                   linestyle="--", alpha=0.3, label="Ideal 50%")

        ax.set_facecolor("#1A1A24")
        ax.set_title(name, color=color, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Measurement outcome", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Probability", color="#aaaaaa", fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

        ax.text(0.5, -0.22, EXPECTED[name], transform=ax.transAxes,
                ha="center", va="top", color="#888899",
                fontsize=8, style="italic")

    plt.savefig("bell_states_results.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved → bell_states_results.png")
    plt.show()


# ── Circuit Diagrams ───────────────────────────────────────────────────────────

def print_circuits(circuits):
    """Print ASCII circuit diagrams for all 4 Bell states."""
    print("\n" + "="*52)
    print("  BELL STATE CIRCUITS")
    print("="*52)
    for qc in circuits:
        print(f"\n  {qc.name}")
        print(qc.draw(output="text", fold=60))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    circuits = [bell_phi_plus(), bell_phi_minus(),
                bell_psi_plus(), bell_psi_minus()]

    print_circuits(circuits)

    print(f"\n{'='*52}")
    print(f"  RUNNING ON AerSimulator  ({SHOTS} shots each)")
    print("="*52)
    results = run_circuits(circuits)

    print("\n  Generating visualization...")
    plot_all(results)
