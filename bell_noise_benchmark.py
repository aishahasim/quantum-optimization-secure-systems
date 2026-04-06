"""
Bell State Noise Model Benchmarking
=====================================
Extends the Bell state simulation with a depolarizing noise model
to simulate real NISQ hardware imperfections.

Runs each Bell state under 4 conditions:
  - Ideal (no noise)
  - Low noise    (p1=0.001, p2=0.005)  — near-term best hardware
  - Medium noise (p1=0.005, p2=0.02)   — typical NISQ device
  - High noise   (p1=0.02,  p2=0.05)   — noisy / older hardware

Outputs:
  - Console table of fidelity scores per state per noise level
  - Side-by-side bar chart comparison
  - bell_noise_benchmark.png
  - bell_noise_results.json

Requirements:
    pip install qiskit qiskit-aer matplotlib

Run:
    python bell_noise_benchmark.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ── Configuration ──────────────────────────────────────────────────────────────
SHOTS = 4096

# Noise levels: (label, single-qubit error rate, two-qubit error rate)
NOISE_LEVELS = [
    ("Ideal",   0.000, 0.000),
    ("Low",     0.001, 0.005),
    ("Medium",  0.005, 0.020),
    ("High",    0.020, 0.050),
]

BELL_COLORS = {
    "|Φ+⟩": "#7C6CF5",
    "|Φ-⟩": "#5BB8F5",
    "|Ψ+⟩": "#4ECBA0",
    "|Ψ-⟩": "#F5875B",
}

# Ideal output distributions (state → expected probs)
IDEAL_DIST = {
    "|Φ+⟩": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
    "|Φ-⟩": {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
    "|Ψ+⟩": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0},
    "|Ψ-⟩": {"00": 0.0, "01": 0.5, "10": 0.5, "11": 0.0},
}


# ── Bell State Circuits ────────────────────────────────────────────────────────

def make_bell_circuits():
    circuits = []

    qc = QuantumCircuit(2, 2, name="|Φ+⟩")
    qc.h(0); qc.cx(0, 1); qc.measure([0,1],[0,1])
    circuits.append(qc)

    qc = QuantumCircuit(2, 2, name="|Φ-⟩")
    qc.h(0); qc.cx(0, 1); qc.z(0); qc.measure([0,1],[0,1])
    circuits.append(qc)

    qc = QuantumCircuit(2, 2, name="|Ψ+⟩")
    qc.h(0); qc.cx(0, 1); qc.x(1); qc.measure([0,1],[0,1])
    circuits.append(qc)

    qc = QuantumCircuit(2, 2, name="|Ψ-⟩")
    qc.h(0); qc.cx(0, 1); qc.x(1); qc.z(0); qc.measure([0,1],[0,1])
    circuits.append(qc)

    return circuits


# ── Noise Model Builder ────────────────────────────────────────────────────────

def build_noise_model(p1, p2):
    """
    Depolarizing noise model:
      p1 = single-qubit gate error rate (applied to H, X, Z)
      p2 = two-qubit gate error rate   (applied to CX/CNOT)

    Depolarizing error: with probability p, apply a random Pauli
    (I, X, Y, Z) uniformly — simulates decoherence and gate imperfection.
    """
    if p1 == 0 and p2 == 0:
        return None   # Ideal: no noise model

    nm = NoiseModel()
    # Single-qubit gate errors
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "x", "z"])
    # Two-qubit gate error (CNOT is the noisiest gate on real hardware)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


# ── Fidelity Metric ────────────────────────────────────────────────────────────

def compute_fidelity(counts, ideal_dist, shots):
    """
    Classical fidelity F = Σ sqrt(p_ideal(x) * p_measured(x))
    Ranges from 0 (completely wrong) to 1.0 (perfect match).
    This is the Bhattacharyya coefficient — a standard benchmark metric.
    """
    states = ["00", "01", "10", "11"]
    fidelity = 0.0
    for s in states:
        p_meas = counts.get(s, 0) / shots
        p_ideal = ideal_dist.get(s, 0.0)
        fidelity += np.sqrt(p_meas * p_ideal)
    return round(fidelity, 4)


# ── Run All Benchmarks ─────────────────────────────────────────────────────────

def run_benchmarks(circuits):
    """
    Run each Bell circuit under each noise level.
    Returns nested dict: results[noise_label][circuit_name] = counts
    """
    all_results = {}
    fidelities  = {}

    for label, p1, p2 in NOISE_LEVELS:
        print(f"\n  [{label} noise]  p1={p1}, p2={p2}")
        noise_model = build_noise_model(p1, p2)
        backend = AerSimulator()

        all_results[label] = {}
        fidelities[label]  = {}

        for qc in circuits:
            transpiled = transpile(qc, backend, optimization_level=1)
            job = backend.run(transpiled, shots=SHOTS,
                              noise_model=noise_model)
            counts = job.result().get_counts()
            all_results[label][qc.name] = counts

            fid = compute_fidelity(counts, IDEAL_DIST[qc.name], SHOTS)
            fidelities[label][qc.name] = fid
            print(f"    {qc.name}  fidelity={fid:.4f}  counts={counts}")

    return all_results, fidelities


# ── Print Fidelity Table ───────────────────────────────────────────────────────

def print_fidelity_table(fidelities, circuits):
    names  = [qc.name for qc in circuits]
    labels = [nl[0] for nl in NOISE_LEVELS]

    col = 12
    print(f"\n{'='*52}")
    print("  FIDELITY BENCHMARK TABLE")
    print(f"{'='*52}")
    header = f"  {'Noise':<10}" + "".join(f"{n:>{col}}" for n in names)
    print(header)
    print("  " + "-" * (10 + col * len(names)))
    for label in labels:
        row = f"  {label:<10}"
        for name in names:
            f = fidelities[label][name]
            row += f"{f:>{col}.4f}"
        print(row)
    print()


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_benchmark(all_results, fidelities, circuits):
    names  = [qc.name for qc in circuits]
    labels = [nl[0] for nl in NOISE_LEVELS]
    states = ["00", "01", "10", "11"]

    fig = plt.figure(figsize=(16, 10), facecolor="#0F0F14")
    fig.suptitle(
        f"Bell State Noise Benchmarking  |  AerSimulator  |  {SHOTS} shots",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    # Layout: 2 rows × 4 cols (top = bar charts, bottom = fidelity comparison)
    gs = gridspec.GridSpec(2, 4, hspace=0.55, wspace=0.3,
                           height_ratios=[2, 1])

    noise_palette = {
        "Ideal":  "#ffffff",
        "Low":    "#4ECBA0",
        "Medium": "#F5C842",
        "High":   "#F5875B",
    }

    # ── Top row: per-Bell-state grouped bars ──
    for col_idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0, col_idx])
        color = BELL_COLORS[name]

        x = np.arange(len(states))
        width = 0.18
        offsets = np.linspace(-(len(labels)-1)/2, (len(labels)-1)/2, len(labels))

        for i, label in enumerate(labels):
            counts = all_results[label][name]
            probs  = [counts.get(s, 0) / SHOTS for s in states]
            nc     = noise_palette[label]
            alpha  = 0.9 if label == "Ideal" else 0.75
            ax.bar(x + offsets[i] * width, probs,
                   width=width, color=nc, alpha=alpha,
                   label=label, edgecolor="#00000040", linewidth=0.3)

        ax.set_facecolor("#1A1A24")
        ax.set_title(name, color=color, fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"|{s}⟩" for s in states], fontsize=8)
        ax.set_ylim(0, 0.72)
        ax.set_ylabel("Probability", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#cccccc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

        # Ideal 50% reference
        ax.axhline(0.5, color="white", linewidth=0.5,
                   linestyle="--", alpha=0.2)

        if col_idx == 0:
            ax.legend(loc="upper right", fontsize=7,
                      framealpha=0.3, labelcolor="white",
                      facecolor="#1A1A24", edgecolor="#333344")

    # ── Bottom row: fidelity line chart ──
    ax_fid = fig.add_subplot(gs[1, :])
    ax_fid.set_facecolor("#1A1A24")

    x_fid = np.arange(len(labels))
    for name in names:
        fids = [fidelities[label][name] for label in labels]
        ax_fid.plot(x_fid, fids, marker="o", linewidth=2,
                    markersize=6, label=name,
                    color=BELL_COLORS[name], alpha=0.9)
        # Annotate last point
        ax_fid.annotate(f"{fids[-1]:.3f}",
                        xy=(x_fid[-1], fids[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=BELL_COLORS[name], fontsize=8)

    ax_fid.set_xticks(x_fid)
    ax_fid.set_xticklabels(labels, color="#cccccc", fontsize=9)
    ax_fid.set_ylabel("Fidelity (Bhattacharyya)", color="#aaaaaa", fontsize=9)
    ax_fid.set_ylim(0.5, 1.05)
    ax_fid.axhline(1.0, color="white", linewidth=0.5,
                   linestyle="--", alpha=0.2, label="Perfect fidelity")
    ax_fid.tick_params(colors="#cccccc", labelsize=8)
    for spine in ax_fid.spines.values():
        spine.set_edgecolor("#333344")
    ax_fid.legend(loc="lower left", fontsize=8, framealpha=0.3,
                  labelcolor="white", facecolor="#1A1A24",
                  edgecolor="#333344", ncol=5)
    ax_fid.set_title("Fidelity vs noise level  (1.0 = perfect)",
                     color="#aaaaaa", fontsize=9, pad=6)

    plt.savefig("bell_noise_benchmark.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved → bell_noise_benchmark.png")
    plt.show()


# ── Export JSON ────────────────────────────────────────────────────────────────

def export_json(all_results, fidelities):
    export = {
        "config": {"shots": SHOTS, "noise_levels": NOISE_LEVELS},
        "fidelities": fidelities,
        "counts": all_results
    }
    with open("bell_noise_results.json", "w") as f:
        json.dump(export, f, indent=2)
    print("  Saved → bell_noise_results.json")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    circuits = make_bell_circuits()

    print(f"\n{'='*52}")
    print(f"  BELL STATE NOISE BENCHMARKING  ({SHOTS} shots)")
    print(f"{'='*52}")

    all_results, fidelities = run_benchmarks(circuits)

    print_fidelity_table(fidelities, circuits)

    print("  Generating visualization...")
    plot_benchmark(all_results, fidelities, circuits)

    export_json(all_results, fidelities)
