"""
QAOA Max-Cut Optimization
==========================
Quantum Approximate Optimization Algorithm (QAOA) applied to the
Max-Cut problem on a small weighted graph.

Max-Cut problem:
  Given a graph G=(V,E), partition nodes into two sets S and S̄
  such that the number (or total weight) of edges crossing the
  partition is maximized.

  Industrial relevance:
    - Network partitioning (telecoms, logistics)
    - Portfolio diversification (Mercedes-Benz supply chain, ExxonMobil asset scheduling)
    - VLSI circuit design, traffic flow optimization

QAOA overview:
  QAOA alternates between two parameterized unitaries:
    - U_C(γ): Problem/cost unitary — encodes the Max-Cut objective
    - U_B(β): Mixer unitary     — explores the solution space
  Parameters (γ, β) are classically optimized to maximize ⟨C⟩,
  the expected cut value. p=1 layer used here (NISQ-friendly).

Graph used (4 nodes, 5 edges):
      0 ── 1
      |  × |
      3 ── 2
  Edges: (0,1), (1,2), (2,3), (3,0), (0,2)  weight=1 each
  Optimal cut: {0,2} vs {1,3} → cut value = 4

Requirements:
    pip install qiskit qiskit-aer scipy matplotlib networkx

Run:
    python qaoa_maxcut.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── Graph Definition ───────────────────────────────────────────────────────────

# 4-node graph as edge list (node_i, node_j, weight)
EDGES = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 2, 1)]
N_NODES = 4
SHOTS = 4096
P_LAYERS = 1       # QAOA depth p=1 (2 parameters: gamma, beta)
BACKEND = AerSimulator()

# ── Cut Value Calculator ───────────────────────────────────────────────────────

def cut_value(bitstring, edges):
    """
    Compute the cut value for a given bitstring partition.
    Bitstring '0110' means nodes 0,3 in set S=0, nodes 1,2 in set S=1.
    An edge (i,j) contributes its weight if nodes i,j are in different sets.
    """
    bits = [int(b) for b in bitstring]
    return sum(w for i, j, w in edges if bits[i] != bits[j])


def expected_cut(counts, edges, shots):
    """Compute ⟨C⟩ = weighted average cut value over all measurement outcomes."""
    total = sum(cut_value(bs, edges) * cnt for bs, cnt in counts.items())
    return total / shots


# ── QAOA Circuit Builder ───────────────────────────────────────────────────────

def build_qaoa_circuit(gamma, beta, edges, n_nodes, p=1):
    """
    Build a p-layer QAOA circuit for Max-Cut.

    Initial state: |+⟩^⊗n  (equal superposition via H on all qubits)

    Cost unitary U_C(γ):
      For each edge (i,j): apply RZZ(2γ) = CX · RZ(2γ) · CX
      This encodes the edge cost into relative phases.

    Mixer unitary U_B(β):
      For each qubit: apply RX(2β)
      This drives transitions between computational basis states.
    """
    qc = QuantumCircuit(n_nodes, n_nodes)

    # Initial state: uniform superposition
    for q in range(n_nodes):
        qc.h(q)

    for layer in range(p):
        g = gamma[layer] if hasattr(gamma, '__len__') else gamma
        b = beta[layer]  if hasattr(beta,  '__len__') else beta

        # ── Cost unitary U_C(γ) ──
        qc.barrier(label=f"U_C γ={g:.3f}")
        for i, j, w in edges:
            # RZZ(2γw) gate decomposition: CX - RZ(2γw) - CX
            qc.cx(i, j)
            qc.rz(2 * g * w, j)
            qc.cx(i, j)

        # ── Mixer unitary U_B(β) ──
        qc.barrier(label=f"U_B β={b:.3f}")
        for q in range(n_nodes):
            qc.rx(2 * b, q)

    qc.measure(range(n_nodes), range(n_nodes))
    return qc


# ── Classical Optimizer ────────────────────────────────────────────────────────

optimization_history = []   # Track ⟨C⟩ per iteration

def objective(params):
    """
    Objective function for classical optimizer.
    Negated because scipy.minimize minimizes — we want to maximize ⟨C⟩.
    """
    gamma, beta = params[0], params[1]
    qc = build_qaoa_circuit(gamma, beta, EDGES, N_NODES, p=P_LAYERS)
    transpiled = transpile(qc, BACKEND, optimization_level=1)
    counts = BACKEND.run(transpiled, shots=SHOTS).result().get_counts()
    exp_cut = expected_cut(counts, EDGES, SHOTS)
    optimization_history.append(exp_cut)
    return -exp_cut   # Negate to minimize


def run_optimization():
    """Run COBYLA optimizer to find optimal (γ*, β*)."""
    print("\n  Starting classical optimization (COBYLA)...")
    print(f"  {'Iter':<6} {'γ':<10} {'β':<10} {'⟨C⟩':<10}")
    print("  " + "-"*36)

    # Initial guess: γ=π/4, β=π/8 (standard QAOA starting point)
    x0 = [np.pi / 4, np.pi / 8]

    result = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": 50, "rhobeg": 0.5}
    )

    gamma_opt, beta_opt = result.x
    print(f"\n  Optimal γ = {gamma_opt:.4f}")
    print(f"  Optimal β = {beta_opt:.4f}")
    print(f"  Best ⟨C⟩  = {-result.fun:.4f}  (max possible = 4)")
    print(f"  Approximation ratio = {-result.fun / 4:.4f}")
    return gamma_opt, beta_opt, result


# ── Final Sampling ─────────────────────────────────────────────────────────────

def sample_optimal(gamma_opt, beta_opt):
    """Run the optimized circuit with more shots to get final distribution."""
    qc = build_qaoa_circuit(gamma_opt, beta_opt, EDGES, N_NODES)
    transpiled = transpile(qc, BACKEND, optimization_level=1)
    counts = BACKEND.run(transpiled, shots=8192).result().get_counts()

    # Annotate each bitstring with its cut value
    annotated = {bs: (cnt, cut_value(bs, EDGES)) for bs, cnt in counts.items()}
    # Sort by cut value descending
    sorted_results = sorted(annotated.items(), key=lambda x: -x[1][1])

    print(f"\n{'='*52}")
    print("  FINAL SAMPLING RESULTS (8192 shots)")
    print(f"{'='*52}")
    print(f"  {'Bitstring':<12} {'Cut val':<10} {'Count':<8} {'Prob':<8}")
    print("  " + "-"*38)
    for bs, (cnt, cv) in sorted_results[:8]:
        prob = cnt / 8192
        marker = " ← optimal" if cv == 4 else ""
        print(f"  {bs:<12} {cv:<10} {cnt:<8} {prob:.3f}{marker}")

    return counts, annotated


# ── Brute Force Classical Comparison ──────────────────────────────────────────

def brute_force_maxcut(edges, n_nodes):
    """Enumerate all 2^n partitions to find the true optimum."""
    best_cut, best_partitions = 0, []
    for mask in range(2**n_nodes):
        bs = format(mask, f'0{n_nodes}b')
        cv = cut_value(bs, edges)
        if cv > best_cut:
            best_cut = cv
            best_partitions = [bs]
        elif cv == best_cut:
            best_partitions.append(bs)
    return best_cut, best_partitions


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_results(counts, annotated, opt_history, gamma_opt, beta_opt):
    fig = plt.figure(figsize=(14, 9), facecolor="#0F0F14")
    fig.suptitle(
        f"QAOA Max-Cut  |  4-node graph  |  p=1  |  γ*={gamma_opt:.3f}, β*={beta_opt:.3f}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.35,
                           height_ratios=[1.4, 1])

    # ── Panel 1: Graph drawing (manual layout) ──
    ax_g = fig.add_subplot(gs[0, 0])
    ax_g.set_facecolor("#1A1A24")
    ax_g.set_xlim(-0.3, 1.3); ax_g.set_ylim(-0.3, 1.3)
    ax_g.set_aspect("equal"); ax_g.axis("off")
    ax_g.set_title("Problem graph", color="#aaaaaa", fontsize=10, pad=6)

    pos = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
    node_colors = {0: "#7C6CF5", 1: "#F5875B", 2: "#7C6CF5", 3: "#F5875B"}
    # Optimal cut: {0,2} vs {1,3}

    for i, j, w in EDGES:
        x0, y0 = pos[i]; x1, y1 = pos[j]
        cut_edge = (node_colors[i] != node_colors[j])
        lc = "#4ECBA0" if cut_edge else "#555566"
        lw = 2.5 if cut_edge else 1.0
        ax_g.plot([x0, x1], [y0, y1], color=lc, linewidth=lw,
                  zorder=1, linestyle="-" if cut_edge else "--")

    for n, (x, y) in pos.items():
        ax_g.scatter(x, y, s=320, color=node_colors[n],
                     zorder=3, edgecolors="white", linewidths=1.2)
        ax_g.text(x, y, str(n), ha="center", va="center",
                  color="white", fontsize=11, fontweight="bold", zorder=4)

    ax_g.text(0.5, -0.22, "Green = cut edges (optimal partition)",
              transform=ax_g.transAxes, ha="center",
              color="#4ECBA0", fontsize=8, style="italic")
    ax_g.text(0.5, -0.32, "Purple = set S={0,2}  Orange = set S̄={1,3}",
              transform=ax_g.transAxes, ha="center",
              color="#888899", fontsize=8)

    # ── Panel 2: Optimization convergence ──
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_conv.set_facecolor("#1A1A24")
    ax_conv.set_title("Optimization convergence", color="#aaaaaa", fontsize=10, pad=6)

    iters = range(1, len(opt_history) + 1)
    ax_conv.plot(iters, opt_history, color="#7C6CF5", linewidth=1.8,
                 alpha=0.9, label="⟨C⟩ per iter")
    ax_conv.axhline(4.0, color="#4ECBA0", linewidth=1, linestyle="--",
                    alpha=0.6, label="Classical optimum = 4")
    ax_conv.axhline(max(opt_history), color="#F5875B", linewidth=0.8,
                    linestyle=":", alpha=0.7,
                    label=f"QAOA best = {max(opt_history):.3f}")

    ax_conv.set_xlabel("Iteration", color="#aaaaaa", fontsize=9)
    ax_conv.set_ylabel("Expected cut ⟨C⟩", color="#aaaaaa", fontsize=9)
    ax_conv.tick_params(colors="#cccccc", labelsize=8)
    for sp in ax_conv.spines.values(): sp.set_edgecolor("#333344")
    ax_conv.legend(fontsize=7, framealpha=0.3, labelcolor="white",
                   facecolor="#1A1A24", edgecolor="#333344")

    # ── Panel 3: Cut value distribution ──
    ax_cv = fig.add_subplot(gs[0, 2])
    ax_cv.set_facecolor("#1A1A24")
    ax_cv.set_title("Cut value distribution", color="#aaaaaa", fontsize=10, pad=6)

    cv_buckets = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_shots = sum(counts.values())
    for bs, cnt in counts.items():
        cv_buckets[cut_value(bs, EDGES)] += cnt

    cvs   = list(cv_buckets.keys())
    probs = [cv_buckets[v] / total_shots for v in cvs]
    bar_colors = ["#E24B4A", "#888780", "#888780", "#F5C842", "#4ECBA0"]
    bars = ax_cv.bar(cvs, probs, color=bar_colors, alpha=0.85,
                     edgecolor="white", linewidth=0.4, width=0.6)

    for bar, prob in zip(bars, probs):
        if prob > 0.005:
            ax_cv.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.005,
                       f"{prob:.1%}", ha="center", va="bottom",
                       color="white", fontsize=8, fontweight="bold")

    ax_cv.set_xlabel("Cut value", color="#aaaaaa", fontsize=9)
    ax_cv.set_ylabel("Probability", color="#aaaaaa", fontsize=9)
    ax_cv.set_xticks(cvs)
    ax_cv.tick_params(colors="#cccccc", labelsize=8)
    for sp in ax_cv.spines.values(): sp.set_edgecolor("#333344")
    ax_cv.text(4, probs[4] + 0.02, "optimal",
               ha="center", color="#4ECBA0", fontsize=7)

    # ── Panel 4: Top bitstrings bar chart ──
    ax_bs = fig.add_subplot(gs[1, :2])
    ax_bs.set_facecolor("#1A1A24")
    ax_bs.set_title("Top measurement outcomes (8192 shots)",
                    color="#aaaaaa", fontsize=10, pad=6)

    top = sorted(annotated.items(), key=lambda x: -x[1][0])[:10]
    labels = [bs for bs, _ in top]
    vals   = [cnt / 8192 for bs, (cnt, cv) in top]
    clrs   = ["#4ECBA0" if annotated[bs][1] == 4 else "#7C6CF5" for bs in labels]

    bars2 = ax_bs.bar(labels, vals, color=clrs, alpha=0.85,
                      edgecolor="white", linewidth=0.4, width=0.6)
    for bar, v in zip(bars2, vals):
        ax_bs.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.002,
                   f"{v:.2%}", ha="center", va="bottom",
                   color="white", fontsize=7)

    ax_bs.set_xlabel("Bitstring (node assignment)", color="#aaaaaa", fontsize=9)
    ax_bs.set_ylabel("Probability", color="#aaaaaa", fontsize=9)
    ax_bs.tick_params(colors="#cccccc", labelsize=8)
    for sp in ax_bs.spines.values(): sp.set_edgecolor("#333344")

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#4ECBA0", label="Cut = 4 (optimal)"),
                  Patch(facecolor="#7C6CF5", label="Sub-optimal")]
    ax_bs.legend(handles=legend_els, fontsize=7, framealpha=0.3,
                 labelcolor="white", facecolor="#1A1A24",
                 edgecolor="#333344")

    # ── Panel 5: Summary metrics ──
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.set_facecolor("#1A1A24")
    ax_sum.axis("off")
    ax_sum.set_title("Benchmark summary", color="#aaaaaa", fontsize=10, pad=6)

    best_cv   = max(opt_history)
    approx_r  = best_cv / 4.0
    opt_prob  = sum(cnt for bs, (cnt, cv) in annotated.items()
                    if cv == 4) / 8192

    metrics = [
        ("Classical optimum",   "4 edges"),
        ("QAOA ⟨C⟩",           f"{best_cv:.3f}"),
        ("Approximation ratio", f"{approx_r:.4f}"),
        ("P(optimal solution)", f"{opt_prob:.2%}"),
        ("Circuit depth p",     "1"),
        ("Qubits used",         "4"),
        ("Optimizer",           "COBYLA"),
        ("Total shots",         f"{SHOTS} + 8192"),
    ]

    for idx, (k, v) in enumerate(metrics):
        y = 0.92 - idx * 0.115
        ax_sum.text(0.02, y, k, transform=ax_sum.transAxes,
                    color="#888899", fontsize=9)
        ax_sum.text(0.98, y, v, transform=ax_sum.transAxes,
                    color="white", fontsize=9, fontweight="bold",
                    ha="right")

    plt.savefig("qaoa_maxcut_results.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved → qaoa_maxcut_results.png")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*52}")
    print("  QAOA MAX-CUT OPTIMIZATION")
    print(f"{'='*52}")
    print(f"  Graph: {N_NODES} nodes, {len(EDGES)} edges")
    print(f"  QAOA depth: p={P_LAYERS}")

    # Classical brute force reference
    best_cut, best_partitions = brute_force_maxcut(EDGES, N_NODES)
    print(f"\n  Classical optimum : cut = {best_cut}")
    print(f"  Optimal partitions: {best_partitions}")

    # QAOA optimization
    gamma_opt, beta_opt, result = run_optimization()

    # Final sampling with optimal parameters
    counts, annotated = sample_optimal(gamma_opt, beta_opt)

    # Visualize
    print("\n  Generating visualization...")
    plot_results(counts, annotated, optimization_history, gamma_opt, beta_opt)
