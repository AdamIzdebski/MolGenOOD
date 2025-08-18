""" Plot synthetic OOD benchmark results. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_benchmark_results(
    results: dict,
    title=None,
    figsize=(8, 4.5),
    dpi=300,
    show=True
):
    """
    Publication-ready grouped bar chart:
    - Group 1: AUPRC (All, IID, OOD, Synth) in blue gradient light→dark
    - Group 2: Tanimoto Similarity to Train (All, IID, OOD, Synth) in yellow-orange gradient light→dark
    Subticks indicate All, IID, OOD, Synth for each group.
    Accepts scalars or 1D numpy arrays (mean ± std computed automatically for arrays).
    """

    def mean_std(x):
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr), 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    # Compute means and stds
    auprc_vals, auprc_errs = zip(*[mean_std(v) for v in (results["auprc_iid"], results["auprc_ood"], results["auprc_synthetic"])])
    sim_vals, sim_errs = zip(*[mean_std(v) for v in (results["similarities_iid"], results["similarities_ood"], results["similarities_synthetic"])])

    # Publication-ready colors (light → dark in each group)
    auprc_colors = ["#9ecae1", "#4292c6", "#08519c"]  # light to dark blue
    sim_colors = ["#fec44f", "#fe9929", "#d95f0e"]    # light yellow to dark orange

    # Positions
    n_sub = len(auprc_vals)
    group_gap = 1
    auprc_x = np.arange(n_sub)
    sim_x = np.arange(n_sub) + n_sub + group_gap

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    # Style
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_axisbelow(True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

    # Plot bars
    ax1.bar(auprc_x, auprc_vals, yerr=auprc_errs, capsize=3, width=0.8,
            linewidth=0.8, edgecolor='black', color=auprc_colors)
    ax2.bar(sim_x, sim_vals, yerr=sim_errs, capsize=3, width=0.8,
            linewidth=0.8, edgecolor='black', color=sim_colors)

    # X-ticks: subticks for each group
    xticks = list(auprc_x) + list(sim_x)
    sublabels = ['IID', 'OOD', 'Synthetic'] * 2
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(sublabels, fontsize=9)

    # Main group labels
    group_centers = [np.mean(auprc_x), np.mean(sim_x)]
    ax1.set_xticks(group_centers, minor=True)
    ax1.set_xticklabels(['AUPRC', 'Tanimoto Similarity to Train'], minor=True, fontsize=10, weight='bold')
    ax1.tick_params(axis='x', which='minor', pad=25)

    # Labels and scales
    if title:
        ax1.set_title(title, fontsize=12, pad=12)
    ax1.set_ylabel('AUPRC', fontsize=11)
    ax2.set_ylabel('Tanimoto Similarity', fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    # Legend
    legend_patches = [
        Patch(facecolor=auprc_colors[i], edgecolor='black', label=f"AUPRC ({lbl})")
        for i, lbl in enumerate(['All Data', 'IID', 'OOD', 'Synthetic'])
    ] + [
        Patch(facecolor=sim_colors[i], edgecolor='black', label=f"Sim. ({lbl})")
        for i, lbl in enumerate(['All Data', 'IID', 'OOD', 'Synthetic'])
    ]
    # ax1.legend(handles=legend_patches, frameon=False, fontsize=8, loc='upper left', ncol=2)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig, ax1, ax2
