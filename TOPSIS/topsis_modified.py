###############################################################################
# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

###############################################################################
# Paper-friendly global style (matplotlib only, no seaborn/plotly)

def set_paper_style():
    """A clean, consistent style suitable for papers."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.2,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,   # TrueType fonts in PDF
        "ps.fonttype": 42,
    })

###############################################################################
# Plot 1: Blue/gray ranked bar chart (horizontal)
# Function name requested: rank_columns_1

def rank_columns_1(scores, labels=None, title="TOPSIS Ranking (Closeness Coefficient, Ci)",
                   xlabel="Closeness coefficient (Ci)", show_values=True,
                   save_path=None, show=True):
    """
    Blue/gray horizontal bar chart ranked by scores (descending).

    Parameters
    ----------
    scores : (m,) array-like
        TOPSIS closeness coefficients.
    labels : list[str] or None
        Alternative labels; if None -> a1..am.
    show_values : bool
        Annotate bars with numeric Ci.
    save_path : str or None
        If provided, save figure (recommended: .pdf for papers).
    show : bool
        Whether to display the figure (plt.show()).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    m = scores.shape[0]

    if labels is None:
        labels = [f"a{i+1}" for i in range(m)]
    if len(labels) != m:
        raise ValueError("labels length must match number of alternatives (len(scores)).")

    # Sort descending
    order = np.argsort(scores)[::-1]
    s_sorted = scores[order]
    l_sorted = [labels[i] for i in order]

    best_pos = 0  # after sorting, best is first
    best_color = "#2F5597"   # deep blue
    other_color = "#B7C7E6"  # light blue-gray
    text_color = "#111827"
    grid_color = "#D1D5DB"

    # Figure size: paper-friendly, adapts to number of bars
    fig_h = max(2.6, 0.45 * m + 1.0)
    fig_w = 6.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = np.arange(m)
    colors = [best_color if i == best_pos else other_color for i in range(m)]
    bars = ax.barh(y, s_sorted, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(l_sorted, color=text_color)
    ax.invert_yaxis()  # best at top

    ax.set_xlabel(xlabel, color=text_color)
    ax.set_title(title, color=text_color, pad=10)

    # Subtle x-grid for readability
    ax.xaxis.grid(True, color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional value labels
    if show_values:
        x_max = np.max(s_sorted) if m > 0 else 1.0
        pad = 0.01 * (x_max if x_max > 0 else 1.0)
        for rect, val in zip(bars, s_sorted):
            ax.text(rect.get_width() + pad,
                    rect.get_y() + rect.get_height() / 2,
                    f"{val:.3f}",
                    va="center", ha="left",
                    fontsize=10, color=text_color)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

###############################################################################
# Plot 2: TOPSIS geometry plot (Route 1): S+ vs S- (purple/green)
# Highlight BEST alternative (max Ci) only, arrow + legend explanation.

def topsis_geometry_plot(S_plus, S_minus, best_idx, title="TOPSIS Geometry (S+ vs S-)",
                         save_path=None, show=True, invert_x=True):
    """
    Scatter plot of distances:
      x = S+ (distance to positive ideal)  [smaller is better]
      y = S- (distance to negative ideal)  [larger is better]

    Only the best alternative is annotated (arrow + label).
    """
    S_plus = np.asarray(S_plus, dtype=float).ravel()
    S_minus = np.asarray(S_minus, dtype=float).ravel()
    m = S_plus.shape[0]
    if S_minus.shape[0] != m:
        raise ValueError("S_plus and S_minus must have the same length.")
    if not (0 <= best_idx < m):
        raise ValueError("best_idx out of range.")

    # Purple/green palette
    purple = "#7C3AED"
    green = "#059669"
    neutral = "#9CA3AF"
    text_color = "#111827"
    grid_color = "#E5E7EB"

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # Plot all alternatives (unlabeled)
    ax.scatter(S_plus, S_minus, s=46, c=purple, alpha=0.75,
               edgecolors="white", linewidths=0.6, label="Alternatives")

    # Highlight best alternative (max Ci)
    ax.scatter([S_plus[best_idx]], [S_minus[best_idx]],
               s=130, c=green, edgecolors="white", linewidths=1.0,
               label="Best alternative (max Ci)", zorder=5)

    ax.set_title(title, color=text_color, pad=10)
    ax.set_xlabel("Distance to positive ideal (S+; smaller is better)", color=text_color)
    ax.set_ylabel("Distance to negative ideal (S-; larger is better)", color=text_color)

    # Light grid
    ax.grid(True, color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)

    # Optional inversion so "better" is visually towards upper-left
    if invert_x:
        ax.invert_xaxis()

    # Arrow annotation to best alternative
    best_x, best_y = S_plus[best_idx], S_minus[best_idx]
    ax.annotate(
        "Best alternative",
        xy=(best_x, best_y),
        xytext=(15, 15),
        textcoords="offset points",
        color=text_color,
        fontsize=10,
        arrowprops=dict(arrowstyle="-|>", color=green, lw=1.2),
        ha="left", va="bottom"
    )

    # Legend with an explicit arrow proxy (so legend explains the arrow)
    arrow_proxy = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>",
                                  mutation_scale=12, color=green, lw=1.2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=purple,
               markeredgecolor="white", markersize=7.5, alpha=0.75, label="Alternatives"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=green,
               markeredgecolor="white", markersize=10.0, label="Best alternative (max Ci)"),
        arrow_proxy
    ]
    labels = ["Alternatives", "Best alternative (max Ci)", "Arrow: points to best alternative"]
    ax.legend(handles, labels, frameon=True, facecolor="white", edgecolor="#D1D5DB",
              loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

###############################################################################
# TOPSIS core

def str_extract(t):
    if type(t) == str:
        return t
    elif type(t) == list or type(t) == tuple or type(t) == np.ndarray:
        return t[0]
    else:
        return t


import numpy as np

def topsis_method(dataset, weights, criterion_type, verbose=True):
    """
    Extended TOPSIS method supporting:
    - max (benefit)
    - min (cost)
    - nominal (target-based)
    - interval (range-based)

    Parameters
    ----------
    dataset : array-like, shape (m, n)
        Alternatives x criteria.
    weights : array-like, shape (n,)
        Criteria weights (non-negative).
    criterion_type : list
        Each element is one of:
          - 'max'
          - 'min'
          - ('nominal', target)
          - ('interval', lower, upper)

    Returns
    -------
    c_i : ndarray, shape (m,)
        Closeness coefficients.
    """

    # ---------- Basic checks ----------
    X = np.asarray(dataset, dtype=float)
    if X.ndim != 2:
        raise ValueError("dataset must be 2D.")
    m, n = X.shape

    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n:
        raise ValueError("weights length mismatch.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    w = w / np.sum(w)

    if len(criterion_type) != n:
        raise ValueError("criterion_type length mismatch.")

    # ---------- Vector normalization ----------
    norms = np.linalg.norm(X, axis=0)
    if np.any(np.isclose(norms, 0.0)):
        raise ValueError("Zero-norm criterion column detected.")
    R = X / norms
    V = R * w

    # ---------- Ideal solutions ----------
    A_pos = np.zeros(n)
    A_neg = np.zeros(n)

    for j in range(n):
        ct = criterion_type[j]

        # Case 1: max / min
        if isinstance(ct, str):
            t = ct.strip().lower()
            if t == "max":
                A_pos[j] = np.max(V[:, j])
                A_neg[j] = np.min(V[:, j])
            elif t == "min":
                A_pos[j] = np.min(V[:, j])
                A_neg[j] = np.max(V[:, j])
            else:
                raise ValueError(f"Unknown criterion type: {ct}")

        # Case 2: nominal (target-based)
        elif isinstance(ct, (list, tuple)) and len(ct) == 2:
            kind, target = ct
            if str(kind).lower() != "nominal":
                raise ValueError(f"Invalid nominal definition: {ct}")

            # target must be mapped to normalized-weighted space
            target = float(target)
            target_v = (target / norms[j]) * w[j]

            # Positive ideal = target
            A_pos[j] = target_v

            # Negative ideal = farthest observed value from target
            distances = np.abs(V[:, j] - target_v)
            A_neg[j] = V[np.argmax(distances), j]

        # Case 3: interval (range-based)
        elif isinstance(ct, (list, tuple)) and len(ct) == 3:
            kind, lower, upper = ct
            if str(kind).lower() != "interval":
                raise ValueError(f"Invalid interval definition: {ct}")

            lower, upper = float(lower), float(upper)
            if lower > upper:
                raise ValueError("Interval lower bound > upper bound.")

            lower_v = (lower / norms[j]) * w[j]
            upper_v = (upper / norms[j]) * w[j]

            # Positive ideal = midpoint of interval
            A_pos[j] = 0.5 * (lower_v + upper_v)

            # Negative ideal = farthest observed value from the interval
            distances = np.where(
                V[:, j] < lower_v, lower_v - V[:, j],
                np.where(V[:, j] > upper_v, V[:, j] - upper_v, 0.0)
            )
            A_neg[j] = V[np.argmax(distances), j]

        else:
            raise ValueError(f"Invalid criterion_type entry: {ct}")

    # ---------- Distances ----------
    S_plus = np.linalg.norm(V - A_pos, axis=1)
    S_minus = np.linalg.norm(V - A_neg, axis=1)

    denom = S_plus + S_minus
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Zero denominator in closeness coefficient.")

    c_i = S_minus / denom

    if verbose:
        for i, val in enumerate(c_i, start=1):
            print(f"a{i}: {val:.4f}")
        best = int(np.argmax(c_i))
        print(f"Best alternative: a{best+1}")

    return c_i

###############################################################################
