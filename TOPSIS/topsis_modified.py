###############################################################################
# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings

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
# Function name requested: topis_bars

def topis_bars(scores, labels=None, title="TOPSIS Ranking (Closeness Coefficient, Ci)",
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
# TOPSIS helpers (type parsing + transformations for Route A)

def str_extract(t):
    """Extract the type-name token from criterion_type entries."""
    if isinstance(t, str):
        return t
    elif isinstance(t, (list, tuple, np.ndarray)):
        return t[0]
    else:
        return t


def _raise_invalid_criterion_param(j, msg):
    """Raise a ValueError with criterion column information (0-based and 1-based)."""
    raise ValueError(f"criterion_type error at criterion column j={j} (1-based {j+1}): {msg}")


def _finite_check_or_raise(arr, name="array"):
    """
    Ensure arr contains only finite values. If not, raise ValueError with positions.
    """
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isfinite(arr)
    if np.any(mask):
        idx = np.argwhere(mask)
        # Show at most first 12 positions to keep message readable
        preview = idx[:12].tolist()
        total = int(idx.shape[0])
        raise ValueError(
            f"{name} contains NaN/inf at {total} position(s). "
            f"First positions (row, col): {preview}"
        )
    return arr


def parse_criterion_type(criterion_type, n):
    """
    Parse criterion_type into (ctype_name, params) lists.

    Supported formats per criterion j:
      - 'max' or 'min'
      - ['nominal', target_value]
      - ['interval', a, b]   (a/b order can be swapped)

    Returns
    -------
    ctype_name : list[str] length n
        One of {'max','min','nominal','interval'}.
    params : list[tuple]
        Parameters for nominal/interval; empty tuple for max/min.
        - nominal: (target,)
        - interval: (L, U) with L <= U
    """
    if len(criterion_type) != n:
        raise ValueError("criterion_type length must equal number of criteria (dataset.shape[1]).")

    ctype_name = []
    params = []

    for j in range(n):
        entry = criterion_type[j]
        name = str_extract(entry)
        if not isinstance(name, str):
            _raise_invalid_criterion_param(j, "Type token must be a string like 'max', 'min', 'nominal', 'interval'.")
        name = name.strip().lower()

        if name not in ("max", "min", "nominal", "interval"):
            raise ValueError("criterion_type entries must be: 'max', 'min', ['nominal', target], ['interval', a, b].")

        if name in ("max", "min"):
            ctype_name.append(name)
            params.append(tuple())
            continue

        # From here: nominal/interval must be list/tuple/ndarray
        if not isinstance(entry, (list, tuple, np.ndarray)):
            _raise_invalid_criterion_param(j, f"'{name}' must be provided as a list, e.g. ['{name}', ...].")

        if name == "nominal":
            if len(entry) < 2:
                _raise_invalid_criterion_param(j, "nominal requires ['nominal', target_value].")
            target = entry[1]
            if not isinstance(target, (int, float, np.integer, np.floating)):
                _raise_invalid_criterion_param(j, f"nominal target_value must be numeric, got {type(target)}.")
            ctype_name.append(name)
            params.append((float(target),))
            continue

        if name == "interval":
            if len(entry) < 3:
                _raise_invalid_criterion_param(j, "interval requires ['interval', a, b] where a/b define bounds.")
            a, b = entry[1], entry[2]
            if not isinstance(a, (int, float, np.integer, np.floating)) or not isinstance(b, (int, float, np.integer, np.floating)):
                _raise_invalid_criterion_param(j, f"interval bounds must be numeric, got {type(a)} and {type(b)}.")
            L, U = float(a), float(b)
            if L > U:
                # Per requirement: swap if order is reversed
                L, U = U, L
            ctype_name.append(name)
            params.append((L, U))
            continue

    return ctype_name, params


def distance_to_target(x, target):
    """Absolute distance to a target value (nominal-the-best)."""
    x = np.asarray(x, dtype=float)
    return np.abs(x - float(target))


def distance_to_interval(x, L, U, mode="piecewise"):
    """
    Distance to an interval [L, U].

    mode='piecewise' implements:
      d=0 if x in [L,U]
      d=L-x if x<L
      d=x-U if x>U

    Keeping 'mode' makes it easy to change the penalty function later.
    """
    if mode != "piecewise":
        raise NotImplementedError(f"Unsupported distance mode: {mode}")

    x = np.asarray(x, dtype=float)
    L = float(L)
    U = float(U)

    d = np.zeros_like(x, dtype=float)
    below = x < L
    above = x > U
    d[below] = L - x[below]
    d[above] = x[above] - U
    return d


def distance_to_benefit(d, eps=1e-12):
    """
    Convert a non-negative distance array into a benefit score in [0, 1], where smaller distance is better.

    Mapping (default, easy to modify):
      benefit = 1 - d / (max(d) + eps)

    Special handling (per requirement):
      - If all distances are the same (including all zeros), benefit becomes all-ones,
        and we emit a "constant column" warning upstream.
    """
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return d, True  # edge case

    # If the column has identical distances, treat as equally good for all alternatives.
    if np.allclose(d, d[0]):
        benefit = np.ones_like(d, dtype=float)
        return benefit, True

    dmax = float(np.max(d))
    benefit = 1.0 - (d / (dmax + float(eps)))
    benefit = np.clip(benefit, 0.0, 1.0)

    # This should be rare after the allclose check, but keep it robust.
    is_constant = bool(np.allclose(benefit, benefit[0]))
    return benefit, is_constant


def apply_routeA_transform(X, ctype_name, params, verbose=True):
    """
    Route A: transform nominal/interval criteria into benefit-type columns before TOPSIS.

    - nominal(target): distance = |x - target|
    - interval(L,U): distance = distance_to_interval(x, L, U)

    After transformation, these columns are treated as 'max' in TOPSIS.

    Returns
    -------
    X_eff : ndarray (m, n)
        Transformed dataset.
    ctype_eff : list[str] length n
        Effective types for TOPSIS: only 'max'/'min'.
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape

    X_eff = np.array(X, copy=True, dtype=float)
    ctype_eff = []

    for j in range(n):
        tname = ctype_name[j]

        if tname == "max":
            ctype_eff.append("max")
            continue

        if tname == "min":
            ctype_eff.append("min")
            continue

        if tname == "nominal":
            target = params[j][0]
            d = distance_to_target(X[:, j], target=target)
            benefit, is_constant = distance_to_benefit(d)
            X_eff[:, j] = benefit
            ctype_eff.append("max")

            if is_constant:
                msg = (
                    f"Route A transform produced a constant column for criterion j={j} (1-based {j+1}) "
                    f"[nominal target={target}]. All alternatives are equally scored on this criterion."
                )
                warnings.warn(msg, RuntimeWarning)
                if verbose:
                    print("Warning:", msg)
            continue

        if tname == "interval":
            L, U = params[j]
            d = distance_to_interval(X[:, j], L=L, U=U, mode="piecewise")
            benefit, is_constant = distance_to_benefit(d)
            X_eff[:, j] = benefit
            ctype_eff.append("max")

            if is_constant:
                msg = (
                    f"Route A transform produced a constant column for criterion j={j} (1-based {j+1}) "
                    f"[interval L={L}, U={U}]. All alternatives are equally scored on this criterion."
                )
                warnings.warn(msg, RuntimeWarning)
                if verbose:
                    print("Warning:", msg)
            continue

        # Should never happen due to parsing validation
        _raise_invalid_criterion_param(j, f"Unhandled criterion type: {tname}")

    return X_eff, ctype_eff


###############################################################################
# TOPSIS core

def topsis_method(dataset, weights, criterion_type,
                  plot_bar=False, plot_geom=False,
                  alt_labels=None,
                  verbose=True,
                  save_prefix=None,
                  show_plots=True):
    """
    TOPSIS method (supports max/min + Route A for nominal/interval).

    Parameters
    ----------
    dataset : array-like, shape (m, n)
        Alternatives x criteria.
    weights : array-like, shape (n,)
        Criteria weights (non-negative). Will be normalized to sum to 1.
    criterion_type : list, length n
        Supported per criterion j:
          - 'max' (benefit)
          - 'min' (cost)
          - ['nominal', target_value]          (target-the-best; Route A -> benefit transform)
          - ['interval', a, b]                (interval-the-best; Route A -> benefit transform)
            where a/b order can be swapped.

        Example:
          ['max', ['nominal', 5], ['interval', 213, 23]]

    plot_bar : bool
        Whether to draw the ranked bar chart (topis_bars).
    plot_geom : bool
        Whether to draw the geometry plot (S+ vs S-).
    alt_labels : list[str] or None
        Alternative labels for the bar chart. If None -> a1..am.
    verbose : bool
        Print Ci values and warnings.
    save_prefix : str or None
        If provided, saves plots as PDF:
          f"{save_prefix}_ranking_bar.pdf"
          f"{save_prefix}_topsis_geometry.pdf"
    show_plots : bool
        If True, show plots; if False, only save (when save_prefix is set).

    Returns
    -------
    c_i : ndarray shape (m,)
        Closeness coefficients (higher is better).
    """
    set_paper_style()

    # Dataset validation
    X = np.asarray(dataset, dtype=float)
    if X.ndim != 2:
        raise ValueError("dataset must be a 2D array of shape (m, n).")
    m, n = X.shape

    # Finite check with positions
    _finite_check_or_raise(X, name="dataset")

    # Weights validation
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n:
        raise ValueError("weights length must equal number of criteria (dataset.shape[1]).")
    _finite_check_or_raise(w, name="weights")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if np.allclose(np.sum(w), 0.0):
        raise ValueError("weights sum to zero; provide at least one positive weight.")
    w = w / np.sum(w)  # normalize weights

    # Parse criterion types (supports nominal/interval)
    ctype_name, cparams = parse_criterion_type(criterion_type, n)

    # Route A: transform nominal/interval into benefit-type columns
    X_eff, ctype_eff = apply_routeA_transform(X, ctype_name, cparams, verbose=verbose)

    # Vector normalization (standard TOPSIS)
    norms = np.linalg.norm(X_eff, axis=0)
    if np.any(np.isclose(norms, 0.0)):
        zero_cols = np.where(np.isclose(norms, 0.0))[0].tolist()
        raise ValueError(f"One or more criteria columns have zero norm (all zeros). Columns: {zero_cols}")

    R = X_eff / norms

    # Weighted normalized matrix
    V = R * w

    # Ideal solutions (only max/min remain after Route A)
    A_pos = np.zeros(n, dtype=float)
    A_neg = np.zeros(n, dtype=float)
    for j in range(n):
        if ctype_eff[j] == "max":
            A_pos[j] = np.max(V[:, j])
            A_neg[j] = np.min(V[:, j])
        elif ctype_eff[j] == "min":
            A_pos[j] = np.min(V[:, j])
            A_neg[j] = np.max(V[:, j])
        else:
            # Should never happen: Route A guarantees only max/min here
            _raise_invalid_criterion_param(j, f"Unhandled effective type: {ctype_eff[j]}")

    # Distances to ideals
    S_plus = np.linalg.norm(V - A_pos, axis=1)
    S_minus = np.linalg.norm(V - A_neg, axis=1)

    # Closeness coefficient
    denom = S_plus + S_minus
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Encountered zero denominator in closeness coefficient computation.")
    c_i = S_minus / denom

    # Best alternative (max Ci)
    best_idx = int(np.argmax(c_i))

    if verbose:
        for i, val in enumerate(c_i, start=1):
            print(f"a{i}: {val:.3f}")
        print(f"Best alternative: a{best_idx+1} (max Ci = {c_i[best_idx]:.3f})")

    # Plotting controlled by two booleans
    if alt_labels is None:
        alt_labels = [f"a{i+1}" for i in range(m)]

    if plot_bar:
        bar_save = None
        if save_prefix is not None:
            bar_save = f"{save_prefix}_ranking_bar.pdf"
        topis_bars(
            scores=c_i,
            labels=alt_labels,
            title="TOPSIS Ranking (Closeness Coefficient, Ci)",
            xlabel="Closeness coefficient (Ci)",
            show_values=True,
            save_path=bar_save,
            show=show_plots
        )

    if plot_geom:
        geom_save = None
        if save_prefix is not None:
            geom_save = f"{save_prefix}_topsis_geometry.pdf"
        topsis_geometry_plot(
            S_plus=S_plus,
            S_minus=S_minus,
            best_idx=best_idx,
            title="TOPSIS Geometry (S+ vs S-)",
            save_path=geom_save,
            show=show_plots,
            invert_x=True
        )

    return c_i

###############################################################################
