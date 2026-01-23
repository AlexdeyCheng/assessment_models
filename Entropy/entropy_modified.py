###############################################################################
# Required Libraries
import itertools
import numpy as np
import warnings

###############################################################################
# Helpers: criterion parsing + Route A transforms (nominal/interval -> benefit)

def _str_extract(t):
    """Extract the type-name token from criterion_type entries."""
    if isinstance(t, str):
        return t
    elif isinstance(t, (list, tuple, np.ndarray)):
        return t[0]
    else:
        return t


def _finite_check_or_raise(arr, name="array"):
    """
    Ensure arr contains only finite values. If not, raise ValueError with positions.
    """
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isfinite(arr)
    if np.any(mask):
        idx = np.argwhere(mask)
        preview = idx[:12].tolist()
        total = int(idx.shape[0])
        raise ValueError(
            f"{name} contains NaN/inf at {total} position(s). "
            f"First positions (row, col): {preview}"
        )
    return arr


def _raise_invalid_criterion_param(j, msg):
    """Raise a ValueError with criterion column information (0-based and 1-based)."""
    raise ValueError(f"criterion_type error at criterion column j={j} (1-based {j+1}): {msg}")


def _parse_criterion_type(criterion_type, n):
    """
    Parse criterion_type into (ctype_name, params).

    Supported formats per criterion j:
      - 'max' or 'min'
      - ['nominal', target_value]
      - ['interval', a, b]   (a/b order can be swapped)

    Returns
    -------
    ctype_name : list[str] length n
        One of {'max','min','nominal','interval'}.
    params : list[tuple]
        - max/min: ()
        - nominal: (target,)
        - interval: (L, U) with L <= U
    """
    if len(criterion_type) != n:
        raise ValueError("criterion_type length must equal number of criteria (dataset.shape[1]).")

    ctype_name, params = [], []

    for j in range(n):
        entry = criterion_type[j]
        name = _str_extract(entry)

        if not isinstance(name, str):
            _raise_invalid_criterion_param(j, "Type token must be a string like 'max', 'min', 'nominal', 'interval'.")

        name = name.strip().lower()
        if name not in ("max", "min", "nominal", "interval"):
            raise ValueError("criterion_type entries must be: 'max', 'min', ['nominal', target], ['interval', a, b].")

        if name in ("max", "min"):
            ctype_name.append(name)
            params.append(tuple())
            continue

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


def _distance_to_target(x, target):
    """Absolute distance to a target value (nominal-the-best)."""
    x = np.asarray(x, dtype=float)
    return np.abs(x - float(target))


def _distance_to_interval(x, L, U, mode="piecewise"):
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


def _distance_to_benefit(d, eps=1e-12):
    """
    Convert distance to benefit score in [0, 1], where smaller distance is better.

    Default mapping (easy to maintain/replace later):
      benefit = 1 - d / (max(d) + eps)

    If distances are constant, returns all-ones and marks as constant.
    """
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return d, True

    if np.allclose(d, d[0]):
        return np.ones_like(d, dtype=float), True

    dmax = float(np.max(d))
    benefit = 1.0 - d / (dmax + float(eps))
    benefit = np.clip(benefit, 0.0, 1.0)
    return benefit, bool(np.allclose(benefit, benefit[0]))


def _apply_routeA_transform_for_entropy(X, ctype_name, params, warn_constant=True):
    """
    Route A for entropy weighting: transform nominal/interval columns into benefit-type columns.

    Returns
    -------
    X_eff : ndarray (m, n)
    ctype_eff : list[str] length n
        Effective types: only 'max'/'min' remain (nominal/interval -> 'max').
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    X_eff = np.array(X, copy=True, dtype=float)
    ctype_eff = []

    for j in range(n):
        tname = ctype_name[j]

        if tname in ("max", "min"):
            ctype_eff.append(tname)
            continue

        if tname == "nominal":
            target = params[j][0]
            d = _distance_to_target(X[:, j], target)
            benefit, is_constant = _distance_to_benefit(d)
            X_eff[:, j] = benefit
            ctype_eff.append("max")

            if warn_constant and is_constant:
                warnings.warn(
                    f"Entropy Route A: criterion j={j} (1-based {j+1}) became constant after nominal transform "
                    f"(target={target}). This criterion provides no discrimination.",
                    RuntimeWarning
                )
            continue

        if tname == "interval":
            L, U = params[j]
            d = _distance_to_interval(X[:, j], L, U, mode="piecewise")
            benefit, is_constant = _distance_to_benefit(d)
            X_eff[:, j] = benefit
            ctype_eff.append("max")

            if warn_constant and is_constant:
                warnings.warn(
                    f"Entropy Route A: criterion j={j} (1-based {j+1}) became constant after interval transform "
                    f"(L={L}, U={U}). This criterion provides no discrimination.",
                    RuntimeWarning
                )
            continue

        _raise_invalid_criterion_param(j, f"Unhandled criterion type: {tname}")

    return X_eff, ctype_eff


###############################################################################
# Function: Entropy (extended for nominal & interval)

def entropy_method(dataset, criterion_type):
    """
    Entropy weight method with support for:
      - 'max' / 'min'
      - ['nominal', target_value]   -> Route A transform to benefit
      - ['interval', a, b]          -> Route A transform to benefit (a/b order auto-swapped)

    Parameters
    ----------
    dataset : array-like (m, n)
        Alternatives x criteria.
    criterion_type : list length n
        Mixed types as described above.

    Returns
    -------
    w : ndarray (n,)
        Entropy weights summing to 1.
    """
    X = np.asarray(dataset, dtype=float)
    if X.ndim != 2:
        raise ValueError("dataset must be a 2D array of shape (m, n).")
    _finite_check_or_raise(X, name="dataset")

    m, n = X.shape
    ctype_name, params = _parse_criterion_type(criterion_type, n)

    # Route A: nominal/interval -> benefit columns, then treat as 'max'
    X_eff, ctype_eff = _apply_routeA_transform_for_entropy(X, ctype_name, params, warn_constant=True)

    # Build probability matrix P (m x n)
    P = np.zeros_like(X_eff, dtype=float)

    for j in range(n):
        col = X_eff[:, j].astype(float)

        # Keep original behavior: use absolute values after preprocessing
        # (benefit columns are already in [0,1], but abs keeps compatibility)
        col = np.abs(col)

        if ctype_eff[j] == "max":
            s = np.sum(col)
            if np.isclose(s, 0.0):
                raise ValueError(f"Entropy: column j={j} (1-based {j+1}) has zero sum after preprocessing.")
            P[:, j] = col / (s + 1e-12)

        else:  # 'min' (cost) -> reciprocal normalization
            inv = 1.0 / (col + 1e-9)
            s = np.sum(inv)
            if np.isclose(s, 0.0):
                raise ValueError(f"Entropy: column j={j} (1-based {j+1}) has zero reciprocal-sum after preprocessing.")
            P[:, j] = inv / (s + 1e-12)

    # Entropy calculation
    H = np.zeros_like(P, dtype=float)
    for j, i in itertools.product(range(n), range(m)):
        if P[i, j] != 0:
            H[i, j] = P[i, j] * np.log(P[i, j] + 1e-9)

    # h_j = -k * sum_i p_ij ln(p_ij),  k = 1/ln(m)
    k = 1.0 / (np.log(m + 1e-9))
    h = -k * np.sum(H, axis=0)

    # Degree of diversification
    d = 1.0 - h
    d = d + 1e-9

    # Weights
    w = d / np.sum(d)
    return w

###############################################################################
