# process_solution_feature.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# Optional: mean-reversion / stationarity test
try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


REQUIRED_COLS = [
    "season", "week",
    "celebrity_name",
    "RQI_width", "entropy",
]


def _load_solution_feature_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {df.columns.tolist()}")

    # Coerce types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["RQI_width"] = pd.to_numeric(df["RQI_width"], errors="coerce")
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    # Drop rows that can't be used for season/week indexing
    df = df.dropna(subset=["season", "week"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    return df


def _to_week_level(df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """
    Convert celebrity-row table to Season-Week table.
    Checks if RQI_width/entropy are consistent within each Season-Week group.
    """
    g = df.groupby(["season", "week"], as_index=False)

    # Consistency check: if within-group max-min > tol, warn
    def _range(x: pd.Series) -> float:
        x = x.dropna()
        if x.empty:
            return 0.0
        return float(x.max() - x.min())

    width_range = g["RQI_width"].apply(_range).rename(columns={"RQI_width": "RQI_width_range"})
    ent_range = g["entropy"].apply(_range).rename(columns={"entropy": "entropy_range"})
    chk = width_range.merge(ent_range, on=["season", "week"], how="outer")

    bad_width = chk[chk["RQI_width_range"] > tol]
    bad_ent = chk[chk["entropy_range"] > tol]
    if (len(bad_width) > 0) or (len(bad_ent) > 0):
        print("WARNING: Detected inconsistent values within the same (season, week) group.", file=sys.stderr)
        if len(bad_width) > 0:
            print("  Inconsistent RQI_width groups (showing up to 10):", file=sys.stderr)
            print(bad_width.head(10).to_string(index=False), file=sys.stderr)
        if len(bad_ent) > 0:
            print("  Inconsistent entropy groups (showing up to 10):", file=sys.stderr)
            print(bad_ent.head(10).to_string(index=False), file=sys.stderr)

    # Use first non-null (they should be identical anyway)
    week_df = g.agg(
        RQI_width=("RQI_width", "first"),
        entropy=("entropy", "first"),
    ).sort_values(["season", "week"]).reset_index(drop=True)

    return week_df


def _segment_entropy_means(week_df: pd.DataFrame) -> pd.DataFrame:
    """
    Output entropy mean for:
      - S1-S2
      - S3-S27
      - S28+
      - Overall
    """
    def seg_mean(season_lo: int | None, season_hi: int | None) -> float:
        seg = week_df.copy()
        if season_lo is not None:
            seg = seg[seg["season"] >= season_lo]
        if season_hi is not None:
            seg = seg[seg["season"] <= season_hi]
        return float(seg["entropy"].mean(skipna=True))

    rows = [
        {"segment": "S1-S2",   "season_range": "[1, 2]",    "entropy_mean": seg_mean(1, 2)},
        {"segment": "S3-S27",  "season_range": "[3, 27]",   "entropy_mean": seg_mean(3, 27)},
        {"segment": "S28+",    "season_range": "[28, +∞)",  "entropy_mean": seg_mean(28, None)},
        {"segment": "Overall", "season_range": "All",       "entropy_mean": float(week_df["entropy"].mean(skipna=True))},
    ]
    return pd.DataFrame(rows)


def _season_entropy_S3_S27(week_df: pd.DataFrame) -> pd.DataFrame:
    """
    For S3-S27: mean entropy per season (averaged over weeks).
    """
    s = week_df[(week_df["season"] >= 3) & (week_df["season"] <= 27)].copy()
    out = (s.groupby("season", as_index=False)
             .agg(entropy_mean=("entropy", "mean"),
                  n_weeks=("entropy", lambda x: int(x.notna().sum()))))
    return out.sort_values("season").reset_index(drop=True)


def _linear_regression_season_entropy(season_df: pd.DataFrame) -> dict:
    """
    Fit entropy_mean ~ season for S3-S27.
    """
    x = season_df["season"].to_numpy(dtype=float)
    y = season_df["entropy_mean"].to_numpy(dtype=float)

    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "r2": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "stderr_slope": float(res.stderr),
        "n": int(len(x)),
    }


def _mean_reversion_tests(season_df: pd.DataFrame) -> dict:
    """
    Simple mean reversion diagnostics on season-level entropy means (S3-S27):
      - AR(1) coefficient via OLS: y_t = a + b*y_{t-1} + e
      - Half-life if 0<b<1
      - ADF test (if statsmodels available)
    """
    y = season_df["entropy_mean"].astype(float).to_numpy()
    y = y[~np.isnan(y)]
    out = {}

    if len(y) < 5:
        out["note"] = "Too few seasons for stable mean-reversion tests."
        return out

    y_lag = y[:-1]
    y_cur = y[1:]

    # OLS for AR(1)
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta, _, _, _ = np.linalg.lstsq(X, y_cur, rcond=None)  # [a, b]
    a, b = float(beta[0]), float(beta[1])

    out["ar1_intercept_a"] = a
    out["ar1_phi_b"] = b

    # Half-life: time for deviation to halve, if 0<phi<1
    if 0.0 < b < 1.0:
        out["half_life_seasons"] = float(-np.log(2.0) / np.log(b))
    else:
        out["half_life_seasons"] = np.nan

    # ADF
    if _HAS_STATSMODELS:
        adf_stat, pval, usedlag, nobs, crit, _ = adfuller(y, autolag="AIC")
        out["adf_stat"] = float(adf_stat)
        out["adf_pvalue"] = float(pval)
        out["adf_usedlag"] = int(usedlag)
        out["adf_nobs"] = int(nobs)
        out["adf_crit_1pct"] = float(crit.get("1%", np.nan))
        out["adf_crit_5pct"] = float(crit.get("5%", np.nan))
        out["adf_crit_10pct"] = float(crit.get("10%", np.nan))
    else:
        out["adf_note"] = "statsmodels not available; ADF test skipped."

    return out


def main():
    ap = argparse.ArgumentParser(description="Process solution_feature.csv (week-level entropy/RQI_width summaries).")
    ap.add_argument("--input", default="D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator/adjusted_solution/output_data/solution_feature.csv", help="D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator/adjusted_solution/output_data/solution_feature.csv")
    ap.add_argument("--out_dir", default="/mnt/data", help="Directory to write summary csv outputs")
    args = ap.parse_args()

    df = _load_solution_feature_csv(args.input)
    week_df = _to_week_level(df)

    # 1) Overall mean of RQI_width (week-level)
    rqi_mean = float(week_df["RQI_width"].mean(skipna=True))

    # 2) Entropy means for segments + overall
    seg_df = _segment_entropy_means(week_df)

    # S3-S27: season-level entropy mean series
    season_df = _season_entropy_S3_S27(week_df)

    # Regression + mean reversion diagnostics
    reg = _linear_regression_season_entropy(season_df) if len(season_df) >= 2 else {"note": "Not enough seasons for regression."}
    mr = _mean_reversion_tests(season_df)

    # ---- Print results ----
    print("\n================= (A) RQI_width =================")
    print(f"Week-level overall mean RQI_width: {rqi_mean:.6f}")

    print("\n================= (B) Entropy segment means =================")
    print(seg_df.to_string(index=False))

    print("\n================= (C) S3-S27 season-level entropy mean =================")
    print(season_df.to_string(index=False))

    print("\n================= (D) Linear regression on S3-S27 (entropy_mean ~ season) =================")
    for k, v in reg.items():
        print(f"{k}: {v}")

    print("\n================= (E) Mean-reversion diagnostics on S3-S27 season means =================")
    for k, v in mr.items():
        print(f"{k}: {v}")

    # ---- Save outputs ----
    os.makedirs(args.out_dir, exist_ok=True)
    seg_path = os.path.join(args.out_dir, "entropy_segment_means.csv")
    season_path = os.path.join(args.out_dir, "entropy_mean_S3_S27_by_season.csv")
    week_path = os.path.join(args.out_dir, "week_level_solution_features.csv")

    seg_df.to_csv(seg_path, index=False)
    season_df.to_csv(season_path, index=False)
    week_df.to_csv(week_path, index=False)

    print("\nSaved:")
    print(f"  {seg_path}")
    print(f"  {season_path}")
    print(f"  {week_path}")


if __name__ == "__main__":
    main()
