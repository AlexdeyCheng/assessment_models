"""
Optimize 3 RQI plots (no titles, legends shown, new filenames to avoid overwrite)

Targets:
- Season 11: highlight Bristol Palin
- Season 25: highlight Vanessa Lachey
- Season 3 : highlight Week 1 cross-contestant spread (max-min ~ 1)

Input CSV must contain columns:
  season, week, celebrity_name, RQI_width
Default input: ./solution_feature.csv (or /mnt/data/solution_feature.csv fallback)

Outputs (PNG) will be saved to --outdir (default: current dir).
Filenames are auto-deconflicted to avoid overwriting existing files.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _safe_path(outdir: str, basename: str) -> str:
    """
    Return a non-overwriting filepath:
      basename.png, basename_v2.png, basename_v3.png, ...
    """
    outdir = outdir or "."
    os.makedirs(outdir, exist_ok=True)

    root, ext = os.path.splitext(basename)
    if not ext:
        ext = ".png"

    candidate = os.path.join(outdir, root + ext)
    if not os.path.exists(candidate):
        return candidate

    k = 2
    while True:
        candidate = os.path.join(outdir, f"{root}_v{k}{ext}")
        if not os.path.exists(candidate):
            return candidate
        k += 1


def _short_label(full_name: str) -> str:
    """
    'Vanessa Lachey' -> 'Lachey V'
    'John O'Hurley'  -> "O'Hurley J"
    """
    s = str(full_name).strip()
    if not s:
        return s
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last = parts[-1]
    return f"{last} {first[0]}"


def _ensure_unique_labels(names: list[str]) -> dict[str, str]:
    """
    Ensure labels are unique within one season. If collision occurs:
      'Smith J' and 'Smith J' -> 'Smith J', 'Smith J2', ...
    Returns map: full_name -> unique_label
    """
    base = {n: _short_label(n) for n in names}
    used = {}
    out = {}
    for n in names:
        lab = base[n]
        if lab not in used:
            used[lab] = 1
            out[n] = lab
        else:
            used[lab] += 1
            out[n] = f"{lab}{used[lab]}"
    return out


def _load_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        path = csv_path
    elif os.path.exists("D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator/adjusted_solution/output_data/solution_feature.csv"):
        path = "D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator/adjusted_solution/output_data/solution_feature.csv"
    else:
        raise FileNotFoundError(
            f"Cannot find CSV at '{csv_path}' or '/mnt/data/solution_feature.csv'."
        )

    df = pd.read_csv(path)
    required = {"season", "week", "celebrity_name", "RQI_width"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["RQI_width"] = pd.to_numeric(df["RQI_width"], errors="coerce")
    df["celebrity_name"] = df["celebrity_name"].astype(str)
    return df


def _season_wide(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Return (wide_df, label_map)
    wide_df index=week, columns=unique labels, values=RQI_width
    """
    sub = df[df["season"] == season].dropna(subset=["RQI_width"]).copy()
    if sub.empty:
        raise ValueError(f"No RQI_width data found for season {season}.")

    names = sorted(sub["celebrity_name"].unique().tolist())
    label_map = _ensure_unique_labels(names)
    sub["label"] = sub["celebrity_name"].map(label_map)

    wide = (
        sub.pivot_table(index="week", columns="label", values="RQI_width", aggfunc="mean")
        .sort_index()
    )
    return wide, label_map


# -----------------------------
# Plotting
# -----------------------------

def plot_season_focus(
    df: pd.DataFrame,
    season: int,
    focus_name: str,
    outdir: str,
    annotate: str | None = None,  # "extremes" or "spike" or None
) -> str:
    wide, label_map = _season_wide(df, season)

    # resolve focus label
    if focus_name not in label_map:
        # try case-insensitive match
        candidates = df.loc[df["season"] == season, "celebrity_name"].unique().tolist()
        hit = [c for c in candidates if c.lower() == focus_name.lower()]
        if not hit:
            raise ValueError(
                f"Focus name '{focus_name}' not found in season {season}. "
                f"Available examples: {candidates[:8]} ..."
            )
        focus_name = hit[0]

    focus_label = label_map[focus_name]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Background lines (context)
    for col in wide.columns:
        if col == focus_label:
            continue
        ax.plot(
            wide.index,
            wide[col],
            linewidth=0.9,
            alpha=0.18,
            color="0.5",      # grey
            marker="o",
            markersize=3,
            label="_nolegend_",
        )

    # Focus line (foreground)
    y_focus = wide[focus_label]
    ax.plot(
        wide.index,
        y_focus,
        linewidth=3.2,
        alpha=0.95,
        marker="o",
        markersize=7,
        label=focus_label,
    )

    # Optional annotations
    if annotate == "extremes":
        # mark max/min points for the focus line
        s = y_focus.dropna()
        if len(s) >= 2:
            w_max = int(s.idxmax())
            w_min = int(s.idxmin())
            y_max = float(s.loc[w_max])
            y_min = float(s.loc[w_min])

            ax.scatter([w_max], [y_max], s=120, marker="^", zorder=5, label="_nolegend_")
            ax.scatter([w_min], [y_min], s=120, marker="v", zorder=5, label="_nolegend_")

            ax.annotate(
                f"max @W{w_max}: {y_max:.2f}",
                xy=(w_max, y_max),
                xytext=(w_max + 0.2, min(1.03, y_max + 0.08)),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=10,
            )
            ax.annotate(
                f"min @W{w_min}: {y_min:.2f}",
                xy=(w_min, y_min),
                xytext=(w_min + 0.2, max(-0.02, y_min - 0.12)),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=10,
            )

    if annotate == "spike":
        # mark the max point as "spike"
        s = y_focus.dropna()
        if len(s) >= 2:
            w_max = int(s.idxmax())
            y_max = float(s.loc[w_max])
            ax.scatter([w_max], [y_max], s=150, marker="*", zorder=6, label="_nolegend_")
            ax.annotate(
                f"spike @W{w_max}: {y_max:.2f}",
                xy=(w_max, y_max),
                xytext=(w_max + 0.2, min(1.03, y_max + 0.08)),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=10,
            )

    # Axes (no title)
    ax.set_xlabel("Week")
    ax.set_ylabel("RQI Width (90% CI)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(sorted(wide.index.unique().tolist()))
    ax.grid(True, alpha=0.25)

    # Legend outside (required)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    # Save without overwriting
    safe_focus = focus_label.replace(" ", "")
    base = f"S{season:02d}_RQI_Focus_{safe_focus}.png"
    outpath = _safe_path(outdir, base)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_season3_week1_range(df: pd.DataFrame, outdir: str) -> str:
    season = 3
    wide, label_map = _season_wide(df, season)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Background lines
    for col in wide.columns:
        ax.plot(
            wide.index,
            wide[col],
            linewidth=0.9,
            alpha=0.18,
            color="0.5",
            marker="o",
            markersize=3,
            label="_nolegend_",
        )

    # Week 1 range highlight (focus element)
    sub_w1 = df[(df["season"] == season) & (df["week"] == 1)].dropna(subset=["RQI_width"]).copy()
    if sub_w1.empty:
        raise ValueError("Season 3 has no Week 1 RQI_width data.")

    # Use the same unique label mapping for names present in season 3
    sub_w1["label"] = sub_w1["celebrity_name"].map(label_map)
    sub_w1 = sub_w1.dropna(subset=["label"])

    i_max = sub_w1["RQI_width"].idxmax()
    i_min = sub_w1["RQI_width"].idxmin()

    max_row = sub_w1.loc[i_max]
    min_row = sub_w1.loc[i_min]

    x = 1
    y_max = float(max_row["RQI_width"])
    y_min = float(min_row["RQI_width"])
    rng = y_max - y_min

    # vertical range bar
    ax.vlines(x, y_min, y_max, linewidth=5.0, alpha=0.6, color="black", label="_nolegend_")

    # endpoint markers (in legend)
    ax.scatter([x], [y_max], s=140, marker="^", zorder=6, label=f"Week1 max: {max_row['label']}")
    ax.scatter([x], [y_min], s=140, marker="v", zorder=6, label=f"Week1 min: {min_row['label']}")

    # annotate range
    ax.annotate(
        f"Week1 range ≈ {rng:.2f}",
        xy=(x, (y_min + y_max) / 2),
        xytext=(x + 0.3, min(1.02, (y_min + y_max) / 2 + 0.05)),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=11,
    )

    # Axes (no title)
    ax.set_xlabel("Week")
    ax.set_ylabel("RQI Width (90% CI)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(sorted(wide.index.unique().tolist()))
    ax.grid(True, alpha=0.25)

    # Legend outside (required)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    outpath = _safe_path(outdir, "S03_RQI_Week1_Range.png")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="solution_feature.csv", help="Path to solution_feature.csv")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for PNGs")
    args = parser.parse_args()

    df = _load_csv(args.csv)

    # S11 focus: Bristol Palin (extremes)
    p1 = plot_season_focus(
        df=df,
        season=11,
        focus_name="Bristol Palin",
        outdir=args.outdir,
        annotate="extremes",
    )

    # S25 focus: Vanessa Lachey (spike)
    p2 = plot_season_focus(
        df=df,
        season=25,
        focus_name="Vanessa Lachey",
        outdir=args.outdir,
        annotate="spike",
    )

    # S3 week1 range highlight
    p3 = plot_season3_week1_range(df=df, outdir=args.outdir)

    print("[Saved]")
    print("  ", p1)
    print("  ", p2)
    print("  ", p3)


if __name__ == "__main__":
    main()
