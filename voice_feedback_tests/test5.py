"""
calibrate_audio_golden_zones.py
────────────────────────────────────────────────────────────────────────────
Computes golden zones from the audio explainer training CSV.

Key difference from the video version:
  • No feature engineering step — the audio CSV already contains flat,
    pre-computed features (no _mean/_std/_max triplets to derive from).
  • Score column is `confidence_score` (not raw_lstm_score / scaled_score).
  • global_stats stores score_min / score_max of confidence_score directly,
    used by the pipeline to clip/normalise if needed.

USAGE:
    python calibrate_audio_golden_zones.py
        --csv  audio_explainer_training_data.csv
        --prev audio_calibration_data_v1.json   (optional)
        --out  audio_calibration_data_v1.json
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import numpy as np
import pandas as pd


# ── Direction detection ───────────────────────────────────────────────────────
def detect_direction(series: pd.Series, scores: pd.Series) -> str:
    """
    Categorises how this feature correlates with the confidence score.
    Requires at least 10 unique values; otherwise returns FLAT.
    """
    if series.nunique() < 10:
        return "FLAT"

    tmp = pd.DataFrame({"x": series, "y": scores})
    tmp["level"] = pd.qcut(
        tmp["x"].rank(method="first"), q=3, labels=["low", "mid", "high"]
    )
    g = tmp.groupby("level", observed=False)["y"].mean()
    low_s, mid_s, high_s = g["low"], g["mid"], g["high"]

    if   low_s < mid_s < high_s:           return "INCREASING"
    elif high_s < mid_s < low_s:           return "DECREASING"
    elif low_s > mid_s and high_s > mid_s: return "U_SHAPED"
    elif low_s < mid_s and high_s < mid_s: return "INVERTED_U"
    else:                                  return "UNCLEAR"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True,  help="audio_explainer_training_data.csv")
    parser.add_argument("--prev", required=False, help="Previous calibration JSON (to carry over categories)")
    parser.add_argument("--out",  default="audio_calibration_data_v1.json")
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  Rows: {len(df)}")

    # ── Feature list ──────────────────────────────────────────────────────────
    # Exclude non-feature columns. No feature engineering needed — CSV is flat.
    exclude = {"file", "confidence_score"}
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"  Features: {len(feature_cols)}")

    # ── Elite / Low splits (top/bottom 25% by confidence_score) ──────────────
    p75 = df["confidence_score"].quantile(0.75)
    p25 = df["confidence_score"].quantile(0.25)
    elite = df[df["confidence_score"] >= p75]
    low   = df[df["confidence_score"] <= p25]
    print(f"  Elite (≥p75={p75:.3f}): {len(elite)} samples")
    print(f"  Low   (≤p25={p25:.3f}): {len(low)}   samples")

    # ── Compute golden zones ──────────────────────────────────────────────────
    print("  Computing golden zones...")
    golden_zones = {}

    for col in feature_cols:
        zone_min  = float(elite[col].quantile(0.25))
        zone_max  = float(elite[col].quantile(0.75))
        zone_mean = float(elite[col].mean())
        direction = detect_direction(df[col], df["confidence_score"])

        golden_zones[col] = {
            "min":       zone_min,
            "max":       zone_max,
            "mean":      zone_mean,
            "direction": direction,
        }

    # ── Global stats — always recomputed fresh from the CSV ───────────────────
    global_stats = {
        "score_min":        float(df["confidence_score"].min()),
        "score_max":        float(df["confidence_score"].max()),
        "score_mean":       float(df["confidence_score"].mean()),
        "score_p25":        float(p25),
        "score_p75":        float(p75),
        "n_samples":        int(len(df)),
    }
    print(
        f"  global_stats: score_min={global_stats['score_min']:.4f}  "
        f"score_max={global_stats['score_max']:.4f}"
    )

    # ── Carry over categories from --prev if supplied (cosmetic only) ─────────
    prev_categories = {}
    if args.prev:
        try:
            with open(args.prev) as f:
                old = json.load(f)
            prev_categories = old.get("categories", {})
            print(f"  Loaded categories from: {args.prev}")
        except FileNotFoundError:
            print(f"  WARNING: {args.prev} not found — skipping categories merge")

    output = {
        "categories":   prev_categories,
        "golden_zones": golden_zones,
        "global_stats": global_stats,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved: {args.out}  ({len(golden_zones)} features)")

    # ── Usefulness audit ──────────────────────────────────────────────────────
    print("\n=== Zone usefulness (% of low scorers correctly flagged outside zone) ===")
    results = []
    seen_cols = set()
    for col, gz in golden_zones.items():
        if col not in low.columns or col in seen_cols:
            continue
        seen_cols.add(col)
        pct = ((low[col] < gz["min"]) | (low[col] > gz["max"])).mean()
        results.append((col, pct, gz["direction"]))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Feature':<50} {'Flagged':>8}  Direction")
    print("-" * 75)
    for col, pct, direction in results[:30]:
        print(f"{col:<50} {pct*100:>7.0f}%  {direction}")

    # Summary by feature group
    print("\n=== Avg flagging rate by feature group ===")
    groups = {
        "Loudness":   ["loudness"],
        "Pitch/F0":   ["F0semitone"],
        "Formants":   ["F2", "F3"],
        "Voice qual": ["HNR", "shimmer", "jitter", "vocal_instability"],
        "Spectral":   ["spectral", "alpha", "hammar", "slope", "brightness"],
        "MFCC":       ["mfcc"],
        "Fluency":    ["voiced", "MeanUnvoiced", "vocal_projection", "vocal_instability"],
    }
    for grp_name, prefixes in groups.items():
        subset = [
            (c, p) for c, p, _ in results
            if any(c.lower().startswith(px.lower()) or px.lower() in c.lower()
                   for px in prefixes)
        ]
        if subset:
            avg = np.mean([p for _, p in subset])
            print(f"  {grp_name:<15}  {avg*100:.0f}%  ({len(subset)} features)")


if __name__ == "__main__":
    main()