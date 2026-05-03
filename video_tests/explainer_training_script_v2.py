# # ─────────────────────────────────────────────────────────────────────────────
# # train_video_explainer.py
# # Trains an XGBoost surrogate on aggregated MediaPipe features → scaled LSTM scores
# # Then runs SHAP to get feature importance for feedback
# #
# # INPUT:  explainer_training_data.csv  (from build_explainer_data.py)
# # OUTPUT:
# #   → video_explainer_model.pkl       (the trained XGBoost)
# #   → video_explainer_features.pkl    (feature list for inference)
# #   → video_shap_summary.csv          (SHAP importance per feature)
# #   → Prints R², MAE, Pearson, and top SHAP features
# #
# # HOW TO RUN:
# #   python train_video_explainer.py --csv explainer_training_data.csv
# # ─────────────────────────────────────────────────────────────────────────────

# import argparse
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import joblib
# import xgboost as xgb
# from sklearn.model_selection import GroupShuffleSplit, cross_validate
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from scipy.stats import pearsonr, spearmanr

# try:
#     import shap
#     HAS_SHAP = True
# except ImportError:
#     HAS_SHAP = False
#     print("WARNING: shap not installed. Run: pip install shap")
#     print("         Skipping SHAP analysis.\n")


# def main():
#     parser = argparse.ArgumentParser(description="Train video explainer XGBoost + SHAP")
#     parser.add_argument("--csv", required=True,
#                         help="Path to explainer_training_data.csv")
#     parser.add_argument("--output-model", default="video_explainer_model_v2.pkl")
#     parser.add_argument("--output-features", default="video_explainer_features_v2.pkl")
#     args = parser.parse_args()

#     # ── Load data ─────────────────────────────────────────────────────────────
#     print(f"Loading: {args.csv}")
#     df = pd.read_csv(args.csv)
#     print(f"  Rows: {len(df)}")

#     # ── Identify columns ─────────────────────────────────────────────────────
#     exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
#     feature_cols = [c for c in df.columns if c not in exclude_cols]
#     # ── Identify columns ─────────────────────────────────────────────────────
#     exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
#     feature_cols = [c for c in df.columns if c not in exclude_cols]

# # ── Drop specific gaze consistency features ───────────────────────────────
#     feature_cols = [c for c in feature_cols 
#                 if not c.endswith("_min")
#                 and c not in ("gaze_consistency_mean", "gaze_consistency_max")]

#     # ── ADD TEMPORAL FEATURES (Strategy 1) ────────────────────────────────────
#     print("  Computing temporal features from mean/std/max...")

#     temporal_cols = {}

#     # Extract base feature names (remove suffixes)
#     base_features = list(set(
#         c.replace("_mean","").replace("_std","").replace("_max","")
#         for c in feature_cols
#     ))

#     for base in base_features:
#         mean_col = f"{base}_mean"
#         std_col  = f"{base}_std"
#         max_col  = f"{base}_max"

#         # Coefficient of Variation (volatility)
#         if mean_col in df.columns and std_col in df.columns:
#             temporal_cols[f"{base}_cv"] = df[std_col] / (df[mean_col].abs() + 1e-6)

#         # Peak Ratio (spikes vs average)
#         if mean_col in df.columns and max_col in df.columns:
#             temporal_cols[f"{base}_peak_ratio"] = df[max_col] / (df[mean_col].abs() + 1e-6)

#     # Convert to DataFrame
#     temporal_df = pd.DataFrame(temporal_cols, index=df.index)

#     # Merge into dataset
#     df = pd.concat([df, temporal_df], axis=1)

#     # Update feature list
#     feature_cols = feature_cols + list(temporal_cols.keys())

#     print(f"  Added temporal features: {len(temporal_cols)}")
#     print(f"  Total features now: {len(feature_cols)}")

#     # NOW define X and y
#     X = df[feature_cols]
#     y = df["scaled_score"]

#     # ── Speaker-aware grouping ────────────────────────────────────────────────
#     # Extract speaker ID from filename: "abc123.000.mp4" → "abc123"
#     df["speaker_id"] = df["file"].str.rsplit(".", n=2).str[0]
#     groups = df["speaker_id"]
#     print(f"  Unique speakers: {groups.nunique()}")

#     # ── Cross-validation ──────────────────────────────────────────────────────
#     print(f"\nRunning 5-fold GroupKFold cross-validation...")

#     model = xgb.XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=500,
#         learning_rate=0.03,
#         max_depth=5,
#         min_child_weight=5,
#         subsample=0.8,
#         colsample_bytree=0.7,
#         reg_lambda=5.0,
#         reg_alpha=1.0,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0,
#     )

#     from sklearn.model_selection import GroupKFold
#     cv = GroupKFold(n_splits=5)

#     scoring = {
#         "mae": "neg_mean_absolute_error",
#         "rmse": "neg_root_mean_squared_error",
#         "r2": "r2",
#     }

#     cv_results = cross_validate(
#         model, X, y,
#         cv=cv,
#         groups=groups,
#         scoring=scoring,
#         return_train_score=True,
#         n_jobs=-1,
#     )

#     print(f"\n{'='*55}")
#     print(f"  CROSS-VALIDATION RESULTS")
#     print(f"{'='*55}")
#     print(f"  Train MAE:  {-cv_results['train_mae'].mean():.4f}")
#     print(f"  Val MAE:    {-cv_results['test_mae'].mean():.4f}")
#     print(f"  Train RMSE: {-cv_results['train_rmse'].mean():.4f}")
#     print(f"  Val RMSE:   {-cv_results['test_rmse'].mean():.4f}")
#     print(f"  Train R²:   {cv_results['train_r2'].mean():.4f}")
#     print(f"  Val R²:     {cv_results['test_r2'].mean():.4f}")
#     print(f"{'='*55}")

#     # ── Train/Val split for final model ───────────────────────────────────────
#     print(f"\nTraining final model with early stopping...")

#     gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
#     train_idx, val_idx = next(gss.split(X, y, groups=groups))

#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     final_model = xgb.XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=800,
#         learning_rate=0.025,
#         max_depth=5,
#         min_child_weight=5,
#         subsample=0.8,
#         colsample_bytree=0.7,
#         reg_lambda=5.0,
#         reg_alpha=1.0,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0,
#         early_stopping_rounds=50,
#         eval_metric="mae",
#     )

#     final_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=50,
#     )

#     # ── Final evaluation ──────────────────────────────────────────────────────
#     y_val_pred = np.clip(final_model.predict(X_val), 0, 1)

#     mae = mean_absolute_error(y_val, y_val_pred)
#     rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
#     r2 = r2_score(y_val, y_val_pred)
#     pearson_corr, p_val = pearsonr(y_val, y_val_pred)
#     spearman_corr, _ = spearmanr(y_val, y_val_pred)

#     print(f"\n{'='*55}")
#     print(f"  FINAL HOLD-OUT PERFORMANCE")
#     print(f"{'='*55}")
#     print(f"  MAE:                {mae:.4f}")
#     print(f"  RMSE:               {rmse:.4f}")
#     print(f"  R²:                 {r2:.4f}")
#     print(f"  Pearson:            {pearson_corr:.4f}  (p={p_val:.2e})")
#     print(f"  Spearman:           {spearman_corr:.4f}")
#     print(f"  Pred range:         [{y_val_pred.min():.3f} – {y_val_pred.max():.3f}]")
#     print(f"  Actual range:       [{y_val.min():.3f} – {y_val.max():.3f}]")
#     print(f"  Best iteration:     {final_model.best_iteration}")
#     print(f"{'='*55}")

#     # ── R² interpretation ─────────────────────────────────────────────────────
#     if r2 >= 0.85:
#         print(f"\n  ✓ R² = {r2:.4f} — EXCELLENT surrogate fidelity")
#         print(f"    SHAP feedback will be trustworthy.")
#     elif r2 >= 0.70:
#         print(f"\n  ~ R² = {r2:.4f} — ACCEPTABLE surrogate fidelity")
#         print(f"    SHAP feedback is usable but not perfect.")
#     else:
#         print(f"\n  ✗ R² = {r2:.4f} — LOW surrogate fidelity")
#         print(f"    SHAP feedback may not accurately reflect LSTM decisions.")
#         print(f"    Consider adding more features or tuning hyperparameters.")

#     # ── XGBoost feature importance ────────────────────────────────────────────
#     print(f"\nTop 15 features (XGBoost gain):")
#     importance = pd.DataFrame({
#         "feature": feature_cols,
#         "gain": final_model.feature_importances_,
#     }).sort_values("gain", ascending=False)
#     print(importance.head(15).to_string(index=False))

#     # ── SHAP analysis ─────────────────────────────────────────────────────────
#     if HAS_SHAP:
#         print(f"\nRunning SHAP TreeExplainer...")
#         explainer = shap.TreeExplainer(final_model)
#         shap_values = explainer.shap_values(X_val)

#         # Mean absolute SHAP value per feature
#         shap_importance = pd.DataFrame({
#             "feature": feature_cols,
#             "mean_abs_shap": np.abs(shap_values).mean(axis=0),
#         }).sort_values("mean_abs_shap", ascending=False)

#         print(f"\nTop 15 features (SHAP importance):")
#         print(shap_importance.head(15).to_string(index=False))

#         shap_importance.to_csv("video_shap_summary_v2.csv", index=False)
#         print(f"\n  Saved SHAP summary → video_shap_summary_v2.csv")

#         # Group by base feature (combine mean/std/min/max)
#         shap_importance["base_feature"] = shap_importance["feature"].str.rsplit("_", n=1).str[0]
#         grouped = shap_importance.groupby("base_feature")["mean_abs_shap"].sum().sort_values(ascending=False)

#         print(f"\nTop 10 features (grouped by body part):")
#         for feat, val in grouped.head(10).items():
#             print(f"  {feat:<30s}  {val:.4f}")

#         # Save SHAP values for threshold computation
#         shap_df = pd.DataFrame(shap_values, columns=feature_cols)
#         shap_df.to_csv("video_shap_values_full_v2.csv", index=False)
#         print(f"  Saved full SHAP values → video_shap_values_full_v2.csv")

#     # ── Save model ────────────────────────────────────────────────────────────

#     # ── Golden Zone Calculation ──────────────────────────────────────────────
#     print("\nCalculating Golden Zones (Target ranges for feedback)...")
    
#     # 1. Define 'High Performers' (Top 25% of scores)
#     top_threshold = df["scaled_score"].quantile(0.75)
#     top_performers = df[df["scaled_score"] >= top_threshold]
    
#     golden_zones = {}
#     for col in feature_cols:
#         # 2. Get the range used by the best performers for this feature
#         golden_zones[col] = {
#             "min": float(top_performers[col].min()),
#             "max": float(top_performers[col].max()),
#             "mean": float(top_performers[col].mean())
#         }
        
#     # 3. Save as JSON for your website/app to use
#     import json
#     with open("golden_zones.json", "w") as f:
#         json.dump(golden_zones, f, indent=4)
        
#     print(f"  Saved Golden Zones → golden_zones.json")
#     joblib.dump(final_model, args.output_model)
#     joblib.dump(feature_cols, args.output_features)

#     print(f"\n  Saved model    → {args.output_model}")
#     print(f"  Saved features → {args.output_features}")
#     print(f"\n  This model is for FEEDBACK ONLY (SHAP explanations).")
#     print(f"  Your LSTM (video_lstm.onnx) remains the scoring model.\n")


# if __name__ == "__main__":
#     main()

# ─────────────────────────────────────────────────────────────────────────────
# train_video_explainer.py
# Trains an XGBoost surrogate on aggregated MediaPipe features → scaled LSTM scores
# Then runs SHAP to get feature importance for feedback
#
# INPUT:  explainer_training_data.csv  (from build_explainer_data.py)
# OUTPUT:
#   → video_explainer_model.pkl       (the trained XGBoost)
#   → video_explainer_features.pkl    (feature list for inference)
#   → video_shap_summary.csv          (SHAP importance per feature)
#   → Prints R², MAE, Pearson, and top SHAP features
#
# HOW TO RUN:
#   python train_video_explainer.py --csv explainer_training_data.csv
# ─────────────────────────────────────────────────────────────────────────────



# __________________________________________________________________________________________________________________________________________
# import argparse
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import joblib
# import xgboost as xgb
# from sklearn.model_selection import GroupShuffleSplit, cross_validate
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from scipy.stats import pearsonr, spearmanr

# try:
#     import shap
#     HAS_SHAP = True
# except ImportError:
#     HAS_SHAP = False
#     print("WARNING: shap not installed. Run: pip install shap")
#     print("         Skipping SHAP analysis.\n")


# def main():
#     parser = argparse.ArgumentParser(description="Train video explainer XGBoost + SHAP")
#     parser.add_argument("--csv", required=True,
#                         help="Path to explainer_training_data.csv")
#     parser.add_argument("--output-model", default="video_explainer_model_v2.pkl")
#     parser.add_argument("--output-features", default="video_explainer_features_v2.pkl")
#     args = parser.parse_args()

#     # ── OUTPUT DIRECTORY (ADDED) ─────────────────────────────────────────────
#     output_dir = Path("backend/video_tests")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # ── Load data ─────────────────────────────────────────────────────────────
#     print(f"Loading: {args.csv}")
#     df = pd.read_csv(args.csv)
#     print(f"  Rows: {len(df)}")

#     # ── Identify columns ─────────────────────────────────────────────────────
#     exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
#     feature_cols = [c for c in df.columns if c not in exclude_cols]
#     exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
#     feature_cols = [c for c in df.columns if c not in exclude_cols]

#     # ── Drop specific gaze consistency features ───────────────────────────────
#     feature_cols = [c for c in feature_cols 
#                 if not c.endswith("_min")
#                 and c not in ("gaze_consistency_mean", "gaze_consistency_max")]

#     # ── ADD TEMPORAL FEATURES (Strategy 1) ────────────────────────────────────
#     print("  Computing temporal features from mean/std/max...")

#     temporal_cols = {}

#     base_features = list(set(
#         c.replace("_mean","").replace("_std","").replace("_max","")
#         for c in feature_cols
#     ))

#     for base in base_features:
#         mean_col = f"{base}_mean"
#         std_col  = f"{base}_std"
#         max_col  = f"{base}_max"

#         if mean_col in df.columns and std_col in df.columns:
#             temporal_cols[f"{base}_cv"] = df[std_col] / (df[mean_col].abs() + 1e-6)

#         if mean_col in df.columns and max_col in df.columns:
#             temporal_cols[f"{base}_peak_ratio"] = df[max_col] / (df[mean_col].abs() + 1e-6)

#     temporal_df = pd.DataFrame(temporal_cols, index=df.index)
#     df = pd.concat([df, temporal_df], axis=1)

#     feature_cols = feature_cols + list(temporal_cols.keys())

#     print(f"  Added temporal features: {len(temporal_cols)}")
#     print(f"  Total features now: {len(feature_cols)}")

#     X = df[feature_cols]
#     y = df["scaled_score"]

#     # ── Speaker-aware grouping ────────────────────────────────────────────────
#     df["speaker_id"] = df["file"].str.rsplit(".", n=2).str[0]
#     groups = df["speaker_id"]
#     print(f"  Unique speakers: {groups.nunique()}")

#     # ── Cross-validation ──────────────────────────────────────────────────────
#     print(f"\nRunning 5-fold GroupKFold cross-validation...")

#     model = xgb.XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=500,
#         learning_rate=0.03,
#         max_depth=5,
#         min_child_weight=5,
#         subsample=0.8,
#         colsample_bytree=0.7,
#         reg_lambda=5.0,
#         reg_alpha=1.0,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0,
#     )

#     from sklearn.model_selection import GroupKFold
#     cv = GroupKFold(n_splits=5)

#     scoring = {
#         "mae": "neg_mean_absolute_error",
#         "rmse": "neg_root_mean_squared_error",
#         "r2": "r2",
#     }

#     cv_results = cross_validate(
#         model, X, y,
#         cv=cv,
#         groups=groups,
#         scoring=scoring,
#         return_train_score=True,
#         n_jobs=-1,
#     )

#     print(f"\n{'='*55}")
#     print(f"  CROSS-VALIDATION RESULTS")
#     print(f"{'='*55}")
#     print(f"  Train MAE:  {-cv_results['train_mae'].mean():.4f}")
#     print(f"  Val MAE:    {-cv_results['test_mae'].mean():.4f}")
#     print(f"  Train RMSE: {-cv_results['train_rmse'].mean():.4f}")
#     print(f"  Val RMSE:   {-cv_results['test_rmse'].mean():.4f}")
#     print(f"  Train R²:   {cv_results['train_r2'].mean():.4f}")
#     print(f"  Val R²:     {cv_results['test_r2'].mean():.4f}")
#     print(f"{'='*55}")

#     # ── Train/Val split for final model ───────────────────────────────────────
#     print(f"\nTraining final model with early stopping...")

#     gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
#     train_idx, val_idx = next(gss.split(X, y, groups=groups))

#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     final_model = xgb.XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=800,
#         learning_rate=0.025,
#         max_depth=5,
#         min_child_weight=5,
#         subsample=0.8,
#         colsample_bytree=0.7,
#         reg_lambda=5.0,
#         reg_alpha=1.0,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0,
#         early_stopping_rounds=50,
#         eval_metric="mae",
#     )

#     final_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=50,
#     )

#     # ── Final evaluation ──────────────────────────────────────────────────────
#     y_val_pred = np.clip(final_model.predict(X_val), 0, 1)

#     mae = mean_absolute_error(y_val, y_val_pred)
#     rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
#     r2 = r2_score(y_val, y_val_pred)
#     pearson_corr, p_val = pearsonr(y_val, y_val_pred)
#     spearman_corr, _ = spearmanr(y_val, y_val_pred)

#     print(f"\n{'='*55}")
#     print(f"  FINAL HOLD-OUT PERFORMANCE")
#     print(f"{'='*55}")
#     print(f"  MAE:                {mae:.4f}")
#     print(f"  RMSE:               {rmse:.4f}")
#     print(f"  R²:                 {r2:.4f}")
#     print(f"  Pearson:            {pearson_corr:.4f}  (p={p_val:.2e})")
#     print(f"  Spearman:           {spearman_corr:.4f}")
#     print(f"  Pred range:         [{y_val_pred.min():.3f} – {y_val_pred.max():.3f}]")
#     print(f"  Actual range:       [{y_val.min():.3f} – {y_val.max():.3f}]")
#     print(f"  Best iteration:     {final_model.best_iteration}")
#     print(f"{'='*55}")

#     # ── SHAP analysis ─────────────────────────────────────────────────────────
#     if HAS_SHAP:
#         print(f"\nRunning SHAP TreeExplainer...")
#         explainer = shap.TreeExplainer(final_model)
#         shap_values = explainer.shap_values(X_val)

#         shap_importance = pd.DataFrame({
#             "feature": feature_cols,
#             "mean_abs_shap": np.abs(shap_values).mean(axis=0),
#         }).sort_values("mean_abs_shap", ascending=False)

#         shap_importance.to_csv(output_dir / "video_shap_summary_v2.csv", index=False)

#         shap_df = pd.DataFrame(shap_values, columns=feature_cols)
#         shap_df.to_csv(output_dir / "video_shap_values_full_v2.csv", index=False)

#     # ── Golden Zone Calculation ──────────────────────────────────────────────
#     import json
#     with open(output_dir / "golden_zones.json", "w") as f:
#         json.dump({}, f)

#     # ── Save model ────────────────────────────────────────────────────────────
#     joblib.dump(final_model, output_dir / args.output_model)
#     joblib.dump(feature_cols, output_dir / args.output_features)

#     print(f"\n  Saved model    → {output_dir / args.output_model}")
#     print(f"  Saved features → {output_dir / args.output_features}")


# if __name__ == "__main__":
#     main()


# ─────────────────────────────────────────────────────────────────────────────
# train_video_explainer.py
# Trains an XGBoost surrogate on aggregated MediaPipe features → scaled LSTM scores
# Then runs SHAP to get feature importance for feedback
#
# INPUT:  explainer_training_data.csv  (from build_explainer_data.py)
# OUTPUT:
#   → video_explainer_model.pkl       (the trained XGBoost)
#   → video_explainer_features.pkl    (feature list for inference)
#   → video_shap_summary.csv          (SHAP importance per feature)
#   → Prints R², MAE, Pearson, and top SHAP features
#
# HOW TO RUN:
#   python train_video_explainer.py --csv explainer_training_data.csv
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")
    print("         Skipping SHAP analysis.\n")


def main():
    parser = argparse.ArgumentParser(description="Train video explainer XGBoost + SHAP")
    parser.add_argument("--csv", required=True,
                        help="Path to explainer_training_data.csv")
    parser.add_argument("--output-model", default="video_explainer_model_v2.pkl")
    parser.add_argument("--output-features", default="video_explainer_features_v2.pkl")
    args = parser.parse_args()

    # ── OUTPUT DIRECTORY (ADDED) ─────────────────────────────────────────────
    output_dir = Path("backend/video_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  Rows: {len(df)}")

    # ── Identify columns ─────────────────────────────────────────────────────
    exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    exclude_cols = ["file", "raw_lstm_score", "scaled_score"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # ── Drop specific gaze consistency features ───────────────────────────────
    feature_cols = [c for c in feature_cols 
                if not c.endswith("_min")
                and c not in ("gaze_consistency_mean", "gaze_consistency_max")]

    # ── ADD TEMPORAL FEATURES (Strategy 1) ────────────────────────────────────
    print("  Computing temporal features from mean/std/max...")

    temporal_cols = {}

    base_features = list(set(
        c.replace("_mean","").replace("_std","").replace("_max","")
        for c in feature_cols
    ))

    for base in base_features:
        mean_col = f"{base}_mean"
        std_col  = f"{base}_std"
        max_col  = f"{base}_max"

        if mean_col in df.columns and std_col in df.columns:
            temporal_cols[f"{base}_cv"] = df[std_col] / (df[mean_col].abs() + 1e-6)

        if mean_col in df.columns and max_col in df.columns:
            temporal_cols[f"{base}_peak_ratio"] = df[max_col] / (df[mean_col].abs() + 1e-6)

    temporal_df = pd.DataFrame(temporal_cols, index=df.index)
    df = pd.concat([df, temporal_df], axis=1)

    feature_cols = feature_cols + list(temporal_cols.keys())

    print(f"  Added temporal features: {len(temporal_cols)}")
    print(f"  Total features now: {len(feature_cols)}")

    X = df[feature_cols]
    y = df["scaled_score"]

    # ── Speaker-aware grouping ────────────────────────────────────────────────
    df["speaker_id"] = df["file"].str.rsplit(".", n=2).str[0]
    groups = df["speaker_id"]
    print(f"  Unique speakers: {groups.nunique()}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\nRunning 5-fold GroupKFold cross-validation...")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=5)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    print(f"\n{'='*55}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Train MAE:  {-cv_results['train_mae'].mean():.4f}")
    print(f"  Val MAE:    {-cv_results['test_mae'].mean():.4f}")
    print(f"  Train RMSE: {-cv_results['train_rmse'].mean():.4f}")
    print(f"  Val RMSE:   {-cv_results['test_rmse'].mean():.4f}")
    print(f"  Train R²:   {cv_results['train_r2'].mean():.4f}")
    print(f"  Val R²:     {cv_results['test_r2'].mean():.4f}")
    print(f"{'='*55}")

    # ── Train/Val split for final model ───────────────────────────────────────
    print(f"\nTraining final model with early stopping...")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    y_val_pred = np.clip(final_model.predict(X_val), 0, 1)

    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    pearson_corr, p_val = pearsonr(y_val, y_val_pred)
    spearman_corr, _ = spearmanr(y_val, y_val_pred)

    print(f"\n{'='*55}")
    print(f"  FINAL HOLD-OUT PERFORMANCE")
    print(f"{'='*55}")
    print(f"  MAE:                {mae:.4f}")
    print(f"  RMSE:               {rmse:.4f}")
    print(f"  R²:                 {r2:.4f}")
    print(f"  Pearson:            {pearson_corr:.4f}  (p={p_val:.2e})")
    print(f"  Spearman:           {spearman_corr:.4f}")
    print(f"  Pred range:         [{y_val_pred.min():.3f} – {y_val_pred.max():.3f}]")
    print(f"  Actual range:       [{y_val.min():.3f} – {y_val.max():.3f}]")
    print(f"  Best iteration:     {final_model.best_iteration}")
    print(f"{'='*55}")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    if HAS_SHAP:
        print(f"\nRunning SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_val)

        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(output_dir / "video_shap_summary_v2.csv", index=False)

        shap_df = pd.DataFrame(shap_values, columns=feature_cols)
        shap_df.to_csv(output_dir / "video_shap_values_full_v2.csv", index=False)

    # ── Golden Zone Calculation ──────────────────────────────────────────────
    import json
    with open(output_dir / "golden_zones.json", "w") as f:
        json.dump({}, f)

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(final_model, output_dir / args.output_model)
    joblib.dump(feature_cols, output_dir / args.output_features)

    print(f"\n  Saved model    → {output_dir / args.output_model}")
    print(f"  Saved features → {output_dir / args.output_features}")


if __name__ == "__main__":
    main()