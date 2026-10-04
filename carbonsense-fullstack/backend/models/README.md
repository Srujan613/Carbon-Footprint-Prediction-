# Drop your trained artifacts here

These are the **exact files your notebook already produces** — just copy
them from `outputs/` and `outputs/models/` (wherever you ran the notebook)
into this folder, flattened (no subfolders).

As of `carbon_emission_v4_tuning_optimized_final.ipynb`, preprocessing was
consolidated into a single fitted `preprocessing_pipeline.joblib` — this
replaces the 7 separate files v3 needed (`ordinal_encoder.joblib`,
`training_columns_post_ohe.json`, `collinear_dropped.json`,
`low_importance_dropped.json`, `train_medians.json`, `train_modes.json`,
`winsor_bounds.json`). If you still have those from a v3 run, they're no
longer read by the backend and can be deleted from this folder.

## From `outputs/models/` (Part 2 — cell 73)
- `ridge_tuned.joblib`
- `rf_tuned.joblib`
- `xgb_tuned.joblib`
- `lgbm_tuned.joblib`
- `catboost_tuned.joblib`
- `results_table.csv` *(optional — used to auto-pick the best tree model for SHAP, same logic as the notebook's model-selection cell; if omitted, the backend just uses CatBoost)*

## From `outputs/` (Part 1 — cell 32)
- `preprocessing_pipeline.joblib` *(the single source of truth for all preprocessing — hard bounds, multi-label parsing, imputation, winsorization, encoding, and feature pruning)*
- `scaler.joblib`
- `feature_names.json`

## From `outputs/explainability/` (Part 3 — cells 110 & 112)
- `kmeans_archetype_model.joblib` *(fitted KMeans — assigns a new user to one of the auto-selected `k` lifestyle clusters. **Must** receive `scaler.joblib`-scaled input, same as Ridge — it was fit on `X_train_s`, not raw features.)*
- `archetype_labels.json` *(maps each cluster id to its data-driven label, e.g. `"0": "High-Emission / Frequent Flyer"`)*
- `stacking_ensemble.joblib` *(the `StackingRegressor` — RF + XGBoost + LightGBM + CatBoost feeding a `RidgeCV` meta-learner. Operates on raw unscaled features, same as the individual tree models, since it wraps them directly.)*

Both of these are optional adders on top of the 8 core files above: without
them, `/api/predict` still runs, just without a `"stacking"` prediction and
with a heuristic (non-KMeans) archetype guess instead of the real cluster
assignment. Check `GET /api/health`'s `stacking_available` /
`archetype_available` flags to confirm they loaded.

That's now 11 files total when you include both add-ons (8 core + 3 for
stacking/archetypes) — still down from 15 in v3.

## Quickest way to grab them all

If you ran the notebook locally, from the folder containing `outputs/`:

```bash
cp outputs/models/ridge_tuned.joblib \
   outputs/models/rf_tuned.joblib \
   outputs/models/xgb_tuned.joblib \
   outputs/models/lgbm_tuned.joblib \
   outputs/models/catboost_tuned.joblib \
   outputs/models/results_table.csv \
   outputs/models/stacking_ensemble.joblib \
   outputs/preprocessing_pipeline.joblib \
   outputs/scaler.joblib \
   outputs/feature_names.json \
   outputs/explainability/kmeans_archetype_model.joblib \
   outputs/explainability/archetype_labels.json \
   /path/to/carbonsense/backend/models/
```

(On Windows, copy the same files individually — File Explorer or `copy` in PowerShell works fine.)

Until **all** the model files and preprocessing artifacts above are present,
`GET /api/health` reports `"mock_mode": true` and the API serves realistic
placeholder numbers instead of your real models — enough to test the full
frontend↔backend flow while you finish exporting.
