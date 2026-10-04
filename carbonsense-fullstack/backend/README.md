# CarbonSense Backend (Flask)

A carbon footprint prediction system with an explainable-AI (SHAP) recommendation
engine. Predictions come from a 5-model ensemble — **Ridge, Random Forest,
CatBoost, LightGBM, and XGBoost** — trained in `carbon_emission_v3_fixed.ipynb`.
This backend loads those trained models directly (no retraining at request
time) and serves:

- the ensemble prediction (average of all 5 models)
- SHAP feature-importance values from the best tree model (`shap.TreeExplainer`)
- ranked recommendations from real counterfactual testing against that same
  tree model (not a static tip library — each suggested action is actually
  run through the model to measure its predicted kg CO2e reduction)

## Run it

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

API comes up at `http://localhost:5000`.

- `GET  /api/health` → `{ "status": "ok", "mock_mode": true/false, "models_loaded": [...] }`
- `POST /api/predict` → body is the 19 raw form fields (see `ml/features.py` —
  keys match the original dataset column names, e.g. `"Body Type"`,
  `"Vehicle Monthly Distance Km"`, `"Recycling": ["Paper","Plastic"]`).
  Returns predictions per model + ensemble, SHAP breakdown, and recommendations.

## Wiring in your trained artifacts

This backend expects the exact files your notebook already produces —
**no retraining, no format conversion**. See `models/README.md` for the
full file list and where each one comes from (`outputs/` vs `outputs/models/`
in the notebook). Once all 15 files are copied into `backend/models/`,
restart the server — `/api/health` will report `"mock_mode": false`.

## How each piece maps to the notebook

| Backend file | Ported from notebook |
|---|---|
| `ml/preprocessing.py` | Cell 36 (`preprocess_new_input`) — same 9-step pipeline: bounds → winsorize → impute → multi-label binarize → ordinal encode → one-hot encode → align → prune → final align |
| `ml/ml_service.py` | Cells 73/82/83/85 — loads the 5 tuned models, picks the best tree model for SHAP the same way (lowest `Tuned_RMSE` from `results_table.csv`, tie-broken by a fixed preference order), and calls `shap.TreeExplainer` exactly as the notebook does |
| `ml/recommender.py` | Cell 115 (`build_action_catalogue` / `generate_whatif`) — same counterfactual actions (switch diet, transport, vehicle, heating, recycling), tested against the real model and ranked by actual predicted reduction |

Ridge is scored on the scaled feature matrix (`scaler.joblib`), matching the
notebook's `model_file_map` (`X_train_s`/`X_test_s` for Ridge, unscaled for
the four tree models).

## Mock mode

Until every model + preprocessing artifact is present, the API runs in
**mock mode**: `/api/health` reports `"mock_mode": true` and predictions come
from a lightweight heuristic in `ml_service.py` — good enough to build and
test the full frontend↔backend flow before your artifacts are in place, with
no code changes needed on either side once they are.

## CORS

`CORS_ORIGINS` in `config.py` defaults to `*` for local development
(with a manual-header fallback if `flask-cors` isn't installed). Lock this
down to your actual frontend origin before deploying.
