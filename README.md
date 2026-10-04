# CarbonSense: Carbon Footprint Prediction

**An Explainable Green Lifestyle Intelligence Platform for Carbon Footprint Prediction and Sustainable Decision-Making**

CarbonSense predicts an individual's monthly carbon emission from lifestyle data, explains *why* the model predicted it using SHAP, and turns those explanations into personalised suggestions for reducing the footprint. It includes the full ML pipeline (notebook) and a full-stack web application (Flask backend).

Developed as a joint internship project at **PES University CoDMAV**.

---

## Highlights

- **Data:** about 10,000 records, 19 lifestyle features covering mobility, home energy, diet, consumption and waste
- **Best model:** tuned CatBoost, chosen over the stacking ensemble for near-identical accuracy and better interpretability
- **Explainability:** global and local SHAP analysis
- **Recommendations:** an engine that maps the biggest emission drivers to actionable reduction tips
- **Full stack:** Flask API serving the trained pipeline, with a web front end

## Results

| Model | RMSE | R² | MAPE |
|---|---|---|---|
| CatBoost (tuned, headline model) | ~90.54 | ~0.9921 | ~3.09% |

The stacking ensemble is kept as an ablation study. See the notebook for the full comparison across CatBoost, XGBoost, LightGBM, Random Forest and Ridge Regression.

## Architecture

```
Mobility ─────────┐
Home energy ──────┼─► Preprocessing ─► Encoding ─► Unified feature matrix
Lifestyle ────────┘                                         │
                                                            ▼
              Model training ─► Prediction ─► SHAP analysis ─► Recommendations
```

## Tech stack

- **ML:** scikit-learn (Pipelines), CatBoost, XGBoost, LightGBM, Optuna (TPE tuning), SHAP, KMeans
- **Backend:** Flask, joblib (cached model and preprocessing artifacts)
- **Data and notebooks:** pandas, NumPy, Jupyter

## Repository structure

```
.
├── carbon_emission_v4_tuning_optimized_final.ipynb   # preprocessing, training, tuning, SHAP
├── Carbon Emission.csv                               # dataset
├── carbonsense-fullstack/                            # Flask backend + web app
└── README.md
```

## Getting started

### 1. Clone

```bash
git clone https://github.com/Srujan613/Carbon-Footprint-Prediction-.git
cd Carbon-Footprint-Prediction-
```

### 2. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r carbonsense-fullstack/requirements.txt
```

### 4. Run the notebook

Open `carbon_emission_v4_tuning_optimized_final.ipynb` in Jupyter or VS Code and run all cells. It trains and tunes the models and generates the SHAP analysis.

### 5. Run the web app

```bash
cd carbonsense-fullstack
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> The Flask app loads the `.joblib` model files once at startup. If you replace the model files, fully restart the server for the change to take effect.

## Dataset

`Carbon Emission.csv` contains lifestyle attributes (body type, sex, diet, shower frequency, heating source, transport, vehicle type and distance, air travel, grocery spend, waste, screen time, clothing, internet use, energy efficiency, recycling, cooking method) and the target `CarbonEmission`.

## Contributors

- **Srujan Yalla**: explainability (SHAP) and the sustainable recommendation engine
- **Sudhanwa** (PES2UG24CS531): dataset preparation, model development and evaluation

## License and acknowledgements

Developed as part of an internship at the **Centre for Data Modeling, Analytics and Visualization (CoDMAV), PES University, Bangalore, India**.

Copyright © 2026 PES University CoDMAV and the project authors. All rights reserved. This repository is shared for academic and educational reference. Please contact the authors before reusing the code or data in other work.
