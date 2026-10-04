"""
ML service layer — wired to the real artifacts produced by
carbon_emission_v4_tuning_optimized_final.ipynb (Parts 1, 2 & 3 — same
schema and artifact filenames as v3). See backend/models/README.md
for exactly which files to copy in from your `outputs/`, `outputs/models/`,
and `outputs/explainability/` folders after running the notebook.
"""
import csv
import json
import logging
import os

import joblib
import numpy as np

from . import preprocessing

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Exact filenames saved by the notebook (cell 73). XGBoost is deliberately
# NOT in this dict — see _load_xgboost_model() below for why it needs its
# own native-format loading path instead of joblib.
MODEL_FILES = {
    "ridge":         "ridge_tuned.joblib",
    "random_forest": "rf_tuned.joblib",
    "catboost":      "catboost_tuned.joblib",
    "lightgbm":      "lgbm_tuned.joblib",
}
TREE_CANDIDATES = ["catboost", "lightgbm", "xgboost", "random_forest"]  # SHAP source, in preference order
ALL_MODEL_NAMES = list(MODEL_FILES.keys()) + ["xgboost"]

# Additional artifacts (notebook cells 110, 112) — optional. The app still
# works without them (falls back to the 5-model mean + a mock archetype),
# but real predictions require them for the stacking ensemble and archetype
# assignment to be genuine rather than heuristic. Loading these is wrapped
# in try/except in load_models() below — a corrupted or version-incompatible
# optional file must never be able to crash the whole app at startup.
STACKING_FILE = "stacking_ensemble.joblib"
ARCHETYPE_MODEL_FILE = "kmeans_archetype_model.joblib"
ARCHETYPE_LABELS_FILE = "archetype_labels.json"

_MODELS = {}
_SHAP_EXPLAINER = None
_SHAP_MODEL_NAME = None
_STACKING_MODEL = None
_ARCHETYPE_MODEL = None
_ARCHETYPE_LABELS = {}
MOCK_MODE = True
STACKING_AVAILABLE = False
ARCHETYPE_AVAILABLE = False


def _model_path(filename):
    return os.path.join(MODEL_DIR, filename)


def _pick_shap_model_name():
    """
    Mirrors the notebook's logic (model-selection cell): prefer CV_RMSE_mean
    (5-fold cross-validation on training data) over a single test-set
    Tuned_RMSE when both are available in results_table.csv — this avoids
    picking a "best" tree model that just got lucky on one test split.
    Falls back to Tuned_RMSE if CV_RMSE_mean isn't present (e.g. an older
    v3-style results_table.csv), then to TREE_CANDIDATES order.
    """
    results_path = _model_path("results_table.csv")
    name_map = {
        "Random Forest": "random_forest", "XGBoost": "xgboost",
        "LightGBM": "lightgbm", "CatBoost": "catboost", "Ridge Regression": "ridge",
    }
    if os.path.exists(results_path):
        try:
            with open(results_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            tree_rows = [r for r in rows if name_map.get(r.get("Model")) in TREE_CANDIDATES
                         and name_map[r["Model"]] in _MODELS]
            if tree_rows:
                has_cv = all(r.get("CV_RMSE_mean") not in (None, "") for r in tree_rows)
                key = "CV_RMSE_mean" if has_cv else "Tuned_RMSE"
                best = min(tree_rows, key=lambda r: float(r[key]))
                return name_map[best["Model"]]
        except Exception as e:
            logger.warning("Could not parse results_table.csv (%s), falling back.", e)

    for name in TREE_CANDIDATES:
        if name in _MODELS:
            return name
    return None


def _load_xgboost_model():
    """
    XGBoost's own docs explicitly warn against pickle/joblib for
    cross-version persistence. Fix: use XGBoost's native save_model()/
    load_model() format, version-stable, for this one model. Accepts
    either native extension (.ubj — binary, current recommended default —
    or .json — text) since the notebook may export either.
    """
    import xgboost as xgb
    for filename in ("xgb_tuned.ubj", "xgb_tuned.json"):
        path = _model_path(filename)
        if os.path.exists(path):
            model = xgb.XGBRegressor()
            model.load_model(path)
            logger.info("XGBoost %s loaded natively from %s (version-stable format, no pickle involved).",
                        xgb.__version__, path)
            return model
    logger.warning(
        "XGBoost model not found (looked for xgb_tuned.ubj and xgb_tuned.json). "
        "Note: XGBoost needs a native export, not xgb_tuned.joblib — see models/README.md."
    )
    return None


def load_models():
    """Called once at app startup (see app.py)."""
    global MOCK_MODE, _SHAP_EXPLAINER, _SHAP_MODEL_NAME
    global _STACKING_MODEL, _ARCHETYPE_MODEL, _ARCHETYPE_LABELS
    global STACKING_AVAILABLE, ARCHETYPE_AVAILABLE

    for name, filename in MODEL_FILES.items():
        path = _model_path(filename)
        if os.path.exists(path):
            _MODELS[name] = joblib.load(path)
        else:
            logger.warning("Model '%s' not found at %s.", name, path)

    xgb_model = _load_xgboost_model()
    if xgb_model is not None:
        _MODELS["xgboost"] = xgb_model

    preproc_ok = preprocessing.artifacts_present()
    if not preproc_ok:
        logger.warning("Preprocessing artifacts missing from backend/models/ — see models/README.md.")

    MOCK_MODE = not (all(name in _MODELS for name in ALL_MODEL_NAMES) and preproc_ok)

    if MOCK_MODE:
        logger.warning(
            "ml_service is in MOCK_MODE: models and/or preprocessing artifacts are "
            "missing from backend/models/. Predictions are heuristic placeholders."
        )
    else:
        _SHAP_MODEL_NAME = _pick_shap_model_name()
        if _SHAP_MODEL_NAME:
            import shap
            _SHAP_EXPLAINER = shap.TreeExplainer(_MODELS[_SHAP_MODEL_NAME])
            logger.info("SHAP explainer ready on '%s'.", _SHAP_MODEL_NAME)
        logger.info("All 5 models + preprocessing artifacts loaded — serving real predictions.")

    # ── Stacking ensemble (notebook cell 112) — optional add-on. Wrapped in
    # try/except deliberately: this pickle contains a full nested XGBoost
    # model (StackingRegressor's base learners), which carries the exact
    # same pickle-fragility risk as the standalone XGBoost model — but
    # unlike that one, this whole object can't be re-saved in a native
    # format (StackingRegressor itself has no such option). If it fails to
    # unpickle (corrupted file, version mismatch, etc.), that must degrade
    # to "stacking unavailable," not crash the entire app at startup. ──────
    stacking_path = _model_path(STACKING_FILE)
    if os.path.exists(stacking_path):
        try:
            _STACKING_MODEL = joblib.load(stacking_path)
            STACKING_AVAILABLE = True
            logger.info("Stacking ensemble loaded from %s.", stacking_path)
        except Exception as e:
            logger.warning(
                "Stacking ensemble at %s failed to load (%s: %s) — likely a corrupted "
                "file or an XGBoost version mismatch in its nested base learner. "
                "/api/predict will omit 'stacking' and fall back to the 5-model mean. "
                "Try re-copying/re-exporting this one file.",
                stacking_path, type(e).__name__, e,
            )
    else:
        logger.warning(
            "Stacking ensemble not found at %s — /api/predict will omit 'stacking' "
            "and fall back to the 5-model mean as the headline number. See models/README.md.",
            stacking_path,
        )

    # ── KMeans lifestyle archetypes (notebook cell 110) — optional add-on,
    # same try/except rationale as stacking above. ────────────────────────
    archetype_model_path = _model_path(ARCHETYPE_MODEL_FILE)
    archetype_labels_path = _model_path(ARCHETYPE_LABELS_FILE)
    if os.path.exists(archetype_model_path) and os.path.exists(archetype_labels_path):
        try:
            _ARCHETYPE_MODEL = joblib.load(archetype_model_path)
            with open(archetype_labels_path, encoding="utf-8") as f:
                _ARCHETYPE_LABELS = json.load(f)  # {"0": "High-Emission / Frequent Flyer", ...}
            ARCHETYPE_AVAILABLE = True
            logger.info("KMeans archetype model + %d labels loaded.", len(_ARCHETYPE_LABELS))
        except Exception as e:
            logger.warning(
                "Archetype model at %s failed to load (%s: %s) — /api/predict will fall "
                "back to a heuristic archetype guess. Try re-copying/re-exporting this file.",
                archetype_model_path, type(e).__name__, e,
            )
    else:
        logger.warning(
            "Archetype model/labels not found at %s / %s — /api/predict will fall back "
            "to a heuristic archetype guess. See models/README.md.",
            archetype_model_path, archetype_labels_path,
        )

def predict_all(payload: dict) -> dict:
    if MOCK_MODE:
        return _mock_predict(payload)

    X_unscaled = preprocessing.preprocess_new_input(payload)
    scaler = preprocessing.load_preprocessing_artifacts()["scaler"]
    X_scaled = scaler.transform(X_unscaled)

    raw_preds = {}
    for name, model in _MODELS.items():
        X = X_scaled if name == "ridge" else X_unscaled
        raw_preds[name] = float(model.predict(X)[0])

    ensemble_raw = sum(raw_preds.values()) / len(raw_preds)

    preds = {name: max(0.0, v) for name, v in raw_preds.items()}  # clamp per-model for display only
    preds["ensemble"] = max(0.0, ensemble_raw)                     # clamp the RAW average, not the clamped values

    # Stacking ensemble (cell 112) — base learners (RF/XGB/LGBM/CatBoost) all
    # operate on raw unscaled features, same as the individual tree models
    # above, so no extra scaling step is needed here.
    if STACKING_AVAILABLE:
        stacking_raw = float(_STACKING_MODEL.predict(X_unscaled)[0])
        preds["stacking"] = max(0.0, stacking_raw)

    return preds


def headline_total(preds: dict) -> float:
    """
    The number shown as the user's primary result. CatBoost is the
    headline model — it's the strongest individual performer by CV RMSE
    among the five tuned models (see cv_summary_df / results_table.csv).
    Falls back to the stacking ensemble, then the simple 5-model mean, if
    for some reason 'catboost' isn't in the predictions dict.
    """
    return preds.get("catboost", preds.get("stacking", preds.get("ensemble", 0.0)))


def explain(payload: dict, top_n: int = 6) -> list:
    """
    Returns [{"feature": str, "value": float, "direction": "pos"|"neg"}, ...]
    sorted by |value| descending. Uses shap.TreeExplainer on the model chosen
    by _pick_shap_model_name(), exactly like the notebook (cell 85).
    """
    if MOCK_MODE:
        return _mock_explain(payload, top_n)

    X_unscaled = preprocessing.preprocess_new_input(payload)
    feature_names = preprocessing.get_feature_names()

    raw_values = _SHAP_EXPLAINER.shap_values(X_unscaled)
    if isinstance(raw_values, list):
        raw_values = raw_values[0]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, 0]
    row_values = values[0]

    ranked = sorted(zip(feature_names, row_values), key=lambda t: abs(t[1]), reverse=True)[:top_n]
    return [
        {"feature": _pretty_feature_name(feat), "value": round(float(val), 0),
         "direction": "pos" if val >= 0 else "neg"}
        for feat, val in ranked
    ]


def assign_archetype(payload: dict) -> dict:
    """
    Assign a lifestyle archetype cluster to a new user (notebook cell 110).

    IMPORTANT: km_final was fit on X_train_s (scaled features) — the same
    scaler.joblib used for Ridge — because KMeans is distance-based and
    unscaled raw features (e.g. Vehicle Monthly Distance Km in the
    thousands) would otherwise dominate binary/ordinal ones. Calling
    .predict() on raw/unscaled features here would silently misclassify
    almost everyone into whichever cluster happens to have the highest
    average vehicle distance. scaler.joblib MUST be applied first.
    """
    if MOCK_MODE or not ARCHETYPE_AVAILABLE:
        return _mock_archetype(payload)

    X_unscaled = preprocessing.preprocess_new_input(payload)
    scaler = preprocessing.load_preprocessing_artifacts()["scaler"]
    X_scaled = scaler.transform(X_unscaled)

    cluster = int(_ARCHETYPE_MODEL.predict(X_scaled)[0])
    label = _ARCHETYPE_LABELS.get(str(cluster), f"Cluster {cluster}")
    return {"cluster": cluster, "label": label}


def _mock_archetype(payload: dict) -> dict:
    """Heuristic stand-in when the real KMeans model/labels aren't loaded."""
    scores = _mock_scores(payload)
    emission_tag = "High-Emission" if sum(scores.values()) >= 3500 else "Low-Emission"
    descriptors = []
    if payload.get("Frequency of Traveling by Air") in ("frequently", "very frequently"):
        descriptors.append("Frequent Flyer")
    if payload.get("Transport") == "private":
        descriptors.append("Private-Transport")
    if payload.get("Heating Energy Source") not in ("electricity", "solar"):
        descriptors.append("Non-Electric-Heating")
    if not descriptors:
        descriptors.append("General Lifestyle")
    return {"cluster": None, "label": f"{emission_tag} / {', '.join(descriptors)} (estimated)"}


def get_model_metrics() -> dict:
    """
    Real per-model accuracy for the Models page — read straight from
    results_table.csv (cell 65/73), never hardcoded. Prefers CV_R2_mean /
    CV_RMSE_mean (5-fold, leakage-safe) over the single-split Tuned_R2 /
    Tuned_RMSE columns, same preference order used everywhere else in this
    backend for picking/reporting on models.

    Returns {"available": bool, "models": {display_name: {...}}}. When
    results_table.csv isn't present, 'available' is False and the frontend
    should show a "connect real metrics" placeholder rather than fake
    numbers — this app has already shipped hardcoded placeholder R² values
    once, on the Models page, which is exactly the bug this replaces.
    """
    results_path = _model_path("results_table.csv")
    if not os.path.exists(results_path):
        return {"available": False, "models": {}}

    display_names = {
        "Ridge Regression": "ridge", "Random Forest": "random_forest",
        "CatBoost": "catboost", "LightGBM": "lightgbm", "XGBoost": "xgboost",
    }
    out = {}
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = display_names.get(row.get("Model"))
            if not key:
                continue
            has_cv = row.get("CV_R2_mean") not in (None, "")
            r2 = float(row["CV_R2_mean"]) if has_cv else float(row.get("Tuned_R2", 0))
            rmse = float(row["CV_RMSE_mean"]) if has_cv else float(row.get("Tuned_RMSE", 0))
            out[key] = {
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(float(row.get("Tuned_MAE", 0)), 2),
                "source": "cv" if has_cv else "test_split",
            }
    return {"available": bool(out), "models": out}


def get_shap_model():
    """Used by recommender.py to run counterfactual what-ifs on the same model SHAP explains."""
    return _MODELS.get(_SHAP_MODEL_NAME) if not MOCK_MODE else None


def _pretty_feature_name(raw_name: str) -> str:
    overrides = {
        "Vehicle Monthly Distance Km": "Vehicle Monthly Distance",
        "Frequency of Traveling by Air": "Air Travel Frequency",
        "How Many New Clothes Monthly": "New Clothes / Month",
        "How Long TV PC Daily Hour": "Daily TV / PC Hours",
        "How Long Internet Daily Hour": "Daily Internet Hours",
        "Waste Bag Weekly Count": "Waste Bags / Week",
        "Waste Bag Size": "Waste Bag Size",
        "Monthly Grocery Bill": "Monthly Grocery Bill",
        "Social Activity": "Social Activity",
        "Energy efficiency": "Energy Efficiency",
    }
    if raw_name in overrides:
        return overrides[raw_name]
    if "_" in raw_name:
        prefix, _, suffix = raw_name.partition("_")
        return f"{prefix}: {suffix.replace('_', ' ').title()}"
    return raw_name


# ─────────────────────────────────────────────────────────────
# Mock fallback — used only until real artifacts are dropped in.
# ─────────────────────────────────────────────────────────────
_DIET_MAP = {"omnivore": 1500, "pescatarian": 900, "vegetarian": 580, "vegan": 280}
_HEAT_MAP = {"coal": 2900, "natural gas": 1900, "wood": 1100, "electricity": 700, "solar": 150}
_AIR_MAP = {"never": 0, "rarely": 480, "frequently": 1700, "very frequently": 3800}
_EFF_MAP = {"No": 0, "Sometimes": -290, "Yes": -520}


def _mock_scores(payload: dict) -> dict:
    diet = _DIET_MAP.get(payload.get("Diet"), 800)
    heating = _HEAT_MAP.get(payload.get("Heating Energy Source"), 1000)
    air = _AIR_MAP.get(payload.get("Frequency of Traveling by Air"), 0)
    dist = float(payload.get("Vehicle Monthly Distance Km") or 0)
    grocery = float(payload.get("Monthly Grocery Bill") or 200)
    clothes = float(payload.get("How Many New Clothes Monthly") or 0)
    tv = float(payload.get("How Long TV PC Daily Hour") or 0)
    net = float(payload.get("How Long Internet Daily Hour") or 0)
    eff = _EFF_MAP.get(payload.get("Energy efficiency"), 0)

    return {
        "transport": dist * 0.19 + air,
        "home": heating + grocery * 0.45 + eff,
        "diet": diet,
        "lifestyle": clothes * 38 + tv * 14 + net * 7,
    }


def _mock_predict(payload: dict) -> dict:
    scores = _mock_scores(payload)
    base_total = sum(scores.values())
    jitter = {"ridge": -0.973, "random_forest": 0.081, "catboost": 0.191, "lightgbm": 0.219, "xgboost": 0.116}
    preds = {name: max(0.0, base_total * (1 + j / 100)) for name, j in jitter.items()}
    preds["ensemble"] = sum(preds.values()) / len(preds)
    return {k: round(v, 1) for k, v in preds.items()}


def _mock_explain(payload: dict, top_n: int) -> list:
    scores = _mock_scores(payload)
    contributions = [
        ("Vehicle Monthly Distance", scores["transport"] * 0.7),
        ("Air Travel Frequency", scores["transport"] * 0.3),
        ("Heating Energy Source", scores["home"] * 0.75),
        ("Monthly Grocery Bill", scores["home"] * 0.25),
        ("Diet", scores["diet"]),
        ("Energy Efficiency", -abs(_EFF_MAP.get(payload.get("Energy efficiency"), 0))),
        ("New Clothes / Month", scores["lifestyle"] * 0.6),
        ("Daily TV / PC Hours", scores["lifestyle"] * 0.25),
        ("Daily Internet Hours", scores["lifestyle"] * 0.15),
    ]
    contributions = [(f, v) for f, v in contributions if abs(v) > 1]
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    contributions = contributions[:top_n]
    return [{"feature": f, "value": round(v, 0), "direction": "pos" if v >= 0 else "neg"} for f, v in contributions]
