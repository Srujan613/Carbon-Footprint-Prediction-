"""
Ported directly from the training notebook
(carbon_emission_v4_tuning_optimized_final.ipynb, cell 36 —
`preprocess_new_input`). As of v4, ALL preprocessing (hard bounds,
vehicle-type fill, multi-label parsing, median/mode imputation,
winsorization, ordinal/one-hot encoding, and both feature-pruning steps)
lives inside a single fitted sklearn Pipeline object saved as
`preprocessing_pipeline.joblib` — the pipeline itself is now the single
source of truth, replacing the 7 separate artifact files (ordinal encoder,
column lists, medians, modes, winsor bounds) that v3 required.

Only change from the notebook version: artifacts are loaded from
backend/models/ instead of outputs/.

Ridge was trained on the *scaled* version of this pipeline's output — see
ml_service.py, which applies scaler.joblib on top of what this function
returns before calling ridge.predict().
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

from . import custom_transformers as _ct

# joblib/pickle records the *module* a class was defined in at save time.
# These classes were defined in the training notebook's own top-level
# (__main__) namespace, so unpickling preprocessing_pipeline.joblib needs
# a class of the same name importable from __main__ — regardless of
# whether the actual running script is app.py, a test script, etc. This
# registers them once, here, before load_preprocessing_artifacts() ever
# runs joblib.load().
_main_module = sys.modules["__main__"]
for _cls in _ct.ALL_CUSTOM_TRANSFORMERS:
    setattr(_main_module, _cls.__name__, _cls)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TARGET_NAME = "CarbonEmission"

_ARTIFACTS_CACHE = {}


def _artifact_path(name):
    return os.path.join(MODEL_DIR, name)


def artifacts_present() -> bool:
    required = [
        "preprocessing_pipeline.joblib",
        "feature_names.json",
        "scaler.joblib",
    ]
    return all(os.path.exists(_artifact_path(f)) for f in required)


def load_preprocessing_artifacts():
    """Load the fitted pipeline, scaler, and feature list once and cache them.

    Loading these from disk on every prediction call would be a serious
    bottleneck for a web app serving many requests per second, so we load
    once and reuse the cached copies on subsequent calls.
    """
    if _ARTIFACTS_CACHE:
        return _ARTIFACTS_CACHE

    _ARTIFACTS_CACHE["pipeline"] = joblib.load(_artifact_path("preprocessing_pipeline.joblib"))
    _ARTIFACTS_CACHE["scaler"] = joblib.load(_artifact_path("scaler.joblib"))
    import json
    with open(_artifact_path("feature_names.json"), encoding="utf-8") as f:
        _ARTIFACTS_CACHE["final_feats"] = json.load(f)

    return _ARTIFACTS_CACHE


def preprocess_new_input(raw_input: dict, scale: bool = False) -> np.ndarray:
    """
    Preprocess a single new user input dict -> numpy array for predict().

    `raw_input` keys must match the *original dataset column names*, e.g.
    "Body Type", "Vehicle Monthly Distance Km", "Recycling" (a list or a
    stringified list like '["Paper","Plastic"]'), etc. — see features.py.

    Runs the raw input through the exact same fitted Pipeline used for
    X_train_raw / X_test_raw, then aligns columns to the training feature
    order (any column the pipeline didn't produce for this row — e.g. an
    unseen one-hot category, or a field the user left out entirely — is
    filled with 0; missing *values* the pipeline knows about are filled
    with its stored training medians/modes internally).

    Parameters
    ----------
    raw_input : dict
        Single new user's raw field values (same schema as the source CSV,
        minus the CarbonEmission target).
    scale : bool
        If True, also apply the fitted StandardScaler (for Ridge). Tree
        models should call this with scale=False (the default). Note:
        ml_service.py currently applies the scaler itself on top of this
        function's default (unscaled) output rather than passing scale=True
        — either path is equivalent, since it's the same scaler.joblib.
    """
    artifacts = load_preprocessing_artifacts()
    pipeline = artifacts["pipeline"]
    feature_names = artifacts["final_feats"]

    row = pd.DataFrame([raw_input]).drop(columns=[TARGET_NAME], errors="ignore")
    processed = pipeline.transform(row)

    # Reindexing to the training feature order/list is the safety net:
    # it guarantees correct column alignment regardless of any row-level
    # quirks in one-hot dummy generation for a single input row, and fills
    # in 0 for anything missing (unseen category, omitted optional field).
    processed = processed.reindex(columns=feature_names, fill_value=0)

    if scale:
        scaled = artifacts["scaler"].transform(processed)
        return np.asarray(scaled, dtype=float)

    return processed.values.astype(float)


def get_feature_names():
    return load_preprocessing_artifacts()["final_feats"]
