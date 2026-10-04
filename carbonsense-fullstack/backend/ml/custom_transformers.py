"""
Exact copies of the custom sklearn transformer classes defined in
carbon_emission_v4_tuning_optimized_final.ipynb (cells 22–24). These are
NOT imported directly by name anywhere in this backend — their only job is
to exist somewhere importable so that `joblib.load("preprocessing_pipeline.joblib")`
can successfully unpickle a fitted Pipeline that contains instances of them.

WHY THIS FILE EXISTS:
When the notebook saved `preprocessing_pipeline.joblib`, these classes were
defined in the notebook's own top-level (`__main__`) namespace. Pickle
records *where a class was defined*, not its source code — so unpickling
requires a class of the same name to be importable from that same location
at load time. Since Flask runs `app.py` as `__main__` instead of the
notebook, `__main__.HardBoundsClipper` etc. don't exist by default, which
is exactly the `AttributeError: Can't get attribute 'HardBoundsClipper' on
<module '__main__' ...>` error this fixes.

preprocessing.py registers these classes into `sys.modules["__main__"]`
before ever calling joblib.load, so this works regardless of which script
is actually the entry point.

IMPORTANT: if you re-run the notebook and it changes any of these classes'
logic, copy the updated versions back into this file too, or predictions
will silently use stale transform logic.
"""
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class HardBoundsClipper(BaseEstimator, TransformerMixin):
    """Clips numeric inputs to hard physical bounds before imputation or
    winsorization. These bounds are domain knowledge (nobody watches 500
    hours of TV a day), not learned statistics — so, like the two
    transformers below, this is safe to fit() on anything."""

    HARD_BOUNDS = {
        "Monthly Grocery Bill":          (0,   5000),
        "Vehicle Monthly Distance Km":   (0,  50000),
        "Waste Bag Weekly Count":        (0,    100),
        "How Long TV PC Daily Hour":     (0,     24),
        "How Many New Clothes Monthly":  (0,    500),
        "How Long Internet Daily Hour":  (0,     24),
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.HARD_BOUNDS.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(lo, hi)
        return X


class VehicleTypeFiller(BaseEstimator, TransformerMixin):
    """Fills missing Vehicle Type with the domain sentinel 'none' — a fixed
    value, not learned from data, so safe before or after the split."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Vehicle Type" in X.columns:
            X["Vehicle Type"] = X["Vehicle Type"].fillna("none")
        return X


class MultiLabelBinarizer(BaseEstimator, TransformerMixin):
    """Parses the Recycling / Cooking_With list-string columns into binary
    flag columns. Deterministic given the raw value, no fitting needed.

    NOTE: this deliberately shadows sklearn.preprocessing.MultiLabelBinarizer
    by name — it's a different, custom implementation. Don't import
    sklearn's version into this module under the same name."""

    RECYCLING_ITEMS = ["Paper", "Plastic", "Glass", "Metal"]
    COOKING_ITEMS   = ["Stove", "Oven", "Microwave", "Grill", "Airfryer"]

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _parse(val):
        try:
            return literal_eval(val) if isinstance(val, str) else []
        except Exception:
            return []

    def transform(self, X):
        X = X.copy()

        if "Recycling" in X.columns:
            parsed = X["Recycling"].apply(self._parse)
            for item in self.RECYCLING_ITEMS:
                X[f"Recycle_{item}"] = parsed.apply(lambda v: int(item in v))
            X = X.drop(columns=["Recycling"])
        else:
            for item in self.RECYCLING_ITEMS:
                col = f"Recycle_{item}"
                if col not in X.columns:
                    X[col] = 0

        if "Cooking_With" in X.columns:
            parsed = X["Cooking_With"].apply(self._parse)
            for item in self.COOKING_ITEMS:
                X[f"Cook_{item}"] = parsed.apply(lambda v: int(item in v))
            X = X.drop(columns=["Cooking_With"])
        else:
            for item in self.COOKING_ITEMS:
                col = f"Cook_{item}"
                if col not in X.columns:
                    X[col] = 0

        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [low_q, high_q] percentiles LEARNED FROM
    WHATEVER DATA IS PASSED TO fit()."""

    def __init__(self, low_q=0.01, high_q=0.99):
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in X.select_dtypes(include="number").columns:
            self.bounds_[col] = (X[col].quantile(self.low_q), X[col].quantile(self.high_q))
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X


class MedianModeImputer(BaseEstimator, TransformerMixin):
    """Median for numeric columns, mode for categorical columns — both
    computed only on whatever is passed to fit() (X_train).

    transform() also CREATES a column outright (filled entirely with the
    learned median/mode) if it's missing altogether, not just if it's
    present-but-NaN. This matters at inference time: a partial user input
    dict means those columns don't exist in the row's DataFrame at all."""

    def fit(self, X, y=None):
        self.medians_ = {col: X[col].median() for col in X.select_dtypes(include="number").columns}
        self.modes_ = {}
        for col in X.select_dtypes(include=["object", "category", "str"]).columns:
            mode = X[col].mode()
            self.modes_[col] = mode.iloc[0] if not mode.empty else None
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.medians_.items():
            if col not in X.columns:
                X[col] = val
            else:
                X[col] = X[col].fillna(val)
        for col, val in self.modes_.items():
            if val is None:
                continue
            if col not in X.columns:
                X[col] = val
            else:
                X[col] = X[col].fillna(val)
        return X


class OrdinalEncoderStep(BaseEstimator, TransformerMixin):
    """Wraps sklearn's OrdinalEncoder with explicit, human-specified
    category orders."""

    def __init__(self, specs):
        self.specs = specs  # {column: [ordered categories]}

    def fit(self, X, y=None):
        self.cols_ = list(self.specs.keys())
        self.encoder_ = OrdinalEncoder(
            categories=[self.specs[c] for c in self.cols_],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.encoder_.fit(X[self.cols_])
        return self

    def transform(self, X):
        X = X.copy()
        present = [c for c in self.cols_ if c in X.columns]
        X[present] = self.encoder_.transform(X[present])
        return X


class OneHotEncoderStep(BaseEstimator, TransformerMixin):
    """One-hot encodes nominal columns with a fixed reference (dropped)
    category per column."""

    def __init__(self, reference_categories):
        self.reference_categories = reference_categories  # {col: ref_category}

    def fit(self, X, y=None):
        self.cols_ = list(self.reference_categories.keys())
        self.categories_ = {}
        for col in self.cols_:
            ref = self.reference_categories[col]
            others = sorted(c for c in X[col].dropna().unique() if c != ref)
            self.categories_[col] = [ref] + others
        return self

    def transform(self, X):
        X = X.copy()
        present = [c for c in self.cols_ if c in X.columns]
        for col in present:
            X[col] = pd.Categorical(X[col], categories=self.categories_[col])
        X = pd.get_dummies(X, columns=present, drop_first=True, dtype=int)

        for col in self.cols_:
            for cat in self.categories_[col][1:]:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in X.columns:
                    X[dummy_col] = 0
        return X


class CollinearityPruner(BaseEstimator, TransformerMixin):
    """Drops one feature from any pair with |correlation| > threshold,
    keeping whichever correlates more strongly with the target."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        y_series = pd.Series(np.asarray(y), index=X.index)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        target_corr = X.corrwith(y_series).abs()

        to_drop = set()
        self.high_corr_pairs_ = []
        for col in upper.columns:
            partners = upper.index[upper[col] > self.threshold].tolist()
            for partner in partners:
                self.high_corr_pairs_.append((col, partner, upper.loc[partner, col]))
                if target_corr.get(col, 0) >= target_corr.get(partner, 0):
                    to_drop.add(partner)
                else:
                    to_drop.add(col)
        self.drop_cols_ = sorted(to_drop)
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X.columns])


class ImportancePruner(BaseEstimator, TransformerMixin):
    """Fits a quick CatBoost reference model on fit() data ONLY and drops
    the lowest-importance features down to a target count."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    @staticmethod
    def _target_n_features(n_available):
        return min(40, max(10, int(n_available * 0.85)))

    def fit(self, X, y=None):
        from catboost import CatBoostRegressor
        ref_model = CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.08,
            random_state=self.random_state, verbose=0,
        )
        ref_model.fit(X, y)
        importances = pd.Series(ref_model.get_feature_importance(), index=X.columns).sort_values()

        self.target_n_features_ = self._target_n_features(X.shape[1])
        n_to_drop = max(0, X.shape[1] - self.target_n_features_)
        self.drop_cols_ = importances.head(n_to_drop).index.tolist()
        self.importances_ = importances
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X.columns])


# The exact set of class names pickle needs to find in `__main__` — see
# the registration step this module's docstring describes, done in
# preprocessing.py.
ALL_CUSTOM_TRANSFORMERS = [
    HardBoundsClipper,
    VehicleTypeFiller,
    MultiLabelBinarizer,
    Winsorizer,
    MedianModeImputer,
    OrdinalEncoderStep,
    OneHotEncoderStep,
    CollinearityPruner,
    ImportancePruner,
]
