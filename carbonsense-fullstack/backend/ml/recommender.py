"""
Real counterfactual recommendations — ported from the training notebook
(carbon_emission_v4_tuning_optimized_final.ipynb, cell 114:
build_action_catalogue / generate_whatif). Instead of a hand-written
library of tips, this tests actual one-hot-encoded feature changes against
the real trained tree model and reports the model's own predicted
reduction for each, exactly like the notebook's what-if analysis.

Encoding conventions (drop_first=True reference categories, from the
notebook's one-hot step):
    Transport     -> "private"     (all Transport_* = 0 means private car)
    Vehicle Type  -> "none"        (all Vehicle Type_* = 0 means no vehicle)
    Heating       -> "electricity" (all Heating_* = 0 means electric)
    Diet          -> "vegetarian"  (all Diet_* = 0 means vegetarian)
"""
from . import ml_service, preprocessing

# Presentation metadata for each action label the catalogue can produce.
# Falls back to a generic card if a label isn't in this map.
ACTION_META = {
    "Stop air travel entirely":        {"icon": "✈️", "title": "Cut Out Air Travel",
        "desc": "Flying is one of the most carbon-intensive activities per hour. Eliminating flights entirely has an outsized effect on your total footprint."},
    "Reduce air travel to rarely":     {"icon": "✈️", "title": "Fly Less Often",
        "desc": "Cutting back to occasional flights instead of frequent ones meaningfully lowers your travel emissions without eliminating travel altogether."},
    "Switch to public transport":      {"icon": "🚌", "title": "Switch to Public Transport",
        "desc": "Moving your daily commute off a private vehicle and onto public transit cuts per-trip emissions substantially."},
    "Switch to walking / cycling":     {"icon": "🚲", "title": "Walk or Cycle Instead",
        "desc": "For trips that don't need a vehicle, walking or cycling brings transport emissions close to zero."},
    "Switch to electric vehicle":      {"icon": "🔌", "title": "Switch to an Electric Vehicle",
        "desc": "Replacing a combustion vehicle with an EV removes tailpipe emissions from your monthly driving distance."},
    "Switch to hybrid vehicle":        {"icon": "🚗", "title": "Switch to a Hybrid Vehicle",
        "desc": "A hybrid meaningfully reduces fuel-driven emissions compared to a standard petrol or diesel vehicle."},
    "Give up personal vehicle":        {"icon": "🚶", "title": "Go Car-Free",
        "desc": "Dropping a personal vehicle entirely in favor of transit, walking, or cycling removes vehicle emissions altogether."},
    "Switch heating to electricity":   {"icon": "🔥", "title": "Switch to Electric Heating",
        "desc": "Moving off coal, gas, or wood heating onto electricity (especially from a cleaner grid) cuts home heating emissions."},
    "Switch to vegetarian diet":       {"icon": "🥗", "title": "Switch to a Vegetarian Diet",
        "desc": "Cutting meat from your diet is one of the highest-impact personal changes for lowering food-related emissions."},
    "Switch to vegan diet":            {"icon": "🌱", "title": "Switch to a Vegan Diet",
        "desc": "Removing all animal products goes a step further than vegetarian and further reduces diet-related emissions."},
    "Recycle paper and plastic":       {"icon": "♻️", "title": "Recycle Paper & Plastic",
        "desc": "Adding these two common materials to your recycling routine is a low-effort way to cut embedded waste emissions."},
    "Recycle all materials":           {"icon": "♻️", "title": "Recycle Everything You Can",
        "desc": "Recycling paper, plastic, glass, and metal together captures the largest available reduction from your waste stream."},
}

_DEFAULT_META = {"icon": "🌍", "title": "Lifestyle Change", "desc": "A tested change that reduces your predicted footprint."}


def build_action_catalogue(feature_names):
    """
    IMPORTANT — dependent-feature consistency (ported from notebook v4,
    cell 114): "Vehicle Monthly Distance Km" is the single highest-importance
    feature in this model. Any action that drops personal-vehicle use
    (public transport, walking/cycling, or giving up the vehicle entirely)
    must also zero out this feature — otherwise the perturbed row says
    "no vehicle" while still carrying a large vehicle-distance value, a
    combination the model never saw in training, which silently understates
    the true predicted impact of the action. Actions that keep driving but
    change *what* is driven (electric/hybrid) correctly leave it untouched.
    """
    fn = set(feature_names)
    actions = []
    distance_col = "Vehicle Monthly Distance Km" if "Vehicle Monthly Distance Km" in fn else None

    air_col = next((c for c in feature_names if c == "Frequency of Traveling by Air"), None)
    if air_col:
        actions.append(("Stop air travel entirely", {air_col: 0}))
        actions.append(("Reduce air travel to rarely", {air_col: 1}))

    transport_cols = [c for c in feature_names if c.startswith("Transport_")]
    if transport_cols:
        zero_t = {c: 0 for c in transport_cols}
        pub = dict(zero_t)
        if "Transport_public" in fn:
            pub["Transport_public"] = 1
        if distance_col:
            pub[distance_col] = 0  # no longer driving a personal vehicle
        actions.append(("Switch to public transport", pub))
        walk = dict(zero_t)
        if "Transport_walk/bicycle" in fn:
            walk["Transport_walk/bicycle"] = 1
        if distance_col:
            walk[distance_col] = 0  # no longer driving a personal vehicle
        actions.append(("Switch to walking / cycling", walk))

    vehicle_cols = [c for c in feature_names if c.startswith("Vehicle Type_")]
    if vehicle_cols:
        zero_v = {c: 0 for c in vehicle_cols}
        # Electric/hybrid: still driving the same distance, just a
        # different vehicle type — distance_col is intentionally NOT touched.
        elec = dict(zero_v)
        if "Vehicle Type_electric" in fn:
            elec["Vehicle Type_electric"] = 1
        actions.append(("Switch to electric vehicle", elec))
        hybrid = dict(zero_v)
        if "Vehicle Type_hybrid" in fn:
            hybrid["Vehicle Type_hybrid"] = 1
        actions.append(("Switch to hybrid vehicle", hybrid))
        # Giving up the vehicle entirely: distance must drop too.
        give_up = dict(zero_v)
        if distance_col:
            give_up[distance_col] = 0
        actions.append(("Give up personal vehicle", give_up))

    heat_cols = [c for c in feature_names if c.startswith("Heating Energy Source_")]
    if heat_cols:
        actions.append(("Switch heating to electricity", {c: 0 for c in heat_cols}))

    diet_cols = [c for c in feature_names if c.startswith("Diet_")]
    if diet_cols:
        zero_d = {c: 0 for c in diet_cols}
        actions.append(("Switch to vegetarian diet", dict(zero_d)))
        vegan = dict(zero_d)
        if "Diet_vegan" in fn:
            vegan["Diet_vegan"] = 1
        actions.append(("Switch to vegan diet", vegan))

    recycle_cols = [c for c in feature_names if c.startswith("Recycle_")]
    if recycle_cols:
        paper_plastic = {c: 1 for c in ["Recycle_Paper", "Recycle_Plastic"] if c in fn}
        if paper_plastic:
            actions.append(("Recycle paper and plastic", paper_plastic))
        actions.append(("Recycle all materials", {c: 1 for c in recycle_cols}))

    return actions


def generate_whatif(user_row, model, feature_names, top_n=5):
    baseline = float(model.predict(user_row.reshape(1, -1))[0])
    catalogue = build_action_catalogue(feature_names)
    results = []

    for label, changes in catalogue:
        relevant = {k: v for k, v in changes.items() if k in feature_names}
        if not relevant:
            continue
        modified = user_row.copy()
        for feat, val in relevant.items():
            modified[feature_names.index(feat)] = val
        new_emission = float(model.predict(modified.reshape(1, -1))[0])
        reduction = baseline - new_emission
        if reduction > 0:
            results.append({"label": label, "reduction_kg": reduction})

    results.sort(key=lambda r: r["reduction_kg"], reverse=True)
    return results[:top_n]


def _priority_for_rank(idx: int) -> str:
    if idx < 2:
        return "high"
    if idx < 4:
        return "mid"
    return "low"


def _priority_label(priority: str) -> str:
    return {"high": "High Impact", "mid": "Medium Impact", "low": "Quick Win"}[priority]


def build_recommendations(payload: dict, shap_items: list, top_n: int = 5) -> list:
    """
    Real path: run the notebook's counterfactual what-if engine against the
    actual trained tree model, using the user's own preprocessed row.

    Mock-mode fallback: derive suggestions from the mock SHAP output instead
    (no real model available to test counterfactuals against).
    """
    model = ml_service.get_shap_model()

    if model is not None:
        X = preprocessing.preprocess_new_input(payload)
        feature_names = preprocessing.get_feature_names()
        whatifs = generate_whatif(X[0], model, feature_names, top_n=top_n)
        recs = []
        for w in whatifs:
            meta = ACTION_META.get(w["label"], _DEFAULT_META)
            recs.append({**meta, "impact_kg": round(w["reduction_kg"], 0)})
    else:
        recs = _mock_recommendations(shap_items)

    for idx, rec in enumerate(recs):
        rec["priority"] = _priority_for_rank(idx)
        rec["priority_label"] = _priority_label(rec["priority"])
    return recs


def _mock_recommendations(shap_items: list) -> list:
    mock_map = {
        "Vehicle Monthly Distance": ACTION_META["Switch to public transport"],
        "Air Travel Frequency": ACTION_META["Reduce air travel to rarely"],
        "Heating Energy Source": ACTION_META["Switch heating to electricity"],
        "Diet": ACTION_META["Switch to vegetarian diet"],
    }
    recs = []
    for item in shap_items:
        if item["direction"] != "pos":
            continue
        meta = mock_map.get(item["feature"])
        if meta:
            recs.append({**meta, "impact_kg": abs(item["value"])})
    recs.sort(key=lambda r: r["impact_kg"], reverse=True)
    return recs
