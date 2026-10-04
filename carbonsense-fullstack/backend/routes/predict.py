from flask import Blueprint, request, jsonify

from ml import ml_service
from ml.features import validate_payload
from ml.recommender import build_recommendations
from config import Config

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mock_mode": ml_service.MOCK_MODE,
        "models_loaded": list(ml_service._MODELS.keys()),
        "stacking_available": ml_service.STACKING_AVAILABLE,
        "archetype_available": ml_service.ARCHETYPE_AVAILABLE,
    })


@predict_bp.route("/api/models", methods=["GET"])
def models():
    return jsonify(ml_service.get_model_metrics())


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    errors = validate_payload(payload)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    predictions = ml_service.predict_all(payload)
    shap_items = ml_service.explain(payload)
    recommendations = build_recommendations(payload, shap_items)
    archetype = ml_service.assign_archetype(payload)

    # Headline number is CatBoost — the strongest individual model by CV
    # RMSE, per the project's deliberate choice to headline it over both
    # the stacking ensemble and the simple 5-model mean (see
    # ml_service.headline_total for the fallback order if catboost is
    # ever missing).
    total = ml_service.headline_total(predictions)
    response = {
        "predictions": predictions,          # per-model + "ensemble" + "stacking" (if available)
        "shap": shap_items,                  # [{feature, value, direction}]
        "recommendations": recommendations,  # ranked reduction tips
        "archetype": archetype,              # {cluster, label}
        "context": {
            "total_kg": round(total, 1),
            "classification": Config.classify(total),
            "percentile": Config.estimate_percentile(total),
            "world_avg_kg": Config.WORLD_AVG_KG,
            "india_avg_kg": Config.INDIA_AVG_KG,
            "us_avg_kg": Config.US_AVG_KG,
            "target_1_5c_kg": Config.TARGET_1_5C_KG,
        },
        "mock_mode": ml_service.MOCK_MODE,
        "stacking_available": ml_service.STACKING_AVAILABLE,
        "archetype_available": ml_service.ARCHETYPE_AVAILABLE,
    }
    return jsonify(response)
