import os


class Config:
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # Allow the frontend (served separately, e.g. via VS Code Live Server)
    # to call this API from a different origin during development.
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

    # Reference points used for the "Global Context" card and percentile framing.
    TARGET_1_5C_KG = 2000
    WORLD_AVG_KG = 4700
    INDIA_AVG_KG = 1900
    US_AVG_KG = 14000

    # Replace with a real lookup against your training data's distribution
    # once you have it (e.g. percentile of df["CarbonEmission"]).
    def estimate_percentile(total_kg: float) -> int:
        if total_kg <= 0:
            return 1
        # crude placeholder curve, centered near the world average
        pct = 50 + (total_kg - Config.WORLD_AVG_KG) / 120
        return int(max(1, min(99, round(pct))))

    def classify(total_kg: float) -> str:
        if total_kg < Config.TARGET_1_5C_KG:
            return "Well Below Average"
        if total_kg < Config.WORLD_AVG_KG:
            return "Below Average"
        if total_kg < Config.WORLD_AVG_KG * 1.5:
            return "Above Average"
        return "High Emissions"
