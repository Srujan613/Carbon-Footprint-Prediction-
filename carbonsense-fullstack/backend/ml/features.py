"""
The 19 raw input fields, using the EXACT column names from the training
dataset (Carbon Emission.csv) and notebook — these are what
preprocessing.py::preprocess_new_input() expects as dict keys.

kind: "categorical" | "multiselect" | "numeric"
"""

FEATURE_SCHEMA = {
    "Body Type":                        ("categorical", ["obese", "overweight", "normal", "underweight"]),
    "Sex":                               ("categorical", ["male", "female"]),
    "Diet":                              ("categorical", ["omnivore", "pescatarian", "vegetarian", "vegan"]),
    "How Often Shower":                  ("categorical", ["daily", "twice a day", "more frequently", "less frequently"]),
    "Heating Energy Source":             ("categorical", ["coal", "natural gas", "wood", "electricity", "solar"]),
    "Transport":                         ("categorical", ["public", "private", "walk/bicycle"]),
    "Vehicle Type":                      ("categorical", ["none", "petrol", "diesel", "hybrid", "lpg", "electric"]),
    "Social Activity":                   ("categorical", ["never", "sometimes", "often"]),
    "Monthly Grocery Bill":              ("numeric", (0, 2000)),
    "Frequency of Traveling by Air":     ("categorical", ["never", "rarely", "frequently", "very frequently"]),
    "Vehicle Monthly Distance Km":       ("numeric", (0, 8000)),
    "Waste Bag Size":                    ("categorical", ["small", "medium", "large", "extra large"]),
    "Waste Bag Weekly Count":            ("numeric", (0, 20)),
    "How Long TV PC Daily Hour":         ("numeric", (0, 24)),
    "How Many New Clothes Monthly":      ("numeric", (0, 30)),
    "How Long Internet Daily Hour":      ("numeric", (0, 24)),
    "Energy efficiency":                 ("categorical", ["Yes", "Sometimes", "No"]),
    "Recycling":                         ("multiselect", ["Paper", "Plastic", "Glass", "Metal"]),
    "Cooking_With":                      ("multiselect", ["Stove", "Oven", "Microwave", "Grill", "Airfryer"]),
}

REQUIRED_FIELDS = list(FEATURE_SCHEMA.keys())


def validate_payload(payload: dict) -> list:
    """Returns a list of error strings; empty list means payload is valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"Missing field: {field}")
            continue
        kind, spec = FEATURE_SCHEMA[field]
        if kind == "categorical" and payload[field] not in spec:
            errors.append(f"Invalid value for {field}: {payload[field]!r} (expected one of {spec})")
        if kind == "multiselect":
            if not isinstance(payload[field], list):
                errors.append(f"Field {field} must be a list (got {type(payload[field]).__name__})")
            else:
                bad = [v for v in payload[field] if v not in spec]
                if bad:
                    errors.append(f"Invalid values for {field}: {bad} (expected subset of {spec})")
        if kind == "numeric":
            try:
                float(payload[field])
            except (TypeError, ValueError):
                errors.append(f"Invalid numeric value for {field}: {payload[field]!r}")
    return errors
