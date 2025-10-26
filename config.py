# ========================== CONFIGURATION ==========================

CONFIG = {
    # --- Global Nutrient Targets (daily goals) ---
    "target_calories": 1690,
    "target_protein": 150,
    "target_carbs": 171,
    "target_fat": 45,

    # --- Flexibility Controls ---
    "macro_tolerance": 0.10,   # ±10% flexibility on each macro
    "calorie_tolerance": 150,  # ±150 kcal flexibility
    "candidate_cap": 12,       # number of top meals per type to consider
    "verbose_solver": False,   # True = show solver logs in console

    # --- Meal Structure ---
    "meal_types": ["breakfast", "lunch", "dinner", "snack"],
    "meal_counts": {
        "breakfast": 1,
        "lunch": 1,
        "dinner": 1,
        "snack": 2,   # two snacks allowed
    },

    # --- Required columns in meals.csv ---
    "required_columns": [
        "meal_name", "type", "protein", "carbs", "fat", "calories", "ingredients"
    ],
}
# ===================================================================
