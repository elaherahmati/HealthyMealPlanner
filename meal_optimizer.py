import json
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


# ========================== CONFIGURATION ==========================
CONFIG = {
    # --- Global Nutrient Targets (daily goals) ---
    "target_calories": 1690,   # total daily calories
    "target_protein": 150,     # grams
    "target_carbs": 171,       # grams
    "target_fat": 45,          # grams

    # --- Number of meals allowed per type ---
    "meal_counts": {
        "breakfast": 1,
        "lunch": 1,
        "dinner": 1,
        "snack": 2,   # 👈 two snacks instead of one
    },
    # --- Flexibility Controls ---
    "macro_tolerance": 0.10,   # ±10% flexibility on each macro
    "calorie_tolerance": 150,  # ±150 kcal flexibility
    "candidate_cap": 12,       # number of top meals per type to consider
    "verbose_solver": False,   # True = show solver logs in console

    # --- Meal Structure ---
    "meal_types": ["breakfast", "lunch", "dinner", "snack"],

    # --- Column Names in meals.csv ---
    "required_columns": ["meal_name", "type", "protein", "carbs", "fat", "calories", "ingredients"],
}
# ====================================================================

MEAL_TYPES = CONFIG["meal_types"]

def normalize_token(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def parse_ingredients(cell: str) -> List[str]:
    if pd.isna(cell) or cell is None:
        return []
    return [normalize_token(tok) for tok in str(cell).split(",")]

def load_meals(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = CONFIG["required_columns"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df["type"] = df["type"].str.lower().str.strip()
    for c in ["protein", "carbs", "fat", "calories"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def filter_by_unavailable(meals_df: pd.DataFrame, unavailable: List[str]) -> pd.DataFrame:
    if not unavailable:
        return meals_df.copy()
    unavailable_norm = set(normalize_token(u) for u in unavailable if u.strip())
    mask = []
    for _, row in meals_df.iterrows():
        ingr = set(parse_ingredients(row.get("ingredients", "")))
        mask.append(len(ingr.intersection(unavailable_norm)) == 0)
    return meals_df[pd.Series(mask, index=meals_df.index)].reset_index(drop=True)

def macro_sum(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {"protein": 0.0, "carbs": 0.0, "fat": 0.0, "calories": 0.0}
    p = sum(float(r["protein"]) for r in rows)
    c = sum(float(r["carbs"]) for r in rows)
    f = sum(float(r["fat"]) for r in rows)
    k = sum(float(r["calories"]) for r in rows)
    return {"protein": p, "carbs": c, "fat": f, "calories": k}

def optimize_day_milp(meals_df: pd.DataFrame,
                      fixed: Dict[str, dict],
                      macro_ranges: Dict[str, Tuple[float, float]],
                      target_calories: float) -> Optional[Dict[str, dict]]:
    if not PULP_AVAILABLE:
        return optimize_day_fallback(meals_df, fixed, macro_ranges, target_calories)

    fixed_names = {v["meal_name"] for v in fixed.values()}
    pool = meals_df[~meals_df["meal_name"].isin(fixed_names)].copy()
    by_type = {t: pool[pool["type"] == t].reset_index(drop=True) for t in MEAL_TYPES}
    for t in MEAL_TYPES:
        if t not in fixed and by_type[t].empty:
            return None

    model = pulp.LpProblem("DailyMealPlanner", pulp.LpMinimize)
                          
    x_vars = {}
    for t in MEAL_TYPES:
        if t in fixed:
            continue
        subset = by_type[t]
        for i in range(len(subset)):
            x_vars[(t, i)] = pulp.LpVariable(f"x_{t}_{i}", lowBound=0, upBound=1, cat="Binary")

    for t in MEAL_TYPES:
        if t in fixed:
            continue
        subset = by_type[t]
        required_count = CONFIG["meal_counts"].get(t, 1)
        model += pulp.lpSum(x_vars[(t, i)] for i in range(len(subset))) == required_count

    fixed_rows = list(fixed.values())
    fixed_totals = macro_sum(fixed_rows)

   # --- Macro constraints with soft penalties ---
    # Add small "slack" variables to let the model deviate slightly if needed
    slacks = {}
    for macro in ["protein", "carbs", "fat"]:
        slacks[macro] = pulp.LpVariable(f"slack_{macro}", lowBound=0)

    for macro in ["protein", "carbs", "fat"]:
        base_lo, base_hi = macro_ranges[macro]
        tol = CONFIG["macro_tolerance"]
        lo = base_lo * (1 - tol)
        hi = base_hi * (1 + tol)

        rem_terms = []
        for t in MEAL_TYPES:
            if t in fixed:
                continue
            subset = by_type[t]
            vals = subset[macro].tolist()
            for i, val in enumerate(vals):
                rem_terms.append(val * x_vars[(t, i)])

        # Combine with already fixed meals
        if rem_terms:
            expr = pulp.lpSum(rem_terms)
            # Relax the lower/upper bounds by allowing slack usage
            model += expr + slacks[macro] >= (lo - fixed_totals[macro])
            model += expr - slacks[macro] <= (hi - fixed_totals[macro])
        else:
            # No variable meals of this type, check fixed totals only
            if not (lo <= fixed_totals[macro] <= hi):
                return None

    # Add a small penalty in the objective for any slack used
    # (Encourages staying close to macro targets but never infeasible)
    model += 0.01 * pulp.lpSum(slacks.values())
    if x_vars:
        kcal_terms = []
        for t in MEAL_TYPES:
            if t in fixed:
                continue
            subset = by_type[t]
            kcals = subset["calories"].tolist()
            for i, k in enumerate(kcals):
                kcal_terms.append(k * x_vars[(t, i)])
        total_kcal = pulp.lpSum(kcal_terms) + fixed_totals["calories"]
        z = pulp.LpVariable("abs_dev_kcal", lowBound=0)
        tolerance = CONFIG["calorie_tolerance"]
        model += (total_kcal - target_calories) <= z + tolerance
        model += (target_calories - total_kcal) <= z + tolerance
        model += z

    status = model.solve(pulp.PULP_CBC_CMD(msg=CONFIG["verbose_solver"]))
    if pulp.LpStatus[status] != "Optimal":
        return None

    selection = {**fixed}
    for t in MEAL_TYPES:
        if t in fixed:
            continue
        subset = by_type[t]
        pick = None
        for i in range(len(subset)):
            if pulp.value(x_vars[(t, i)]) > 0.5:
                pick = subset.iloc[i].to_dict()
                break
        if pick is None:
            return None
        selection[t] = pick
    return selection

def optimize_day_fallback(meals_df: pd.DataFrame,
                          fixed: Dict[str, dict],
                          macro_ranges: Dict[str, Tuple[float, float]],
                          target_calories: float) -> Optional[Dict[str, dict]]:
    import itertools
    CANDIDATE_CAP = CONFIG["candidate_cap"]                       
    fixed_names = {v["meal_name"] for v in fixed.values()}
    pool = meals_df[~meals_df["meal_name"].isin(fixed_names)].copy()
    by_type = {}
    for t in MEAL_TYPES:
        if t in fixed:
            continue
        subset = pool[pool["type"] == t].sort_values("protein", ascending=False)
        by_type[t] = subset.head(CANDIDATE_CAP).reset_index(drop=True)

    if len(fixed) == 4:
        totals = macro_sum(list(fixed.values()))
        ok = all(macro_ranges[m][0] <= totals[m] <= macro_ranges[m][1] for m in ["protein","carbs","fat"])
        return fixed if ok else None

    remain_types = [t for t in MEAL_TYPES if t not in fixed]
    lists = [by_type[t].to_dict("records") for t in remain_types]
    if any(len(lst) == 0 for lst in lists):
        return None

    best = None
    best_dev = float("inf")
    for combo in itertools.product(*lists):
        rows = list(combo) + list(fixed.values())
        s = macro_sum(rows)
        ok = True
        for m in ["protein", "carbs", "fat"]:
            lo, hi = macro_ranges[m]
            if not (lo <= s[m] <= hi):
                ok = False
                break
        if not ok:
            continue
        dev = abs(s["calories"] - target_calories)
        if dev < best_dev:
            best_dev = dev
            best = {**fixed}
            for t, row in zip(remain_types, combo):
                best[t] = row
    return best

def format_plan(selection: Dict[str, dict]) -> str:
    parts = ["✅ Optimized Daily Plan"]
    totals = macro_sum([selection[t] for t in MEAL_TYPES if t in selection])
    for t in MEAL_TYPES:
        if t in selection:
            if isinstance(selection[t], list):  # multiple meals of same type
                for idx, r in enumerate(selection[t], 1):
                    parts.append(f"{t.title()} {idx}: {r['meal_name']}  ({int(r['protein'])}P {int(r['carbs'])}C {int(r['fat'])}F, {int(r['calories'])} kcal)")
            else:
                r = selection[t]
                parts.append(f"{t.title():<9}: {r['meal_name']}  ({int(r['protein'])}P {int(r['carbs'])}C {int(r['fat'])}F, {int(r['calories'])} kcal)")
    return "\n".join(parts)
