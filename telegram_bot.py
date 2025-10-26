import os
import json
from typing import Dict, List

import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters

from meal_optimizer import (
    MEAL_TYPES, load_meals, filter_by_unavailable, optimize_day_milp, format_plan
)
from config import CONFIG

DATA_PATH = os.getenv("DATA_JSON", "data.json")
MEALS_CSV = os.getenv("MEALS_CSV", "meals.csv")

DEFAULT_MACRO_RANGES = {
    "protein": (CONFIG["target_protein"], CONFIG["target_protein"]),
    "carbs": (CONFIG["target_carbs"], CONFIG["target_carbs"]),
    "fat": (CONFIG["target_fat"], CONFIG["target_fat"]),
}
target_calories = CONFIG["target_calories"]
TARGET_CALORIES = 1690

def load_state() -> Dict:
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_state(state: Dict):
    with open(DATA_PATH, "w") as f:
        json.dump(state, f, indent=2)

def get_user(state: Dict, user_id: str) -> Dict:
    return state.setdefault(user_id, {"unavailable": [], "fixed": {}, "last_plan": {}})

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I’m your Meal Planner Bot 🍽️\n\n"
        "• /set_ingredients – tell me what you DON'T have this week\n"
        "• /plan_day – plan today (you can fix breakfast/lunch/etc., I'll optimize the rest)\n"
        "• /help – see commands\n\n"
        "Upload or edit meals.csv in your repo with nutrition & ingredients."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/set_ingredients – Set weekly unavailable ingredients\n"
        "/plan_day – Choose any meals manually, then I'll optimize the rest\n"
        "/cancel – Clear current selections"
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    u = get_user(state, str(update.effective_user.id))
    u["fixed"] = {}
    save_state(state)
    await update.message.reply_text("Cleared your fixed choices for today.")

# --- Set Ingredients ---
async def set_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "List ingredients you DON'T have this week (comma-separated), e.g.:\n"
        "`salmon, avocado, tofu`",
        parse_mode="Markdown"
    )
    return

async def set_ingredients_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    u = get_user(state, str(update.effective_user.id))
    raw = update.message.text or ""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    u["unavailable"] = items
    save_state(state)
    await update.message.reply_text(f"Got it. I will exclude: {', '.join(items) if items else '(none)'}")

# --- Plan Day Flow ---
def build_mealtype_toggle_keyboard(selected: List[str]):
    rows = []
    for t in MEAL_TYPES:
        label = f"{'✅' if t in selected else '⬜️'} {t.title()}"
        rows.append([InlineKeyboardButton(label, callback_data=f"toggle_{t}")])
    rows.append([InlineKeyboardButton("Done ✅", callback_data="toggle_done")])
    return InlineKeyboardMarkup(rows)

async def plan_day(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    u = get_user(state, str(update.effective_user.id))
    u["fixed"] = {}
    save_state(state)
    context.user_data["selected_types"] = []
    await update.message.reply_text("Which meals do you want to choose manually? You can pick multiple, then press Done.",
                                    reply_markup=build_mealtype_toggle_keyboard([]))

async def toggle_mealtype(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "toggle_done":
        selected = context.user_data.get("selected_types", [])
        if not selected:
            await query.edit_message_text("No manual picks. I'll choose everything optimally. Running…")
            await run_optimizer(update, context)
            return
        context.user_data["pick_queue"] = selected[:]
        await query.edit_message_text("Great! Let's pick your meals.")
        await prompt_next_pick(update, context)
        return

    _, t = data.split("_", 1)
    selected = context.user_data.get("selected_types", [])
    if t in selected:
        selected.remove(t)
    else:
        selected.append(t)
    context.user_data["selected_types"] = selected
    await query.edit_message_reply_markup(reply_markup=build_mealtype_toggle_keyboard(selected))

async def prompt_next_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    user_id = str(update.effective_user.id)
    u = get_user(state, user_id)

    if not context.user_data.get("pick_queue"):
        save_state(state)
        await run_optimizer(update, context)
        return

    t = context.user_data["pick_queue"][0]
    # Load meals and filter
    df = load_meals(MEALS_CSV)
    df = filter_by_unavailable(df, u.get("unavailable", []))
    subset = df[df["type"] == t].head(12).reset_index(drop=True)

    if subset.empty:
        # skip this type
        context.user_data["pick_queue"].pop(0)
        await update.effective_chat.send_message(f"No available {t} options (after filtering). Skipping.")
        await prompt_next_pick(update, context)
        return

    # Build keyboard with first 6 options
    buttons = []
    for i, row in subset.iterrows():
        label = f"{row['meal_name']} ({int(row['protein'])}P {int(row['carbs'])}C {int(row['fat'])}F)"
        buttons.append([InlineKeyboardButton(label, callback_data=f"pick_{t}_{i}")])
    # Trim to max 8 rows for Telegram message size safety
    buttons = buttons[:8]
    await update.effective_chat.send_message(
        f"Choose your *{t}*:", parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons)
    )
    # Store subset in user_data for later retrieval
    context.user_data[f"options_{t}"] = subset.to_dict("records")

async def handle_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, t, idx_str = query.data.split("_", 2)
    idx = int(idx_str)

    options = context.user_data.get(f"options_{t}", [])
    if idx < 0 or idx >= len(options):
        await query.edit_message_text("Invalid selection. Try again with /plan_day")
        return

    chosen = options[idx]
    state = load_state()
    user_id = str(update.effective_user.id)
    u = get_user(state, user_id)
    fixed = u.get("fixed", {})
    fixed[t] = chosen
    u["fixed"] = fixed
    save_state(state)

    await query.edit_message_text(f"Fixed {t}: {chosen['meal_name']}")
    # pop from queue and continue
    if context.user_data.get("pick_queue"):
        context.user_data["pick_queue"].pop(0)
    await prompt_next_pick(update, context)

async def run_optimizer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    user_id = str(update.effective_user.id)
    u = get_user(state, user_id)

    try:
        df = load_meals(MEALS_CSV)
    except Exception as e:
        await update.effective_chat.send_message(f"Failed to read meals.csv: {e}")
        return

    df = filter_by_unavailable(df, u.get("unavailable", []))
    fixed = u.get("fixed", {})
    selection = optimize_day_milp(
        df,
        fixed=fixed,
        macro_ranges=DEFAULT_MACRO_RANGES,
        target_calories=TARGET_CALORIES
    )
    if selection is None:
        await update.effective_chat.send_message(
            "❌ No feasible plan. Try un-fixing some meals, widening macro ranges, or adding more meals."
        )
        return

    # Save and send
    u["last_plan"] = selection
    save_state(state)
    await update.effective_chat.send_message(format_plan(selection))

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN environment variable")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("cancel", cancel))

    app.add_handler(CommandHandler("set_ingredients", set_ingredients))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, set_ingredients_text))

    app.add_handler(CommandHandler("plan_day", plan_day))
    app.add_handler(CallbackQueryHandler(toggle_mealtype, pattern="^toggle_"))
    app.add_handler(CallbackQueryHandler(handle_pick, pattern="^pick_"))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
