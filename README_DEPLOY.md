# Meal Planner Telegram Bot (Render)

## Quick Start
1. **Create Telegram bot** via @BotFather and copy the token.
2. Put these files in a GitHub repo:
   - `telegram_bot.py`
   - `meal_optimizer.py`
   - `requirements.txt`
   - `render.yaml`
   - `meals.csv`
   - `data.json` (empty `{}` to start)

3. **Connect repo to Render**
   - Create **New** → **Blueprint** → point to this repo
   - Render will detect `render.yaml` and create a **Worker**.
   - Set env var `TELEGRAM_TOKEN` to your BotFather token.
   - (Optional) Adjust `MEALS_CSV`, `DATA_JSON` envs if you rename paths.

4. **Deploy**
   - Render installs requirements and starts the worker.
   - Open Telegram and message your bot: `/start`

## Commands
- `/set_ingredients` — set weekly *unavailable* ingredients (comma-separated)
- `/plan_day` — choose any meals manually; bot optimizes the rest to hit macros
- `/cancel` — clears fixed choices for today

## CSV Schema
`meals.csv` needs this exact schema:
```
meal_name,type,protein,carbs,fat,calories,ingredients
Greek Yogurt Bowl,breakfast,20,25,5,230,"yogurt,berries,granola"
```
Types must be one of: `breakfast`, `lunch`, `dinner`, `snack`.

## Notes
- The optimizer uses PuLP (CBC solver bundled) with a fallback bounded search.
- Render free tier storage may reset on redeploy; consider persistent backing (S3/DB) for long-term history.
- Macro targets are currently fixed in code to 150P/171C/45F bands (140–160P, 160–180C, 40–50F). You can expose them via commands later.