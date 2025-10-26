# --- Base image: lightweight official Python ---
FROM python:3.11-slim

# --- Set working directory ---
WORKDIR /app

# --- Copy project files into container ---
COPY . /app

# --- Install dependencies ---
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# --- Set environment variable (optional) ---
ENV TELEGRAM_TOKEN=${TELEGRAM_TOKEN}

# --- Run your bot ---
CMD ["python", "telegram_bot.py"]
