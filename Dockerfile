# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (optional but prevents build errors)
RUN apt-get update && apt-get install -y gcc python3-dev build-essential && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container (not required for bot, but harmless)
EXPOSE 8080

# Define environment variable (optional)
ENV TELEGRAM_TOKEN=${TELEGRAM_TOKEN}

# Run telegram bot
CMD ["python", "telegram_bot.py"]
