# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /appAdd commentMore actions

# Copy application code to the container
COPY . /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r instructions.txt

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "fake_news_app/app.py"]
