# Use Python base image
FROM python:3.9-slimAdd commentMore actions

# Set the working directory
WORKDIR /app

# Copy application code to the container
COPY . /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Expose port
# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r instructions.txt

# Expose the application port
EXPOSE 5000
