# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy app code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r instructions.txt

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "fake_news_app/app.py"]
