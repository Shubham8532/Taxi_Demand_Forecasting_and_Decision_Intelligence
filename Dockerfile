# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first(faster builds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]