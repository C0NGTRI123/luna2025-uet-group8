FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the lung-nodule package and backend dependencies
COPY lung-nodule /app/lung-nodule
COPY requirements-backend.txt /app/

RUN pip install --no-cache-dir \
    -r /app/requirements-backend.txt \
    /app/lung-nodule

# Copy backend code
COPY backend/ /app/backend/

# Copy weight files (data, mounted at runtime in production)
COPY results/ /app/results/

WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
