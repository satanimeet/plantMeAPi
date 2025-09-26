FROM python:3.10-bullseye

WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create directory for downloaded models
RUN mkdir -p /app/api/model

# Set environment variable for model download
ENV MODEL_URL=""

# Render requires port 10000
EXPOSE 10000

# Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]