# Use the official Python image
FROM python:3.11

# Install required system dependencies for OpenCV and OpenGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including CSV files)
COPY . .

# Expose the default Cloud Run port
EXPOSE 8000

# Run FastAPI using the PORT environment variable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
