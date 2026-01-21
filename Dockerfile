# Use an official Python runtime as a parent image
# Using 3.9 to match development environment
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git: for some pip packages
# build-essential: for compiling c extensions if any
# curl: for healthchecks
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Healthcheck to ensure the service is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["streamlit", "run", "web_demo.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
