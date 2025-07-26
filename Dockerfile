# Use an official Python runtime as a parent image
# Keep this at 3.11 as previously fixed <-- MOVED COMMENT TO ITS OWN LINE
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# --- START: Added/Modified lines for robustness and debugging ---
# Upgrade pip to the latest version to avoid potential issues with outdated pip
RUN pip install --no-cache-dir --upgrade pip

# (Optional) Debugging step: print Python version to confirm
RUN python --version
# --- END: Added/Modified lines ---

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app

# Create the models directory
RUN mkdir -p /app/app/models

# Expose the port the app runs on (FastAPI) and the metrics port (Prometheus)
EXPOSE 8000
EXPOSE 8001

# Command to run the application using Uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container
# The --port 8000 specifies the port for the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
