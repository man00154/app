# app
# ITSM Incident Forecasting API

This project provides a RESTful API for forecasting ITSM (IT Service Management) incidents, detecting data drift in the incident data, and automating model retraining. It uses FastAPI for the API, Docker for containerization, Prometheus for monitoring, Grafana for visualization, and GitHub Actions for CI/CD.

## Project Structure


.
├── app/
│   ├── main.py             # FastAPI application with forecasting, drift detection, and retraining logic
│   ├── models/             # Directory to store trained models (created by the app)
│   └── ITSM_data.csv       # Your raw ITSM incident data
├── Dockerfile              # Dockerfile for building the application image
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Defines services for FastAPI app, Prometheus, and Grafana
├── prometheus.yml          # Prometheus configuration for scraping metrics
├── grafana/
│   └── provisioning/
│       └── dashboards/
│           └── dashboard.json # Grafana dashboard definition
│       └── datasources/
│           └── datasource.yml # Grafana datasource definition (Prometheus)
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # GitHub Actions CI/CD pipeline
└── README.md


## Features

* **Incident Forecasting:** Predicts future ITSM incident counts based on historical data.
* **Data Drift Detection:** Monitors incoming data for significant changes compared to the training data.
* **Automated Retraining:** Triggers model retraining when data drift is detected or manually via an API endpoint.
* **RESTful API:** Exposes functionalities via FastAPI.
* **Containerization:** Dockerized application for easy deployment.
* **Monitoring:** Integrated with Prometheus for collecting application metrics.
* **Visualization:** Grafana dashboards for visualizing application performance and model health.
* **CI/CD:** Automated build, push, and deployment using GitHub Actions.

## Setup and Local Development

### Prerequisites

* Docker and Docker Compose
* Python 3.8+
* Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/itsm-forecasting-api.git](https://github.com/your-username/itsm-forecasting-api.git)
    cd itsm-forecasting-api
    ```

2.  **Place your data:**
    Ensure your `ITSM_data.csv` file is placed in the `app/` directory.

3.  **Build and run with Docker Compose:**
    ```bash
    docker-compose build
    docker-compose up -d
    ```
    This will:
    * Build the Docker image for your FastAPI application.
    * Start the FastAPI app, Prometheus, and Grafana containers.
    * The FastAPI app will automatically start training models on startup.

4.  **Access the applications:**
    * **FastAPI Docs:** `http://localhost:8000/docs`
    * **Prometheus UI:** `http://localhost:9090`
    * **Grafana UI:** `http://localhost:3000` (Login with `admin`/`admin` - change in production!)

## API Endpoints

* **`GET /`**
    * **Description:** Root endpoint, returns a welcome message.
    * **Response:** `{"message": "ITSM Incident Forecasting API is running. Visit /docs for API documentation."}`

* **`POST /forecast`**
    * **Description:** Generates future incident forecasts for a specific category and granularity.
    * **Request Body (JSON):**
        ```json
        {
          "category": "Software",
          "granularity": "quarterly",
          "future_periods": 4
        }
        ```
    * **Response (JSON):**
        ```json
        {
          "category": "Software",
          "granularity": "quarterly",
          "forecast": [
            {"date": "2024-01-01", "predicted_value": 120.5},
            {"date": "2024-04-01", "predicted_value": 115.2},
            ...
          ]
        }
        ```

* **`POST /retrain`**
    * **Description:** Initiates a full model retraining pipeline in the background.
    * **Request Body (JSON - optional):**
        ```json
        {
          "file_path": "app/ITSM_data.csv" # Optional, defaults to this path
        }
        ```
    * **Response (JSON):**
        ```json
        {"message": "Model retraining initiated in the background. Check logs for progress."}
        ```

* **`POST /detect_drift`**
    * **Description:** Detects data drift by comparing new data against the historical snapshot used for training.
    * **Request Body (JSON):**
        ```json
        {
          "category": "Hardware",
          "granularity": "quarterly",
          "new_data": [
            {"ds": "2023-10-01", "y": 95},
            {"ds": "2024-01-01", "y": 110}
          ],
          "threshold_factor": 0.1 # Optional, default is 0.1 (10%)
        }
        ```
    * **Response (JSON):**
        ```json
        {"drift_detected": true, "message": "Data drift detected. Consider retraining models."}
        ```
        or
        ```json
        {"drift_detected": false, "message": "No significant data drift detected."}
        ```

## Monitoring with Prometheus and Grafana

* **Prometheus:** Scrapes metrics from the FastAPI application (exposed on port 8001). You can access its UI at `http://localhost:9090`.
* **Grafana:** Visualizes the metrics collected by Prometheus. Access its UI at `http://localhost:3000`. A basic dashboard is provisioned automatically. You can explore and create more detailed dashboards.

## CI/CD with GitHub Actions

The `.github/workflows/ci-cd.yml` file defines a GitHub Actions workflow that:

1.  **Builds and Pushes:** On every push to `main` or pull request to `main`, it builds a Docker image of your application and pushes it to Docker Hub.
2.  **Deploys:** After a successful build, it connects to your remote server via SSH (using provided GitHub Secrets) and pulls the latest Docker image, stops the old containers, and starts new ones.

### GitHub Secrets Configuration

For the CI/CD pipeline to work, you need to configure the following secrets in your GitHub repository (**Settings > Secrets and variables > Actions > New repository secret**):

* `DOCKER_USERNAME`: Your Docker Hub username.
* `DOCKER_PASSWORD`: Your Docker Hub access token (or password).
* `SSH_HOST`: The IP address or hostname of your deployment server.
* `SSH_USERNAME`: The SSH username for your deployment server.
* `SSH_PRIVATE_KEY`: The private SSH key used to authenticate with your deployment server. Ensure it's correctly formatted (e.g., multiline string).

## Retraining Strategy

* **Manual Retraining:** You can trigger retraining at any time by calling the `/retrain` API endpoint.
* **Data Drift Triggered Retraining:** The `/detect_drift` endpoint identifies data drift. While this example *reports* drift, in a production scenario, you might extend the logic to automatically trigger the `/retrain` endpoint if drift is detected. This would typically be done by a separate monitoring script or a scheduled job that periodically calls `/detect_drift` and then `/retrain` if necessary.

## Future Enhancements

* **More Sophisticated Drift Detection:** Implement statistical tests (e.g., KS-test, ADWIN) for more robust data drift detection.
* **Model Performance Monitoring:** Store and expose model performance metrics (e.g., RMSE, MAE) in Prometheus after each training run.
* **Alerting:** Configure Prometheus Alertmanager to send notifications (e.g., Slack, email) when drift is detected or model performance degrades.
* **A/B Testing/Canary Deployments:** For more advanced deployments, consider rolling updates or A/B testing new model versions.
* **Database Integration:** Instead of CSV, use a proper database (PostgreSQL, MongoDB) for storing ITSM data.
* **Authentication/Authorization:** Add security to your API endpoints.
* **Logging:** Implement more structured logging.

---
