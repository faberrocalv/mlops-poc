# PoC MLOps

This Proof of Concept (PoC) demonstrates how to use Jenkins and Docker to automate the training, testing, and deployment of a machine learning model.

## Prerequisites

Before you begin, make sure to place the provided Excel file `sentimientos.xlsx` inside the `raw_data` folder.

## Steps to Run the PoC

### 1. Clone the Repository

```bash
git clone https://github.com/faberrocalv/mlops-poc.git
```

### 2. Build Docker Images and Start Services

Use Docker Compose to build the necessary images and run the Jenkins and FastAPI service containers:

```bash
docker compose up -d
```

### 3. Access Jenkins and Retrieve Initial Password

To set up Jenkins, you need to access the Jenkins container and retrieve the initial admin password:

```bash
docker exec -it mlops-poc-jenkins-1 bash
```

### 4. Build the Training and Testing Environment

Build the Docker image that Jenkins will use to run the `train_test_model` container during pipeline execution:

```bash
docker build -t train-test-env -f docker/train_test_model/Dockerfile .
```
