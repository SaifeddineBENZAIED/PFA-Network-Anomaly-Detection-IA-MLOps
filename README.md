# PFA: Network Anomaly Detection with AI and MLOps

This repository contains the **Network Anomaly Detection** project, developed as part of my **Final Year Project (PFA)**. The project leverages **Artificial Intelligence (AI)** and **MLOps practices** to detect anomalies in network traffic. It includes a **web application** built with **Django** (backend) and **Angular** (frontend) to interact with the AI models, as well as a **CNN-based deep learning model** for anomaly detection. The project also implements a complete **MLOps pipeline** using **BentoML** for model lifecycle management.

---

## 🚀 Project Overview

The goal of this project is to:
1. **Detect anomalies in network traffic** using AI models (Machine Learning and Deep Learning).
2. **Compare different models** (e.g., traditional ML models vs. CNN) and select the best-performing one.
3. **Implement MLOps practices** to automate the model lifecycle, including:
   - Data preprocessing.
   - Model training and evaluation.
   - Model deployment using **FastAPI** and **BentoML**.
4. **Build a web application** to visualize results and interact with the deployed model.

---

## 🛠️ Technologies Used

### AI & Machine Learning
- **Python**: Primary programming language.
- **Deep Learning**: CNN (Convolutional Neural Network) for anomaly detection.
- **Machine Learning**: Comparison with traditional ML models.
- **BentoML**: Model deployment and lifecycle management.
- **FastAPI**: API for serving the model.

### Web Development
- **Django**: Backend framework for the web app.
- **Angular**: Frontend framework for the user interface.
- **RESTful APIs**: Communication between frontend and backend.

### MLOps Tools
- **BentoML**: Model packaging and deployment.
- **Docker**: Containerization of the application.
- **CI/CD**: Jenkins or GitHub Actions for automation.
- **Monitoring**: Prometheus and Grafana for model performance tracking.

### Data Processing
- **Pandas, NumPy**: Data manipulation and preprocessing.
- **Scikit-learn**: Traditional ML models and evaluation.
- **TensorFlow/Keras**: Deep learning model development.

---

## 📂 Repository Structure

PFA-Network-Anomaly-Detection-IA-MLOps/

├── ai-models/ # AI models (CNN, ML models)

│ ├── cnn_model/ # CNN model code and weights

│ ├── ml_models/ # Traditional ML models (e.g., Random Forest, SVM)

│ ├── data_preprocessing/ # Data preprocessing scripts

│ └── model_evaluation/ # Model evaluation and comparison scripts

├── web-app/ # Web application (Django + Angular)

│ ├── backend/ # Django backend code

│ ├── frontend/ # Angular frontend code

│ └── Dockerfile # Dockerfile for the web app

├── mlops-pipeline/ # MLOps pipeline scripts

│ ├── bentoml/ # BentoML model deployment

│ ├── fastapi/ # FastAPI for serving the model

│ └── monitoring/ # Prometheus and Grafana configurations

├── README.md # Project documentation

└── .gitignore # Git ignore file


---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Node.js (for Angular frontend)
- TensorFlow/Keras
- BentoML
- Django
- Angular CLI

### Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SaifeddineBENZAIED/PFA-Network-Anomaly-Detection-IA-MLOps.git
   cd PFA-Network-Anomaly-Detection-IA-MLOps
   ```
Set Up the AI Models:

Train the CNN model:

```bash
cd ai-models/cnn_model
python train_cnn.py
```

Compare ML models:

```bash
cd ai-models/ml_models
python compare_models.py
```

Deploy the Model with BentoML:

Package the model:

```bash
cd mlops-pipeline/bentoml
bentoml build
```

Serve the model using FastAPI:

```bash
bentoml serve
```

Run the Web Application:

Start the Django backend:

```bash
cd web-app/backend
python manage.py runserver
```

Start the Angular frontend:

```bash
cd web-app/frontend
ng serve
```

Monitor the Model:

Set up Prometheus and Grafana for monitoring:

```bash
cd mlops-pipeline/monitoring
docker-compose up -d
```

🔍 Key Features
AI Models

- CNN Model: Deep learning model for anomaly detection.

- Model Comparison: Comparison of CNN with traditional ML models (e.g., Random Forest, SVM).

- Model Evaluation: Metrics like accuracy, precision, recall, and F1-score.

Web Application

- User Interface: Built with Angular for visualizing results.

- Backend API: Built with Django for handling requests and interacting with the AI model.

MLOps Pipeline

- Data Preprocessing: Automated data cleaning and feature engineering.

- Model Training: Automated training and evaluation of models.

- Model Deployment: Deployment using BentoML and FastAPI.

- Monitoring: Real-time monitoring of model performance using Prometheus and Grafana.

📊 Results

- Best Model: CNN achieved the highest accuracy and F1-score for anomaly detection.

- MLOps Pipeline: Successfully automated the model lifecycle, from data preprocessing to deployment.

- Web App: Provides an intuitive interface for users to interact with the model and visualize results.

