## # Mobile Price Classification – AWS SageMaker End-to-End ML Pipeline

This project implements a **complete machine learning workflow on AWS SageMaker**, from data ingestion and preprocessing to model training, deployment, and real-time inference. It demonstrates how to build scalable, production-grade ML systems using AWS cloud services.

---

## Project Overview

The goal of this project is to **predict the price range of mobile phones** using their hardware specifications such as battery power, RAM, display size, camera resolution, and memory.

The model is trained and deployed using **Amazon SageMaker** and integrates with:

- **Amazon S3** for data and artifact storage  
- **AWS IAM** for secure permissions  
- **Boto3** and **SageMaker SDK** for automation  
- **SageMaker endpoints** for real-time prediction  

---

## Architecture

The end-to-end pipeline includes:

1. **Amazon S3**
   - Stores input training data and output model artifacts.

2. **AWS IAM**
   - Provides execution roles for SageMaker to securely access S3 and other resources.

3. **Amazon SageMaker – Training**
   - Trains the model on a managed compute instance using a built-in algorithm (e.g., XGBoost).
   - Handles environment provisioning, logging, and artifact storage.

4. **Amazon SageMaker – Hosting (Endpoint)**
   - Deploys the trained model to a real-time HTTPS endpoint.
   - Serves predictions with low latency.

5. **Boto3 & SageMaker SDK**
   - Automates:
     - Data upload to S3
     - Training job creation
     - Model deployment
     - Inference calls

---

## 📂 Project Structure

```text
├── research.ipynb                         # Main Jupyter notebook (end-to-end pipeline)
├── mob_price_classification_train.csv     # Training dataset (sample)
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies
