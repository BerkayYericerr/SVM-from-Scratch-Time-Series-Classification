# SVM with Kernel Methods for Time-Series Classification

This project implements a Support Vector Machine (SVM) from scratch and applies it to real-world time-series datasets. The goal is to analyze how different kernel functions capture temporal patterns and improve classification performance.

## 📌 Project Overview

In this project, an SVM classifier is implemented **without using built-in libraries (like sklearn SVC)**. The model is trained on two datasets:

- 📊 Electricity Consumption (AEP dataset)
- 🌫️ Air Pollution (PM2.5 - Seoul dataset)

The task is to predict whether the **next day exceeds a defined threshold** using a **7-day sliding window** of previous values.

---

## ⚙️ Features

- SVM implementation from scratch (dual optimization)
- Multiple kernel functions:
  - Linear Kernel
  - Polynomial Kernel
  - RBF (Gaussian) Kernel
  - Custom Time-Series Kernel (trend-aware)
- Time-series aware train-test split
- Time-based k-fold cross-validation
- Hyperparameter tuning
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Handling class imbalance

---

## 🧠 Methodology

### 🔹 Data Processing
- Hourly data converted to daily averages
- Sliding window (7 days → predict next day)
- Feature normalization applied

### 🔹 Classification Tasks
- **Energy Dataset:** Predict if next day's consumption > average
- **Air Pollution Dataset:** Predict if PM2.5 > 35 (unhealthy)

---

## 📊 Exploratory Data Analysis

### Air Pollution (PM2.5)
- Strong class imbalance:
  - Class 0: ~85.5%
  - Class 1: ~14.5%
- Contains spikes and irregular patterns over time

### Energy Consumption
- More balanced dataset:
  - Class 0: ~54.2%
  - Class 1: ~45.8%
- Shows seasonal and periodic behavior

---


---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy cvxopt

python main.py
