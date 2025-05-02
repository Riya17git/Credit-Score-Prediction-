
# 📊 Smart Credit Score Advisor

This project uses machine learning to predict a user's **credit score for the next 3 and 6 months** based on their spending patterns, financial history, and other relevant features. It is designed to help users anticipate their financial future and make better decisions.

---

## 🔍 Problem Statement

Financial institutions and individuals benefit greatly from knowing how credit scores may change in the near future. This project aims to:
- Predict credit score 3 and 6 months into the future.
- Provide insights into the factors influencing changes in credit score.

---

## 🧠 Model Overview

The solution includes:
- **Data preprocessing & feature engineering** from financial transaction data.
- **Supervised machine learning models** (e.g., Random Forest, XGBoost, etc.)
- Forecasting using trained models saved as `.pkl` files.
- A simple **Flask-based app** (`app.py`) to interact with the model.

---
# 💰 Monthly Expense Forecasting with LSTM

This project forecasts **total monthly expenses** and **category-wise expenses** for individual clients using **LSTM-based time series forecasting**. It helps users and financial systems anticipate future spending habits for better planning and budgeting.

---

## 📌 Features

- Forecasts **total monthly expenses** for the next month.
- Forecasts **category-specific expenses** (e.g., groceries, restaurants).
- Trains lightweight models dynamically per category.
- Built with **PyTorch**, **Streamlit**, and **scikit-learn**.
- Real-time interactive dashboard via Streamlit.

---

## 🧠 Model Summary

Each client’s expense data is used to:
- Prepare time series inputs using sliding windows.
- Normalize data using a `MinMaxScaler`.
- Train an **LSTM neural network** per category (on-the-fly).
- Forecast the next value (next month’s expense).

---

##  Forecasted Categories

We focus on five major expense categories:
- 🏥 Drug Stores and Pharmacies  
- 🛒 Grocery Stores and Supermarkets  
- ⚡ Utilities (Electric, Gas, Water)  
- 🍽️ Restaurants  
- 🚕 Taxicabs and Limousines  

## Requirements 
- pandas
- numpy
- torch
- scikit-learn
- streamlit







