
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import torch.serialization


# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1])

def create_timeseries(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Load models
model_3m = joblib.load("model_3m.pkl")
model_6m = joblib.load("model_6m.pkl")
torch.serialization.add_safe_globals({'LSTMModel': LSTMModel})
expense_model = torch.load("expense_last_model.pkl", weights_only=False)
expense_model.eval()

# Load data
monthly_expenses = pd.read_csv("monthly_expenses.csv")

# Streamlit UI
st.title("💳 Financial Forecasting Dashboard")

st.header("📈 Credit Score Prediction")
credit_limit = st.number_input("💳 Credit Limit", min_value=1000, max_value=100000, value=20000)
yearly_income = st.number_input("💰 Yearly Income", min_value=5000, max_value=200000, value=50000)
total_debt = st.number_input("📉 Total Debt", min_value=0, max_value=150000, value=10000)
num_credit_cards = st.number_input("📇 Number of Credit Cards", min_value=0, max_value=10, value=2)
current_age = st.number_input("🎂 Current Age", min_value=18, max_value=100, value=30)
birth_year = st.number_input("📅 Birth Year", min_value=1920, max_value=2025, value=1995)
birth_month = st.number_input("🗓️ Birth Month", min_value=1, max_value=12, value=5)

input_data = pd.DataFrame({
    'credit_limit': [credit_limit],
    'yearly_income': [yearly_income],
    'total_debt': [total_debt],
    'num_credit_cards': [num_credit_cards],
    'current_age': [current_age],
    'birth_year': [birth_year],
    'birth_month': [birth_month]
})

if st.button("🔍 Predict Credit Score"):
    score_3m = model_3m.predict(input_data)[0]
    score_6m = model_6m.predict(input_data)[0]
    st.success(f"✅ Credit Score in 3 Months: {round(score_3m, 2)}")
    st.success(f"✅ Credit Score in 6 Months: {round(score_6m, 2)}")

st.divider()
st.header("💸 Expense Forecasting")

client_id = st.number_input("Enter Client ID", value=1759, step=1)
time_steps = 3
scaler = MinMaxScaler()

# Total Monthly Expense Forecast
client_data = monthly_expenses[monthly_expenses['client_id'] == client_id]
client_data['month_year'] = pd.to_datetime(client_data['month_year'], format='%Y-%m')
client_data['month'] = client_data['month_year'].dt.to_period('M').astype(str)

total_expenses = client_data.groupby('month')['amount'].sum().reset_index()
total_values = total_expenses['amount'].fillna(0).values

if len(total_values) > time_steps:
    scaled = scaler.fit_transform(total_values.reshape(-1, 1)).flatten()
    X, y = create_timeseries(scaled, time_steps)
    X_input = torch.tensor(X[-1:], dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        pred = expense_model(X_input).item()
        predicted_amount = scaler.inverse_transform([[pred]])[0][0]
        current_month_amount = total_values[-1]

    st.subheader("📊 Total Monthly Expenses")
    st.info(f"📅 Current Month's Total Expenses: ${round(current_month_amount, 2)}")
    st.success(f"📅 Predicted Next Month's Total Expenses: ${round(predicted_amount, 2)}")
else:
    st.warning("Not enough data for total expense forecasting.")

# Category-wise Expense Forecast
st.subheader("📦 Category-wise Expense Forecast (Selected Categories)")

target_categories = [
    "Drug Stores and Pharmacies", "Grocery Stores, Supermarkets",
    "Utilities - Electric, Gas, Water, Sanitary", "Eating Places and Restaurants",
    "Taxicabs and Limousines"
]

predictions = {}

for category in target_categories:
    cat_data = monthly_expenses[
        (monthly_expenses['client_id'] == client_id) &
        (monthly_expenses['mcc_description'] == category)
    ]['amount'].fillna(0).values

    if len(cat_data) > time_steps:
        scaled_data = scaler.fit_transform(cat_data.reshape(-1, 1)).flatten()
        X_cat, y_cat = create_timeseries(scaled_data, time_steps)
        X_train = torch.tensor(X_cat, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_cat, dtype=torch.float32)

        # Train small model per category
        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_function(output, y_train)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            next_val = model(X_train[-1:]).item()
            predicted_amount = scaler.inverse_transform([[next_val]])[0][0]
            current_amt = cat_data[-1]
            predictions[category] = (current_amt, predicted_amount)

# Show category results
for category, (curr, pred) in predictions.items():
    st.markdown(f"**{category}**")
    st.info(f"Current: ${round(curr, 2)}")
    st.success(f"Next Month (Predicted): ${round(pred, 2)}")
