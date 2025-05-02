
import streamlit as st
import pandas as pd
import joblib
import numpy as np 

# Load trained models
model_3m = joblib.load("model_3m.pkl")
model_6m = joblib.load("model_6m.pkl")

st.title("🔮 Credit Score Forecasting")
st.write("Enter your financial details to predict credit score for the next 3 and 6 months.")

# Manual inputs from user
credit_limit = st.number_input("💳 Credit Limit", min_value=1000, max_value=100000, value=20000)
yearly_income = st.number_input("💰 Yearly Income", min_value=5000, max_value=200000, value=50000)
total_debt = st.number_input("📉 Total Debt", min_value=0, max_value=150000, value=10000)
num_credit_cards = st.number_input("📇 Number of Credit Cards", min_value=0, max_value=10, value=2)
current_age = st.number_input("🎂 Current Age", min_value=18, max_value=100, value=30)
birth_year = st.number_input("📅 Birth Year", min_value=1920, max_value=2025, value=1995)
birth_month = st.number_input("🗓️ Birth Month", min_value=1, max_value=12, value=5)

# Create a dataframe for prediction
input_data = pd.DataFrame({
    'credit_limit': [credit_limit],
    'yearly_income': [yearly_income],
    'total_debt': [total_debt],
    'num_credit_cards': [num_credit_cards],
    'current_age': [current_age],
    'birth_year': [birth_year],
    'birth_month': [birth_month] 
})

# Predict
if st.button("🔍 Predict Credit Score"):
    score_3m = model_3m.predict(input_data)[0]
    score_6m = model_6m.predict(input_data)[0]

    st.success(f"✅ Predicted Credit Score in 3 Months: {round(score_3m, 2)}")
    st.success(f"✅ Predicted Credit Score in 6 Months: {round(score_6m, 2)}")

# --- Expense Forecasting Section ---
st.header("📊 Monthly Expense Forecasting")
client_id = st.number_input("🧾 Enter Client ID", min_value=1, value=1001)

if st.button("📈 Forecast Monthly Expenses"):
    # Load the prepared data for the given client_id (you'd normally retrieve this from a database or pre-processed file)
    data = pd.read_csv("monthly_expenses.csv")  # You need to have this file with aggregated monthly data
    client_data = data[data['client_id'] == client_id].sort_values("month")

    if len(client_data) > 3:
        # General total monthly expense forecast
        expense_values = client_data['amount'].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(expense_values.reshape(-1, 1)).flatten()
        X = torch.tensor(scaled[-3:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        expense_last_model.eval()
        with torch.no_grad():
            pred_scaled = expense_last_model(X).item()
            predicted_expense = scaler.inverse_transform([[pred_scaled]])[0][0]

        current_expense = expense_values[-1]

        st.success(f"💸 Last Month's Actual Expense: ${current_expense:.2f}")
        st.success(f"📅 Predicted Expense for Next Month: ${predicted_expense:.2f}")

        # Category-wise predictions
        st.subheader("🔍 Predicted Spending by Category")
        target_categories = [
            "Drug Stores and Pharmacies",
            "Grocery Stores, Supermarkets",
            "Utilities - Electric, Gas, Water, Sanitary",
            "Eating Places and Restaurants",
            "Taxicabs and Limousines"
        ]

        for category in target_categories:
            if category in category_models:
                model, scaler = category_models[category]
                cat_exp = client_data[client_data['mcc_description'] == category]['amount'].values
                if len(cat_exp) >= 4:
                    scaled_cat = scaler.transform(cat_exp.reshape(-1, 1)).flatten()
                    X_cat = torch.tensor(scaled_cat[-3:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    model.eval()
                    with torch.no_grad():
                        pred_scaled = model(X_cat).item()
                        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
                        st.write(f"🔹 {category}: ${pred:.2f}")
                else:
                    st.info(f"Not enough data to forecast {category}")
    else:
        st.warning("Not enough historical data for this client.")
