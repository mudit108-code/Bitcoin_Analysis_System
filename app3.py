import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import numpy as np
import statsmodels.api as sm
from datetime import timedelta

st.title("Blockchain Price Data Analysis App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Data Info
    st.subheader("Dataset Information")
    st.write("Number of Rows and Columns:", data.shape)
    st.write("Column Names:", data.columns.tolist())

    # Feature 1: Display Summary Statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Feature 2: Visualize Price Over Time
    st.subheader("Price Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(data['Date']), data['Price'], label='Price', color='blue')
    ax.set_title("Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Feature 3: Histogram of Price
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['Price'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Feature 4: Filter Data by Date Range
    st.subheader("Filter Data by Date Range")
    start_date_input = st.date_input("Start Date:", value=pd.to_datetime(data['Date']).min())
    end_date_input = st.date_input("End Date:", value=pd.to_datetime(data['Date']).max())
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)

    if start_date <= end_date:
        filtered_data = data[(pd.to_datetime(data['Date']) >= start_date) & 
                             (pd.to_datetime(data['Date']) <= end_date)]
        st.write(f"Filtered Data ({start_date.date()} to {end_date.date()}):")
        st.dataframe(filtered_data)
    else:
        st.error("Start Date must be before End Date!")

    # Feature 5: Moving Average Visualization
    st.subheader("Moving Average")
    window = st.slider("Select Moving Average Window (days):", min_value=1, max_value=30, value=5)
    data['Moving Average'] = data['Price'].rolling(window=window).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(data['Date']), data['Price'], label='Price', color='blue')
    ax.plot(pd.to_datetime(data['Date']), data['Moving Average'], label=f'{window}-day Moving Average', color='orange')
    ax.set_title("Price with Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Feature 6: Download Filtered Data
    st.subheader("Download Filtered Data")
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv'
    )

    # Simple Price Prediction Model
    st.subheader("Simple Price Prediction Model")
    model_data = data[['Date', 'Price']].dropna()
    model_data['Date'] = pd.to_datetime(model_data['Date'])
    model_data['Date'] = model_data['Date'].map(lambda x: x.toordinal())

    X = model_data[['Date']]
    y = model_data['Price']

    model = LinearRegression()
    model.fit(X, y)

    future_date = pd.to_datetime("2025-01-01").toordinal()
    predicted_price = model.predict([[future_date]])
    st.write(f"Predicted Price on 2025-01-01: ${predicted_price[0]:.2f}")

    # Time Series Forecasting with SARIMAX
    st.subheader("Time Series Forecasting (SARIMAX)")
    ts_data = data[['Date', 'Price']].dropna()
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])
    ts_data.set_index('Date', inplace=True)

    # Fit SARIMAX model
    model_sarimax = sm.tsa.statespace.SARIMAX(ts_data['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results_sarimax = model_sarimax.fit()

    # Forecasting for the next 365 days
    forecast = results_sarimax.get_forecast(steps=365)
    forecast_index = [ts_data.index[-1] + timedelta(days=i) for i in range(1, 366)]
    forecast_values = forecast.predicted_mean

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts_data.index, ts_data['Price'], label='Historical Price', color='blue')
    ax.plot(forecast_index, forecast_values, label='Forecasted Price', color='orange')
    ax.set_title("Price Forecasting (SARIMAX)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Price Anomaly Detection (Z-score)
    st.subheader("Price Anomaly Detection")
    data['Z-score'] = zscore(data['Price'])
    anomaly_data = data[data['Z-score'].abs() > 3]
    st.write(f"Anomalous Prices (Z-score > 3):")
    st.dataframe(anomaly_data[['Date', 'Price', 'Z-score']])

    # Price Heatmap of per day and per hour separately
    st.subheader("Price Heatmap (Per Day & Per Hour)")
    data['Hour'] = pd.to_datetime(data['Date']).dt.hour
    hourly_avg = data.groupby(['Date', 'Hour'])['Price'].mean().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(hourly_avg, cmap="coolwarm", ax=ax)
    ax.set_title("Price Heatmap (Per Hour)")
    st.pyplot(fig)

    # Real-time Data Updates (Simulated)
    st.subheader("Real-time Data Updates (Simulated)")
    real_time_data = data.tail(10)
    st.write("Latest Data:")
    st.dataframe(real_time_data)

    

else:
    st.info("Please upload a dataset to proceed.")
