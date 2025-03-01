import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_data():
    """Function to load and preprocess the dataset."""
    try:
        df = pd.read_csv("bitcoin_data.csv")  # Update with actual file path
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
    
    df.replace({"": np.nan}, inplace=True)  # Handle blank spaces
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill if needed
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['remaining_size'] = pd.to_numeric(df['remaining_size'], errors='coerce')
    df.dropna(inplace=True)
    return df

def display_summary(df):
    """Displays key statistics of the dataset."""
    st.subheader("Dataset Overview")
    st.write(df.head(10))
    st.write("Summary Statistics:")
    st.write(df.describe())

def price_trend(df):
    """Plots Bitcoin price trends over time."""
    st.subheader("Bitcoin Price Trends")
    fig = px.line(df, x='time', y='price', title='Bitcoin Price Over Time')
    st.plotly_chart(fig)

def trade_volume_analysis(df):
    """Analyzes buy vs sell trades."""
    st.subheader("Trade Volume Analysis")
    trade_counts = df['side'].value_counts()
    fig = px.bar(trade_counts, x=trade_counts.index, y=trade_counts.values, title='Trade Volume: Buy vs Sell')
    st.plotly_chart(fig)

def order_status_analysis(df):
    """Displays order status breakdown."""
    st.subheader("Order Status Breakdown")
    order_status_counts = df['reason'].value_counts()
    fig = px.pie(order_status_counts, names=order_status_counts.index, values=order_status_counts.values, title='Order Status Distribution')
    st.plotly_chart(fig)

def bid_price_distribution(df):
    """Displays a histogram of bid prices."""
    st.subheader("Bid Price Distribution")
    fig = px.histogram(df, x='bid', title='Bid Price Distribution', nbins=50)
    st.plotly_chart(fig)

def recommendation_system(df):
    """Provides buy, sell, or hold recommendations based on price trends."""
    st.subheader("Bitcoin Trading Recommendation")
    avg_price = df['price'].mean()
    last_price = df['price'].iloc[-1]
    
    if last_price < avg_price * 0.95:
        st.success("🔵 Recommendation: Consider Buying - Price is below average")
    elif last_price > avg_price * 1.05:
        st.warning("🔴 Recommendation: Consider Selling - Price is above average")
    else:
        st.info("🟡 Recommendation: Hold - Price is stable")

def main():
    st.title("Bitcoin Analysis & Recommendation System")
    df = load_data()
    if df is not None:
        display_summary(df)
        price_trend(df)
        trade_volume_analysis(df)
        order_status_analysis(df)
        bid_price_distribution(df)
        recommendation_system(df)

if __name__ == "__main__":
    main()
