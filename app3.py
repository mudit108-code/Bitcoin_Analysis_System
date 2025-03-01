import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

def load_data(file_path="bitcoin_data.csv"):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    df.replace({"": np.nan}, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['remaining_size'] = pd.to_numeric(df['remaining_size'], errors='coerce')
    df.dropna(inplace=True)

    # Remove outliers based on price (e.g., beyond 3 standard deviations)
    price_mean = df['price'].mean()
    price_std = df['price'].std()
    df = df[(df['price'] >= price_mean - 3 * price_std) & (df['price'] <= price_mean + 3 * price_std)]

    return df

def display_summary(df):
    """Displays dataset summary."""
    st.subheader("Dataset Summary")
    st.write(df.head(10))
    st.write("Summary Statistics:")
    st.write(df.describe())

def price_trend(df):
    """Plots Bitcoin price trends."""
    st.subheader("Bitcoin Price Trends")
    fig = px.line(df, x='time', y='price', title='Bitcoin Price Over Time')
    st.plotly_chart(fig)

def trade_volume_analysis(df):
    """Analyzes trade volume."""
    st.subheader("Trade Volume Analysis")
    trade_counts = df['side'].value_counts()
    fig = px.bar(trade_counts, x=trade_counts.index, y=trade_counts.values, title='Trade Volume: Buy vs Sell')
    st.plotly_chart(fig)

def order_status_analysis(df):
    """Analyzes order status."""
    st.subheader("Order Status Analysis")
    order_status_counts = df['reason'].value_counts()
    fig = px.pie(order_status_counts, names=order_status_counts.index, values=order_status_counts.values, title='Order Status Distribution')
    st.plotly_chart(fig)

def bid_ask_spread(df):
    """Plots bid-ask spread."""
    st.subheader("Bid-Ask Spread")
    df['spread'] = df['ask'] - df['bid']
    fig = px.line(df, x='time', y='spread', title='Bid-Ask Spread Over Time')
    st.plotly_chart(fig)

def price_volatility(df):
    """Calculates and displays price volatility."""
    st.subheader("Price Volatility")
    df['price_change'] = df['price'].diff()
    df['volatility'] = df['price_change'].rolling(window=30).std()  # 30-period rolling standard deviation
    fig = px.line(df, x='time', y='volatility', title='Price Volatility Over Time')
    st.plotly_chart(fig)

def recommendation_system(df):
    """Provides trading recommendations."""
    st.subheader("Bitcoin Trading Recommendation")
    avg_price = df['price'].mean()
    last_price = df['price'].iloc[-1]
    price_change = df['price'].diff().iloc[-1]
    volatility = df['price'].pct_change().rolling(window=30).std().iloc[-1] * 100 #volatility in percentage.
    
    st.write(f"Last Price: {last_price:.2f}, Average Price: {avg_price:.2f}")
    st.write(f"Recent Price Change: {price_change:.2f}")
    st.write(f"Recent Volatility: {volatility:.2f}%")

    if last_price < avg_price * 0.95 and price_change > 0 and volatility < 2:
        st.success("🔵 Strong Buy: Price is below average, recent increase, low volatility.")
    elif last_price < avg_price * 0.98 and price_change > 0 :
        st.success("🟢 Buy: Price is slightly below average, recent increase.")
    elif last_price > avg_price * 1.05 and price_change < 0 and volatility < 2:
        st.warning("🔴 Strong Sell: Price above average, recent decrease, low volatility.")
    elif last_price > avg_price * 1.02 and price_change < 0:
        st.warning("🟠 Sell: Price is slightly above average, recent decrease.")
    else:
        st.info("🟡 Hold: Price is relatively stable.")

def main():
    st.title("Bitcoin Analysis & Recommendation System")
    df = load_data()
    if df is not None:
        display_summary(df)
        price_trend(df)
        trade_volume_analysis(df)
        order_status_analysis(df)
        bid_ask_spread(df)
        price_volatility(df)
        recommendation_system(df)

if __name__ == "__main__":
    main()
