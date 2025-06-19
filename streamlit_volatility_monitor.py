# streamlit_volatility_monitor.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Volatility Monitor", layout="wide")

@st.cache_data(show_spinner=False)
def fetch_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        if df.empty:
            return None

        if 'Adj Close' in df.columns:
            df = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})
        elif 'Close' in df.columns:
            df = df[['Close']]
        else:
            return None

        df.dropna(inplace=True)
        if df.empty:
            return None

        return df
    except Exception:
        return None

def compute_volatility(df, window):
    log_returns = np.log(df["Close"]).diff()
    volatility = log_returns.rolling(window).std() * np.sqrt(252)
    return volatility.squeeze()  # Garantisce 1D anche se per errore fosse 2D


def get_percentile(val, series):
    return np.sum(series <= val) / len(series) * 100

# Sidebar inputs
st.sidebar.title("📌 Settings")
if "tickers" not in st.session_state:
    st.session_state.tickers = []

ticker_input = st.sidebar.text_input("Enter ticker (e.g. AAPL, UCG.MI)")
rolling_window = st.sidebar.number_input("Rolling window (days)", min_value=2, value=21, key="window")
history_period = st.sidebar.selectbox("Select historical period", ["6mo", "1y", "2y", "3y", "5y", "10y"], index=3, key="period")

if ticker_input and ticker_input.upper() not in st.session_state.tickers:
    if st.sidebar.button("Add Ticker"):
        st.session_state.tickers.append(ticker_input.upper())

st.sidebar.markdown("### Selected Tickers")
for t in st.session_state.tickers:
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(t)
    if col2.button("❌", key=f"remove_{t}"):
        st.session_state.tickers.remove(t)
        st.experimental_rerun()

if not st.session_state.tickers:
    st.info("Add a ticker to start.")
    st.stop()

# Tabs for navigation
tab1, tab2 = st.tabs(["📊 Overview", "🔎 Single Stock Detail"])

vol_data = {}
summary_data = []
correlation_df = pd.DataFrame()
chart_data = []

for ticker in st.session_state.tickers:
    df = fetch_data(ticker, st.session_state.period)
    if df is None or df.empty:
        st.warning(f"No valid data for {ticker}.")
        continue

    vol = compute_volatility(df, st.session_state.window)
    vol_data[ticker] = vol

    if not vol.dropna().empty:
        chart_data.append(go.Scatter(x=vol.dropna().index, y=vol.dropna().values, mode='lines', name=ticker))
        summary_data.append({
            "Ticker": ticker,
            "Min": float(vol.min()),
            "Max": float(vol.max()),
            "Mean": float(vol.mean()),
            "Current": float(vol.iloc[-1]),
            "Percentile": get_percentile(vol.iloc[-1], vol.dropna())
        })

        correlation_df[ticker] = np.log(df["Close"] / df["Close"].shift(1))

# --- OVERVIEW TAB ---
with tab1:
    st.subheader("📊 Volatility Overview")

    if not chart_data:
        st.warning("No valid tickers with data to display.")
    else:
        fig = go.Figure(chart_data)
        fig.update_layout(title="Historical Volatility", xaxis_title="Date", yaxis_title="Volatility")
        st.plotly_chart(fig, use_container_width=True)

        summary_df = pd.DataFrame(summary_data).set_index("Ticker")
        if not summary_df.empty:
            highest_percentile_ticker = summary_df["Percentile"].idxmax()
            lowest_vol_ticker = summary_df["Current"].idxmin()
            highest_vol_ticker = summary_df["Current"].idxmax()

            st.markdown(f"**📈 Highest Volatility:** `{highest_vol_ticker}` - {float(summary_df.loc[highest_vol_ticker, 'Current']):.2%}")
            st.markdown(f"**📉 Lowest Volatility:** `{lowest_vol_ticker}` - {float(summary_df.loc[lowest_vol_ticker, 'Current']):.2%}")
            st.markdown(f"**⚡ Highest Percentile:** `{highest_percentile_ticker}` - {int(summary_df.loc[highest_percentile_ticker, 'Percentile'])} Percentile")

            summary_df_style = summary_df.style.apply(lambda x: ["background-color: #ffb3b3" if x.name == highest_percentile_ticker else ""] * len(x), axis=1)
            st.dataframe(summary_df_style, use_container_width=True)

        if len(correlation_df.columns) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = correlation_df.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Daily Returns Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)

# --- DETAIL TAB ---
with tab2:
    st.subheader("🔎 Single Stock Detail")
    selected_detail = st.selectbox("Select stock for detail view", st.session_state.tickers)
    if selected_detail:
        vol = vol_data.get(selected_detail)
        if vol is not None and not vol.dropna().empty:
            st.plotly_chart(px.line(x=vol.dropna().index, y=vol.dropna().values, labels={'x': 'Date', 'y': 'Volatility'}, title=f"{selected_detail} Volatility Over Time"), use_container_width=True)
            st.plotly_chart(px.histogram(vol.dropna(), nbins=30, title=f"{selected_detail} Volatility Distribution"), use_container_width=True)
            st.plotly_chart(go.Figure(data=[go.Box(y=vol.dropna().values, name=selected_detail)]).update_layout(title="Box Plot"), use_container_width=True)
            st.markdown(f"[📰 View latest news on Yahoo Finance for {selected_detail}](https://finance.yahoo.com/quote/{selected_detail}/news)")
        else:
            st.warning(f"No valid volatility data to display for {selected_detail}.")
