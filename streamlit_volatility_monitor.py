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

@st.cache_data(show_spinner=False)
def get_financial_data(symbol):
    """
    Ottiene dati finanziari completi per un singolo ticker
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Informazioni base
        info = ticker.info
        
        # Dati finanziari
        financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
        
        # Raccomandazioni analisti
        recommendations = ticker.recommendations
        
        # Target price e upgrade/downgrade
        upgrades_downgrades = ticker.upgrades_downgrades
        
        return {
            'info': info,
            'financials': financials,
            'quarterly_financials': quarterly_financials,
            'recommendations': recommendations,
            'upgrades_downgrades': upgrades_downgrades
        }
    except Exception as e:
        st.error(f"Errore nel recupero dati finanziari per {symbol}: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def get_earnings_data(symbol):
    """
    Ottiene dati su earnings per un ticker
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Earnings trimestrali
        quarterly_earnings = ticker.quarterly_earnings
        
        # Earnings annuali
        earnings = ticker.earnings
        
        # Calendar earnings (prossimi earnings)
        calendar = ticker.calendar
        
        return {
            'quarterly_earnings': quarterly_earnings,
            'earnings': earnings,
            'calendar': calendar
        }
    except Exception as e:
        st.error(f"Errore nel recupero dati earnings per {symbol}: {str(e)}")
        return None

def compute_volatility(df, window):
    log_returns = np.log(df["Close"]).diff()
    volatility = log_returns.rolling(window).std() * np.sqrt(252)
    return volatility.squeeze()  # Garantisce 1D anche se per errore fosse 2D

def get_percentile(val, series):
    return np.sum(series <= val) / len(series) * 100

def display_revenue_earnings(symbol):
    """
    Visualizza dati di revenue e earnings
    """
    earnings_data = get_earnings_data(symbol)
    financial_data = get_financial_data(symbol)
    
    if not earnings_data or not financial_data:
        st.warning("Dati finanziari non disponibili")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 Earnings Trimestrali**")
        if earnings_data['quarterly_earnings'] is not None and not earnings_data['quarterly_earnings'].empty:
            df_earnings = earnings_data['quarterly_earnings'].head(8)  # Ultimi 8 trimestri
            
            # Grafico earnings
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_earnings.index,
                y=df_earnings['Earnings'],
                name='Earnings',
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Earnings Trimestrali (EPS)",
                xaxis_title="Trimestre",
                yaxis_title="Earnings per Share ($)",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabella earnings
            earnings_display = df_earnings.copy()
            # Formatta le date - controllo se l'indice è datetime
            if hasattr(earnings_display.index, 'strftime'):
                earnings_display.index = earnings_display.index.strftime('%Y-%m-%d')
            else:
                earnings_display.index = earnings_display.index.astype(str)
            st.dataframe(earnings_display, use_container_width=True)
        else:
            st.info("Dati earnings trimestrali non disponibili")
    
    with col2:
        st.write("**💰 Revenue**")
        if financial_data['quarterly_financials'] is not None and not financial_data['quarterly_financials'].empty:
            # Estrai revenue dai dati finanziari
            financials = financial_data['quarterly_financials']
            if 'Total Revenue' in financials.index:
                revenue_data = financials.loc['Total Revenue'].dropna()
                
                # Converti in miliardi per visualizzazione migliore
                revenue_billions = revenue_data / 1e9
                
                # Grafico revenue
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=revenue_billions.index,
                    y=revenue_billions.values,
                    name='Revenue',
                    marker_color='#2ca02c'
                ))
                fig.update_layout(
                    title="Revenue Trimestrale",
                    xaxis_title="Trimestre",
                    yaxis_title="Revenue (Miliardi $)",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Converti in DataFrame per visualizzazione
                revenue_df = pd.DataFrame({
                    'Trimestre': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in revenue_data.index],
                    'Revenue (B$)': [f"{val/1e9:.2f}" for val in revenue_data.values]
                })
                st.dataframe(revenue_df, use_container_width=True)
            else:
                st.info("Dati revenue non disponibili")
        else:
            st.info("Dati finanziari trimestrali non disponibili")

def display_analyst_recommendations(symbol):
    """
    Visualizza raccomandazioni degli analisti
    """
    financial_data = get_financial_data(symbol)
    
    if not financial_data or financial_data['recommendations'] is None:
        st.warning("Raccomandazioni analisti non disponibili")
        return
    
    recommendations = financial_data['recommendations']
    
    if not recommendations.empty:
        # Ultimi 15 raccomandazioni
        recent_recommendations = recommendations.head(15)
        
        st.write("**🎯 Raccomandazioni Recenti**")
        
        # Prepara dati per visualizzazione
        display_recs = recent_recommendations.copy()
        if not display_recs.empty:
            # Formatta le date - controllo se l'indice è datetime
            if hasattr(display_recs.index, 'strftime'):
                display_recs.index = display_recs.index.strftime('%Y-%m-%d')
            else:
                # Se non è datetime, converte l'indice in stringa
                display_recs.index = display_recs.index.astype(str)
            
            # Seleziona colonne rilevanti
            cols_to_show = []
            if 'Firm' in display_recs.columns:
                cols_to_show.append('Firm')
            if 'To Grade' in display_recs.columns:
                cols_to_show.append('To Grade')
            if 'From Grade' in display_recs.columns:
                cols_to_show.append('From Grade')
            if 'Action' in display_recs.columns:
                cols_to_show.append('Action')
            
            if cols_to_show:
                st.dataframe(display_recs[cols_to_show], use_container_width=True)
        
        # Riassunto raccomandazioni attuali
        if 'To Grade' in recent_recommendations.columns:
            grade_counts = recent_recommendations['To Grade'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📊 Distribuzione Raccomandazioni**")
                for grade, count in grade_counts.head(5).items():
                    st.write(f"• {grade}: {count}")
            
            with col2:
                # Grafico a torta delle raccomandazioni
                if len(grade_counts) > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=grade_counts.index,
                        values=grade_counts.values,
                        hole=0.4
                    )])
                    fig.update_layout(
                        title="Distribuzione Raccomandazioni",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nessuna raccomandazione disponibile")

def display_target_price(symbol):
    """
    Visualizza target price e upgrade/downgrade
    """
    financial_data = get_financial_data(symbol)
    
    if not financial_data:
        st.warning("Dati non disponibili")
        return
    
    info = financial_data['info']
    upgrades_downgrades = financial_data['upgrades_downgrades']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎯 Target Price Consensus**")
        
        # Target price dai dati info
        target_high = info.get('targetHighPrice', 'N/A')
        target_low = info.get('targetLowPrice', 'N/A')
        target_mean = info.get('targetMeanPrice', 'N/A')
        target_median = info.get('targetMedianPrice', 'N/A')
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Target Alto", f"${target_high:.2f}" if target_high != 'N/A' else 'N/A')
            st.metric("Target Medio", f"${target_mean:.2f}" if target_mean != 'N/A' else 'N/A')
        
        with metrics_col2:
            st.metric("Target Basso", f"${target_low:.2f}" if target_low != 'N/A' else 'N/A')
            st.metric("Target Mediano", f"${target_median:.2f}" if target_median != 'N/A' else 'N/A')
        
        # Calcola upside/downside potenziale
        if target_mean != 'N/A' and current_price != 'N/A':
            upside = ((target_mean - current_price) / current_price) * 100
            color = "normal" if upside >= 0 else "inverse"
            st.metric("Upside Potenziale", f"{upside:.1f}%", 
                     delta=f"vs prezzo attuale ${current_price:.2f}")
    
    with col2:
        st.write("**📈 Upgrade/Downgrade Recenti**")
        
        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            recent_changes = upgrades_downgrades.head(10)
            
            # Prepara dati per visualizzazione
            display_changes = recent_changes.copy()
            # Formatta le date - controllo se l'indice è datetime
            if hasattr(display_changes.index, 'strftime'):
                display_changes.index = display_changes.index.strftime('%Y-%m-%d')
            else:
                # Se non è datetime, converte l'indice in stringa
                display_changes.index = display_changes.index.astype(str)
            
            # Seleziona colonne disponibili
            cols_to_show = []
            if 'Firm' in display_changes.columns:
                cols_to_show.append('Firm')
            if 'ToGrade' in display_changes.columns:
                cols_to_show.append('ToGrade')
            elif 'To Grade' in display_changes.columns:
                cols_to_show.append('To Grade')
            if 'FromGrade' in display_changes.columns:
                cols_to_show.append('FromGrade')
            elif 'From Grade' in display_changes.columns:
                cols_to_show.append('From Grade')
            if 'Action' in display_changes.columns:
                cols_to_show.append('Action')
            
            if cols_to_show:
                st.dataframe(display_changes[cols_to_show], use_container_width=True)
            else:
                st.dataframe(display_changes, use_container_width=True)
        else:
            st.info("Nessun upgrade/downgrade recente disponibile")

def display_financial_section(symbol):
    """
    Visualizza la sezione finanziaria per un singolo stock
    """
    st.subheader("📊 Analisi Finanziaria Avanzata")
    
    # Tab per organizzare i dati
    tab1, tab2, tab3 = st.tabs(["💰 Revenue & Earnings", "🎯 Raccomandazioni Analisti", "📈 Target Price"])
    
    with tab1:
        display_revenue_earnings(symbol)
    
    with tab2:
        display_analyst_recommendations(symbol)
    
    with tab3:
        display_target_price(symbol)

def create_optimized_removal_interface():
    """
    Interfaccia ottimizzata per la rimozione di stock dalla watchlist
    """
    if not st.session_state.tickers:
        return
    
    with st.sidebar.expander("🗑️ Gestione Avanzata Watchlist", expanded=False):
        st.write("**Rimozione Multipla**")
        
        # Checkbox per ogni ticker
        stocks_to_remove = []
        for ticker in st.session_state.tickers:
            if st.checkbox(f"❌ {ticker}", key=f"multi_remove_{ticker}"):
                stocks_to_remove.append(ticker)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Rimuovi", key="remove_selected"):
                if stocks_to_remove:
                    for stock in stocks_to_remove:
                        if stock in st.session_state.tickers:
                            st.session_state.tickers.remove(stock)
                    st.success(f"✅ Rimossi {len(stocks_to_remove)} titoli")
                    st.rerun()
                else:
                    st.warning("⚠️ Seleziona almeno un titolo")
        
        with col2:
            if st.button("🧹 Svuota", key="clear_all"):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.tickers = []
                    st.session_state.confirm_clear = False
                    st.success("✅ Watchlist svuotata!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Clicca di nuovo per confermare")
        
        st.markdown("---")
        st.write("**Rimozione Rapida**")
        quick_remove = st.selectbox(
            "Seleziona:", 
            options=["Seleziona ticker..."] + st.session_state.tickers,
            key="quick_remove"
        )
        
        if quick_remove != "Seleziona ticker..." and st.button("❌ Rimuovi", key="quick_remove_btn"):
            st.session_state.tickers.remove(quick_remove)
            st.success(f"✅ {quick_remove} rimosso")
            st.rerun()

# Sidebar inputs
st.sidebar.title("📌 Settings")
if "tickers" not in st.session_state:
    st.session_state.tickers = []

ticker_input = st.sidebar.text_input("Enter ticker (e.g. AAPL, UCG.MI)")
rolling_window = st.sidebar.number_input("Rolling window (days)", min_value=2, value=21, key="window")
history_period = st.sidebar.selectbox("Select historical period", ["6mo", "1y", "2y", "3y", "5y", "10y"], index=3, key="period")

if ticker_input and ticker_input.upper() not in st.session_state.tickers:
    if st.sidebar.button("➕ Add Ticker"):
        st.session_state.tickers.append(ticker_input.upper())
        st.rerun()

st.sidebar.markdown("### 📋 Selected Tickers")
for t in st.session_state.tickers:
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(f"• {t}")
    if col2.button("❌", key=f"remove_{t}"):
        st.session_state.tickers.remove(t)
        st.experimental_rerun()

# Interfaccia rimozione ottimizzata
create_optimized_removal_interface()

if not st.session_state.tickers:
    st.info("➕ Aggiungi un ticker per iniziare.")
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
        st.warning(f"⚠️ No valid data for {ticker}.")
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
        st.warning("⚠️ No valid tickers with data to display.")
    else:
        fig = go.Figure(chart_data)
        fig.update_layout(
            title="📈 Historical Volatility", 
            xaxis_title="Date", 
            yaxis_title="Volatility",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        summary_df = pd.DataFrame(summary_data).set_index("Ticker")
        if not summary_df.empty:
            highest_percentile_ticker = summary_df["Percentile"].idxmax()
            lowest_vol_ticker = summary_df["Current"].idxmin()
            highest_vol_ticker = summary_df["Current"].idxmax()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Highest Volatility", highest_vol_ticker, 
                         f"{float(summary_df.loc[highest_vol_ticker, 'Current']):.2%}")
            with col2:
                st.metric("📉 Lowest Volatility", lowest_vol_ticker, 
                         f"{float(summary_df.loc[lowest_vol_ticker, 'Current']):.2%}")
            with col3:
                st.metric("⚡ Highest Percentile", highest_percentile_ticker, 
                         f"{int(summary_df.loc[highest_percentile_ticker, 'Percentile'])}th")

            # Formattazione tabella
            summary_display = summary_df.copy()
            for col in ['Min', 'Max', 'Mean', 'Current']:
                summary_display[col] = summary_display[col].apply(lambda x: f"{x:.2%}")
            summary_display['Percentile'] = summary_display['Percentile'].apply(lambda x: f"{x:.0f}th")
            
            st.dataframe(summary_display, use_container_width=True)

        if len(correlation_df.columns) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = correlation_df.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r', 
                title="📊 Daily Returns Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# --- DETAIL TAB ---
with tab2:
    st.subheader("🔎 Single Stock Detail")
    selected_detail = st.selectbox("Select stock for detail view", st.session_state.tickers)
    
    if selected_detail:
        vol = vol_data.get(selected_detail)
        if vol is not None and not vol.dropna().empty:
            # Grafici volatilità esistenti
            col1, col2 = st.columns(2)
            
            with col1:
                fig_line = px.line(
                    x=vol.dropna().index, 
                    y=vol.dropna().values, 
                    labels={'x': 'Date', 'y': 'Volatility'}, 
                    title=f"📈 {selected_detail} Volatility Over Time"
                )
                st.plotly_chart(fig_line, use_container_width=True)
            
            with col2:
                fig_hist = px.histogram(
                    vol.dropna(), 
                    nbins=30, 
                    title=f"📊 {selected_detail} Volatility Distribution"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Box plot
            fig_box = go.Figure(data=[go.Box(y=vol.dropna().values, name=selected_detail)])
            fig_box.update_layout(title=f"📦 {selected_detail} Volatility Box Plot")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Link Yahoo Finance
            st.markdown(f"[📰 View latest news on Yahoo Finance for {selected_detail}](https://finance.yahoo.com/quote/{selected_detail}/news)")
            
            # --- NUOVA SEZIONE FINANZIARIA ---
            st.markdown("---")
            display_financial_section(selected_detail)
            
        else:
            st.warning(f"⚠️ No valid volatility data to display for {selected_detail}.")
