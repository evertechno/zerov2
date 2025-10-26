import streamlit as st
import pandas as pd
import json
import re
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import ta  # Technical Analysis library
import base64
import fitz # PyMuPDF for reading PDFs

# --- AI Imports ---
try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    st.error("Google Generative AI library not found. Please install it using `pip install google-generativeai`.")
    st.stop()
    
# --- KiteConnect Imports ---
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Please install it using `pip install kiteconnect`.")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Portfolio Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, validating investment compliance, and AI-powered analysis.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# Initialize session state variables
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state: st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state: st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state: st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state: st.session_state["last_fetched_symbol"] = None
if "current_market_data" not in st.session_state: st.session_state["current_market_data"] = None
if "holdings_data" not in st.session_state: st.session_state["holdings_data"] = None
if "compliance_results_df" not in st.session_state: st.session_state["compliance_results_df"] = pd.DataFrame()
if "advanced_metrics" not in st.session_state: st.session_state["advanced_metrics"] = None
if "ai_analysis_response" not in st.session_state: st.session_state["ai_analysis_response"] = None


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    
    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not gemini_conf.get("api_key"):
        errors.append("Google Gemini API key")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Ensure your `secrets.toml` includes both [kite] and [google_gemini] sections.")
        st.stop()
    return kite_conf, gemini_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS = load_secrets()
genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])


# --- KiteConnect Client Initialization ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()

# --- Utility Functions (These functions remain unchanged from the previous version) ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns: df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns: df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e: return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return {"_error": "Kite not authenticated."}
    try: return kite_instance.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
    except Exception as e: return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    instruments_df = load_instruments_cached(api_key, access_token)
    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        token = find_instrument_token(instruments_secondary, symbol, index_exchange)
    if not token: return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    try:
        data = kite_instance.historical_data(token, from_date=datetime.combine(from_date, datetime.min.time()), to_date=datetime.combine(to_date, datetime.max.time()), interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]); df.set_index("date", inplace=True); df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce'); df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e: return pd.DataFrame({"_error": [str(e)]})

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty: return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None

def add_technical_indicators(df: pd.DataFrame, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns: return df.copy()
    df_copy = df.copy()
    for period in sma_periods:
        if period > 0: df_copy[f'SMA_{period}'] = ta.trend.sma_indicator(df_copy['close'], window=period)
    for period in ema_periods:
        if period > 0: df_copy[f'EMA_{period}'] = ta.trend.ema_indicator(df_copy['close'], window=period)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'], df_copy['MACD_signal'], df_copy['MACD_hist'] = macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'], df_copy['Bollinger_Low'], df_copy['Bollinger_Mid'], df_copy['Bollinger_Width'] = bollinger.bollinger_hband(), bollinger.bollinger_lband(), bollinger.bollinger_mavg(), bollinger.bollinger_wband()
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    df_copy.fillna(method='bfill', inplace=True); df_copy.fillna(method='ffill', inplace=True)
    return df_copy.dropna()

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2: return {}
    daily_returns_decimal = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty: return {}
    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0
    annualized_return = ((1 + daily_returns_decimal.mean()) ** TRADING_DAYS_PER_YEAR - 1) * 100
    annualized_volatility = daily_returns_decimal.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    risk_free_rate_decimal = risk_free_rate / 100.0
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else np.nan
    if not cumulative_returns.empty: max_drawdown = (((1 + cumulative_returns).cummax() - (1 + cumulative_returns)) / (1 + cumulative_returns).cummax()).max() * 100
    else: max_drawdown = np.nan
    def round_if_float(x): return round(x, 4) if isinstance(x, (int, float)) and not np.isnan(x) else np.nan
    return {"Total Return (%)": round_if_float(total_return), "Annualized Return (%)": round_if_float(annualized_return), "Annualized Volatility (%)": round_if_float(annualized_volatility), "Sharpe Ratio": round_if_float(sharpe_ratio), "Max Drawdown (%)": round_if_float(max_drawdown)}


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    if not st.session_state["kite_access_token"]:
        st.markdown(f"Click the link, authorize, and you'll be redirected back.")
        st.link_button("🔗 Open Kite login", login_url, use_container_width=True)
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token"); st.session_state["kite_login_response"] = data
                st.sidebar.success("Kite authentication successful."); st.query_params.clear(); st.rerun()
            except Exception as e: st.sidebar.error(f"Authentication failed: {e}")
    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", use_container_width=True):
            st.session_state.clear(); st.success("Logged out from Kite."); st.rerun()
    else: st.info("Not authenticated with Kite yet.")
    st.markdown("---")
    st.markdown("### 2. Quick Data Access")
    if st.session_state["kite_access_token"]:
        if st.button("Fetch Current Holdings", key="sidebar_fetch_holdings_btn", use_container_width=True):
            current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
            try:
                holdings = current_k_client_for_sidebar.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings); st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e: st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
                st.download_button("Download Holdings (CSV)", st.session_state["holdings_data"].to_csv(index=False).encode('utf-8'), "kite_holdings.csv", "text/csv", key="download_holdings_sidebar_csv", use_container_width=True)
    else: st.info("Login to Kite to access quick data.")

# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main UI - Tabs for modules ---
tabs = st.tabs(["📈 Market & Historical", "💼 Investment Compliance", "🤖 AI-Powered Analysis"])
tab_market, tab_compliance, tab_ai = tabs

# --- Tab Logic Functions ---

def render_market_historical_tab(kite_client, api_key, access_token):
    st.header("📈 Market Data & Historical Candles with TA")
    if not kite_client: st.info("Login first to fetch market data."); return
    st.subheader("Current Market Data Snapshot")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="RELIANCE", key="market_symbol_tab")
        if st.button("Get Market Data", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange)
            if ltp_data and "_error" not in ltp_data: st.session_state["current_market_data"] = ltp_data
            else: st.error(f"Failed: {ltp_data.get('_error', 'Unknown error')}")
    with col_market_quote2:
        if st.session_state.get("current_market_data"): st.json(st.session_state["current_market_data"])
        else: st.info("Market data will appear here.")
    st.markdown("---")
    st.subheader("Historical Price Data & Technical Analysis")
    hist_symbol = st.text_input("Tradingsymbol", value="INFY", key="hist_sym_tab_input_ta")
    col_fetch, col_interval, col_dates = st.columns(3)
    with col_dates:
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=365), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
    with col_interval:
        interval = st.selectbox("Interval", ["day", "week", "minute", "5minute", "30minute", "month"], index=0, key="hist_interval_selector_ta")
    with col_fetch:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch & Prepare Data", key="fetch_historical_data_ta_btn", type="primary"):
            with st.spinner(f"Fetching data..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, DEFAULT_EXCHANGE)
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist; st.session_state["last_fetched_symbol"] = hist_symbol
                else: st.error(f"Fetch failed: {df_hist.get('_error', 'Unknown error')}")
    if not st.session_state.get("historical_data", pd.DataFrame()).empty:
        df = st.session_state["historical_data"]
        with st.expander("Technical Indicator & Plotting Options"):
            ta_c1, ta_c2, ta_c3 = st.columns(3)
            with ta_c1: sma_periods_str = st.text_input("SMA Periods", "20,50"); ema_periods_str = st.text_input("EMA Periods", "12,26"); rsi_window = st.number_input("RSI Window", 5, 50, 14)
            with ta_c2: macd_fast=st.number_input("MACD Fast",5,50,12); macd_slow=st.number_input("MACD Slow",10,100,26); macd_signal=st.number_input("MACD Signal",5,50,9)
            with ta_c3: bb_window=st.number_input("Bollinger Window",5,50,20); bb_std_dev=st.number_input("Bollinger Std Dev",1.0,4.0,2.0,0.5); chart_type=st.selectbox("Chart Style", ["Candlestick", "Line"]); indicators_to_plot=st.multiselect("Plot on Price Chart",["SMA", "EMA", "Bollinger Bands"])
            sma_periods = [int(p.strip()) for p in sma_periods_str.split(',') if p.strip().isdigit()]
            ema_periods = [int(p.strip()) for p in ema_periods_str.split(',') if p.strip().isdigit()]
            df_with_ta = add_technical_indicators(df, sma_periods, ema_periods, rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev)
        st.subheader(f"Technical Analysis for {st.session_state['last_fetched_symbol']} ({interval})")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.2, 0.2])
        if chart_type == "Candlestick": fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        else: fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
        if "SMA" in indicators_to_plot:
            for p in sma_periods: fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'SMA_{p}'), mode='lines', name=f'SMA {p}'), row=1, col=1)
        if "EMA" in indicators_to_plot:
            for p in ema_periods: fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta.get(f'EMA_{p}'), mode='lines', name=f'EMA {p}'), row=1, col=1)
        if "Bollinger Bands" in indicators_to_plot:
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_High'], mode='lines', line=dict(width=0.5, color='gray'), name='BB High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['Bollinger_Low'], mode='lines', line=dict(width=0.5, color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='BB Low'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['RSI'], mode='lines', name='RSI'), row=3, col=1); fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5); fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
        fig.add_trace(go.Bar(x=df_with_ta.index, y=df_with_ta['MACD_hist'], name='MACD Hist', marker_color='orange'), row=4, col=1); fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=4, col=1); fig.add_trace(go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='red')), row=4, col=1)
        fig.update_layout(height=1000, xaxis_rangeslider_visible=False, title_text=f"{st.session_state['last_fetched_symbol']} Technical Analysis", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Performance Snapshot")
        metrics = calculate_performance_metrics(df['close'].pct_change(), risk_free_rate=6.0)
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).style.format("{:.4f}"))

def parse_and_validate_rules(rules_text: str, portfolio_df: pd.DataFrame):
    results = []
    if not rules_text.strip() or portfolio_df.empty: return results
    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum(); stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    rating_weights = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series()
    asset_class_weights = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series()
    def check_pass(actual, op, threshold):
        if op == '>': return actual > threshold;
        if op == '<': return actual < threshold;
        if op == '>=': return actual >= threshold;
        if op == '<=': return actual <= threshold;
        if op == '=': return actual == threshold;
        return False
    for rule in rules_text.strip().split('\n'):
        rule = rule.strip();
        if not rule or rule.startswith('#'): continue
        parts = re.split(r'\s+', rule); rule_type = parts[0].upper()
        try:
            actual_value = None; details = ""
            if len(parts) < 3: results.append({'rule': rule, 'status': 'Error', 'details': 'Invalid format.'}); continue
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']: results.append({'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'."}); continue
            threshold = float(parts[-1].replace('%', ''))
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper();
                if symbol in stock_weights.index: actual_value = stock_weights.get(symbol, 0.0); details = f"Actual for {symbol}: {actual_value:.2f}%"
                else: results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found."}); continue
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper();
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector: actual_value = sector_weights.get(matching_sector, 0.0); details = f"Actual for {matching_sector}: {actual_value:.2f}%"
                else: results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sector_name}' not found."}); continue
            elif rule_type in ['RATING', 'ASSET_CLASS']:
                name = ' '.join(parts[1:-2]).upper();
                weights_series = rating_weights if rule_type == 'RATING' else asset_class_weights
                actual_value = weights_series.get(name, 0.0); details = f"Actual for {name}: {actual_value:.2f}%"
            elif rule_type in ['TOP_N_STOCKS', 'TOP_N_SECTORS'] and len(parts) == 4:
                n = int(parts[1]);
                if rule_type == 'TOP_N_STOCKS': actual_value = portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum(); details = f"Actual weight of top {n} stocks: {actual_value:.2f}%"
                else: actual_value = sector_weights.nlargest(n).sum(); details = f"Actual weight of top {n} sectors: {actual_value:.2f}%"
            elif rule_type == 'COUNT_STOCKS' and len(parts) == 3:
                actual_value = len(portfolio_df); details = f"Actual count: {actual_value}"
            else: results.append({'rule': rule, 'status': 'Error', 'details': 'Unrecognized rule format.'}); continue
            if actual_value is not None:
                passed = check_pass(actual_value, op, threshold); status = "✅ PASS" if passed else "❌ FAIL"
                results.append({'rule': rule, 'status': status, 'details': f"{details} | Rule: {op} {threshold}"})
        except (ValueError, IndexError) as e: results.append({'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}"})
    return results

def calculate_advanced_metrics(portfolio_df, api_key, access_token):
    symbols = portfolio_df['Symbol'].tolist()
    weights = (portfolio_df['Real-time Value (Rs)'] / portfolio_df['Real-time Value (Rs)'].sum()).values
    from_date = datetime.now().date() - timedelta(days=366); to_date = datetime.now().date()
    returns_df = pd.DataFrame(); failed_symbols = []
    progress_bar = st.progress(0, "Fetching historical data for metrics...")
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and '_error' not in hist_data.columns: returns_df[symbol] = hist_data['close'].pct_change()
        else: failed_symbols.append(symbol)
        progress_bar.progress((i + 1) / len(symbols), f"Fetching data for {symbol}...")
    if failed_symbols: st.warning(f"Could not fetch historical data for: {', '.join(failed_symbols)}. They will be excluded.")
    returns_df.dropna(how='all', inplace=True); returns_df.fillna(0, inplace=True)
    if returns_df.empty: st.error("Not enough historical data to calculate advanced metrics."); return None
    portfolio_returns = returns_df.dot(weights); var_95 = portfolio_returns.quantile(0.05)
    benchmark_data = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    if benchmark_data.empty or '_error' in benchmark_data.columns: st.error(f"Could not fetch benchmark data. Beta cannot be calculated."); portfolio_beta = None
    else:
        benchmark_returns = benchmark_data['close'].pct_change()
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna(); aligned_returns.columns = ['portfolio', 'benchmark']
        covariance = aligned_returns.cov().iloc[0, 1]; benchmark_variance = aligned_returns['benchmark'].var()
        portfolio_beta = covariance / benchmark_variance if benchmark_variance > 0 else None
    progress_bar.empty()
    return {"var_95": var_95, "beta": portfolio_beta}

def render_investment_compliance_tab(kite_client, api_key, access_token):
    st.header("💼 Investment Compliance & Portfolio Analysis")
    st.markdown("Upload a portfolio, define compliance rules, and get a real-time, multi-dimensional analysis.")
    if not kite_client: st.info("Please login to Kite Connect to fetch live prices."); return
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("1. Upload Portfolio"); uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Required: 'Symbol', 'Industry', 'Quantity', 'Market/Fair Value(Rs. in Lacs)'.")
    with col2:
        st.subheader("2. Define Compliance Rules"); rules_text = st.text_area("Enter one rule per line.", height=150, key="compliance_rules_input", help="e.g., STOCK RELIANCE < 10%")
        with st.expander("📖 View Rule Syntax Guide"): st.markdown("- **STOCK**: `STOCK [Symbol] <operator> [Value]%`\n- **SECTOR**: `SECTOR [Name] <operator> [Value]%`\n- **TOP_N_STOCKS**: `TOP_N_STOCKS [N] <operator> [Value]%`\n- ... and more.")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file); df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/','_') for col in df.columns]
            header_map = {'isin': 'ISIN', 'name_of_the_instrument': 'Name', 'symbol': 'Symbol', 'industry': 'Industry', 'quantity': 'Quantity', 'rating': 'Rating', 'asset_class': 'Asset Class', 'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'}
            df = df.rename(columns=header_map)
            for col in ['Rating', 'Asset Class', 'Industry']:
                if col in df.columns: df[col] = df[col].str.strip().str.upper()
            if st.button("Analyze & Validate Portfolio", type="primary", use_container_width=True):
                with st.spinner("Fetching live prices and analyzing portfolio..."):
                    symbols = df['Symbol'].unique().tolist(); ltp_data = kite_client.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols]); prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    df_results = df.copy(); df_results['LTP'] = df_results['Symbol'].map(prices); df_results['Real-time Value (Rs)'] = (df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum(); df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    st.session_state.compliance_results_df = df_results; st.session_state.advanced_metrics = None
        except Exception as e: st.error(f"Failed to process CSV file. Error: {e}")
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    if not results_df.empty and 'Weight %' in results_df.columns:
        st.markdown("---")
        analysis_tabs = st.tabs(["📊 Dashboard", "🔍 Breakdowns", "📈 Advanced Analytics", "⚖️ Compliance Check", "📄 Detailed Holdings"])
        with analysis_tabs[0]:
            st.subheader("Portfolio Dashboard"); total_value = results_df['Real-time Value (Rs)'].sum(); kpi_cols = st.columns(4)
            kpi_cols[0].metric("Portfolio Value", f"₹ {total_value:,.2f}"); kpi_cols[1].metric("Holdings Count", f"{len(results_df)}"); kpi_cols[2].metric("Unique Sectors", f"{results_df['Industry'].nunique()}");
            if 'Rating' in results_df.columns: kpi_cols[3].metric("Unique Ratings", f"{results_df['Rating'].nunique()}")
            st.markdown("#### Concentration Analysis"); conc_cols = st.columns(2)
            with conc_cols[0]:
                st.metric("Top 5 Stocks Weight", f"{results_df.nlargest(5, 'Weight %')['Weight %'].sum():.2f}%"); st.metric("Top 10 Stocks Weight", f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%"); st.metric("Top 3 Sectors Weight", f"{results_df.groupby('Industry')['Weight %'].sum().nlargest(3).sum():.2f}%")
            with conc_cols[1]:
                stock_hhi = (results_df['Weight %'] ** 2).sum(); sector_hhi = (results_df.groupby('Industry')['Weight %'].sum() ** 2).sum()
                def get_hhi_category(score): return "Low Concentration" if score < 1500 else "Moderate" if score <= 2500 else "High Concentration"
                st.metric("Stock HHI", f"{stock_hhi:,.0f}", help=get_hhi_category(stock_hhi)); st.metric("Sector HHI", f"{sector_hhi:,.0f}", help=get_hhi_category(sector_hhi))
        with analysis_tabs[1]:
            st.subheader("Portfolio Breakdowns"); bd_cols = st.columns(2)
            with bd_cols[0]: st.markdown("##### Top 10 Holdings"); top_10_stocks = results_df.nlargest(10, 'Weight %'); fig_bar_stocks = px.bar(top_10_stocks, x='Weight %', y='Name', orientation='h', title='Top 10 Holdings by Weight'); fig_bar_stocks.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig_bar_stocks, use_container_width=True)
            with bd_cols[1]: st.markdown("##### Sector Exposure"); sector_weights = results_df.groupby('Industry')['Weight %'].sum().nlargest(10).reset_index(); fig_bar_sectors = px.bar(sector_weights, x='Weight %', y='Industry', orientation='h', title='Top 10 Sector Exposures'); fig_bar_sectors.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig_bar_sectors, use_container_width=True)
            st.markdown("##### Interactive Treemap"); fig_treemap = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Name'], values='Real-time Value (Rs)', hover_data={'Weight %': ':.2f%'}); fig_treemap.update_layout(margin = dict(t=25, l=25, r=25, b=25)); st.plotly_chart(fig_treemap, use_container_width=True)
        with analysis_tabs[2]:
            st.subheader("Advanced Risk & Return Metrics")
            if st.button("Calculate Advanced Metrics (VaR, Beta)", key="calc_adv_metrics"):
                with st.spinner("Calculating... This may take a minute."): st.session_state.advanced_metrics = calculate_advanced_metrics(results_df, api_key, access_token)
            if st.session_state.advanced_metrics:
                metrics = st.session_state.advanced_metrics; adv_cols = st.columns(2)
                adv_cols[0].metric("Daily Value at Risk (95%)", f"{metrics['var_95'] * 100:.2f}%", help="Expected maximum loss in a single day, with 95% confidence.")
                if metrics['beta'] is not None: adv_cols[1].metric(f"Portfolio Beta (vs {BENCHMARK_SYMBOL})", f"{metrics['beta']:.2f}", help="Volatility relative to the market. >1 is more volatile, <1 is less volatile.")
            else: st.info("Click the button to compute advanced metrics based on 1 year of historical data.")
        with analysis_tabs[3]:
            st.subheader("Compliance Rule Validation")
            validation_results = parse_and_validate_rules(rules_text, results_df)
            if not validation_results: st.info("Define rules to see validation results.")
            else:
                for res in validation_results:
                    if res['status'] == "✅ PASS": st.success(f"**{res['status']}:** `{res['rule']}` ({res['details']})")
                    elif res['status'] == "❌ FAIL": st.error(f"**{res['status']}:** `{res['rule']}` ({res['details']})")
                    else: st.warning(f"**{res['status']}:** `{res['rule']}` ({res['details']})")
        with analysis_tabs[4]:
            st.subheader("Detailed Holdings View"); display_df = results_df.copy(); format_dict = {'Real-time Value (Rs)': '₹{:,.2f}', 'LTP': '₹{:,.2f}', 'Weight %': '{:.2f}%'}
            column_order = ['Name', 'Symbol', 'Industry', 'Real-time Value (Rs)', 'Weight %', 'Quantity', 'LTP']
            if 'Asset Class' in display_df.columns: column_order.insert(3, 'Asset Class')
            if 'Rating' in display_df.columns: column_order.insert(3, 'Rating')
            display_columns = [col for col in column_order if col in display_df.columns]
            st.dataframe(display_df[display_columns].style.format(format_dict), use_container_width=True)
            st.download_button("📥 Download Full Report (CSV)", display_df.to_csv(index=False).encode('utf-8'), f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# --- AI ANALYSIS TAB FUNCTIONS ---
def extract_text_from_files(uploaded_files):
    full_text = ""
    for file in uploaded_files:
        full_text += f"\n\n--- DOCUMENT: {file.name} ---\n\n"
        if file.type == "application/pdf":
            with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                for page in doc: full_text += page.get_text()
        else: full_text += file.getvalue().decode("utf-8")
    return full_text

def get_portfolio_summary(df):
    if df.empty: return "No portfolio data available."
    total_value = df['Real-time Value (Rs)'].sum()
    top_10_stocks = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    summary = f"**Portfolio Snapshot (as of {datetime.now().strftime('%Y-%m-%d')})**\n- **Total Value:** ₹ {total_value:,.2f}\n- **Number of Holdings:** {len(df)}\n\n**Top 10 Holdings:**\n"
    for _, row in top_10_stocks.iterrows(): summary += f"- {row['Name']}: {row['Weight %']:.2f}%\n"
    summary += "\n**Top 10 Sector Exposures:**\n"
    for sector, weight in sector_weights.items(): summary += f"- {sector}: {weight:.2f}%\n"
    return summary

def render_ai_analysis_tab(kite_client):
    st.header("🤖 AI-Powered Compliance Analysis (with Google Gemini)")
    st.markdown("Analyze your portfolio against scheme documents (SID/KIM) and general regulatory guidelines.")
    portfolio_df = st.session_state.get("compliance_results_df")
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Please upload and analyze a portfolio in the 'Investment Compliance' tab first."); return
    st.info("This tool uses AI for analysis. The output is for informational purposes and is not financial or legal advice. Verify all findings independently.", icon="💡")
    uploaded_files = st.file_uploader("Upload Scheme Information Documents (SID), KIMs, etc.", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
            with st.spinner("Reading documents and preparing for AI analysis..."):
                try:
                    docs_text = extract_text_from_files(uploaded_files); portfolio_summary = get_portfolio_summary(portfolio_df)
                    prompt = f"""You are an expert investment compliance analyst for an Indian Asset Management Company. Your task is to analyze a given investment portfolio against the provided scheme documents and general SEBI/AMFI regulations for mutual funds.
                    **PORTFOLIO DATA:**
                    ```
                    {portfolio_summary}
                    ```
                    **SCHEME DOCUMENT(S) TEXT:**
                    ```
                    {docs_text[:150000]} 
                    ```
                    (Note: Document text may be truncated for brevity)
                    **YOUR TASK:**
                    Based on ALL the information above, provide a comprehensive compliance report. Structure your response in markdown format with the following sections:
                    1.  **Executive Summary:** A brief overview of your key findings. Is the portfolio generally aligned with the scheme's objectives?
                    2.  **Investment Objective & Strategy Alignment:** Compare the portfolio's composition (top holdings, sector weights) with the stated investment strategy in the document (e.g., large-cap focus, sector-agnostic, value-oriented). Point out specific holdings or sector concentrations that align or deviate.
                    3.  **Specific Restriction Checks:** Based on the scheme document, check for any violations of specific rules mentioned (e.g., maximum exposure to a single stock, single sector limits, derivatives usage, etc.). Quote the relevant clause from the document if you find a potential breach.
                    4.  **General Regulatory Health Check (SEBI/AMFI Guidelines):** Briefly assess the portfolio against common regulatory norms for mutual funds in India. Key points to consider:
                        - **Diversification & Concentration Risk:** Analyze the weight of the top 5/10 holdings and sector exposures. Is there significant concentration risk?
                        - **Liquidity:** (Qualitative assessment) Are the top holdings generally liquid, large-cap stocks?
                    5.  **Potential Risks & Recommendations:** Highlight any potential risks (concentration, style drift, etc.) and provide actionable recommendations for the fund manager.
                    Be clear, concise, and professional. Use bullet points and bold text to structure your analysis. If the documents lack specific quantifiable rules, state that and perform a qualitative analysis based on the fund's stated philosophy.
                    """
                    with st.spinner("AI is analyzing... This may take a moment."):
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        ]
                        response = model.generate_content(prompt, safety_settings=safety_settings)
                        st.session_state.ai_analysis_response = response.text
                except Exception as e:
                    st.error(f"An error occurred during AI analysis: {e}")
                    st.session_state.ai_analysis_response = None
    if st.session_state.get("ai_analysis_response"):
        st.markdown("---"); st.subheader("AI Analysis Report")
        st.markdown(st.session_state.ai_analysis_response)

# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

with tab_market: 
    render_market_historical_tab(k, api_key, access_token)
with tab_compliance:
    render_investment_compliance_tab(k, api_key, access_token)
with tab_ai:
    render_ai_analysis_tab(k)
