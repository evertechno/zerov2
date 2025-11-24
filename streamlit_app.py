import streamlit as st
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta  # Technical Analysis library

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Kite Connect - Analysis & Bulk Data", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect (Lite)")
st.markdown("A platform for market data analysis, machine learning insights, and bulk data extraction.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"

# Initialize session state variables
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "kt_ticker" not in st.session_state:
    st.session_state["kt_ticker"] = None
if "kt_thread" not in st.session_state:
    st.session_state["kt_thread"] = None
if "kt_running" not in st.session_state:
    st.session_state["kt_running"] = False
if "kt_ticks" not in st.session_state:
    st.session_state["kt_ticks"] = []
if "kt_live_prices" not in st.session_state:
    st.session_state["kt_live_prices"] = pd.DataFrame(columns=['timestamp', 'last_price', 'instrument_token'])
if "kt_status_message" not in st.session_state:
    st.session_state["kt_status_message"] = "Not started"
if "_rerun_ws" not in st.session_state:
    st.session_state["_rerun_ws"] = False


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n```")
        st.stop()
    return kite_conf

KITE_CREDENTIALS = load_secrets()

# --- KiteConnect Client Initialization (Unauthenticated for login URL) ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...") 
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    """Returns pandas.DataFrame of instrument data."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return {"_error": "Kite not authenticated."}
    
    exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
    try:
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol)
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})

    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]})

    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})

    from_datetime = datetime.combine(from_date, datetime.min.time())
    to_datetime = datetime.combine(to_date, datetime.max.time())
    try:
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None

def add_technical_indicators(df: pd.DataFrame, sma_short=10, sma_long=50, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std_dev=2) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['SMA_Short'] = ta.trend.sma_indicator(df_copy['close'], window=sma_short)
    df_copy['SMA_Long'] = ta.trend.sma_indicator(df_copy['close'], window=sma_long)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_signal'] = macd_obj.macd_signal()
    df_copy['MACD_hist'] = macd_obj.macd_diff() 
    
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'] = bollinger.bollinger_hband()
    df_copy['Bollinger_Low'] = bollinger.bollinger_lband()
    df_copy['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df_copy['Bollinger_Width'] = bollinger.bollinger_wband()
    
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    df_copy['Lag_1_Close'] = df_copy['close'].shift(1)
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    daily_returns_decimal = returns_series / 100.0
    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100

    num_periods = len(returns_series)
    if num_periods > 0:
        annualized_return = ((1 + daily_returns_decimal).prod())**(TRADING_DAYS_PER_YEAR/num_periods) - 1
    else:
        annualized_return = 0
    annualized_return *= 100

    daily_volatility = returns_series.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_volatility is not None else 0

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / (peak + 1e-9)
    max_drawdown = drawdown.min() * 100

    negative_returns = returns_series[returns_series < 0]
    downside_std_dev = negative_returns.std()
    sortino_ratio = (annualized_return - risk_free_rate) / (downside_std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std_dev != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Sortino Ratio": sortino_ratio
    }


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear() 
            st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame()
            st.success("Logged out.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

    st.markdown("---")
    st.markdown("### Quick Data Access")
    if st.session_state["kite_access_token"]:
        current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
        if st.button("Fetch Holdings", key="sidebar_fetch_holdings_btn"):
            try:
                holdings = current_k_client_for_sidebar.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
    else:
        st.info("Login to access data.")


# --- Authenticated KiteConnect client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main UI - Tabs ---
tabs = st.tabs(["Dashboard", "Portfolio", "Market & Historical", "Machine Learning", "Risk Analysis", "Performance", "Multi-Asset", "Websocket", "Instruments & Bulk Data"])
tab_dashboard, tab_portfolio, tab_market, tab_ml, tab_risk, tab_performance, tab_multi_asset, tab_ws, tab_inst = tabs

# --- Tab Functions ---

def render_dashboard_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Dashboard")
    if not kite_client:
        st.info("Please login to Kite Connect.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Account")
        try:
            profile = kite_client.profile()
            margins = kite_client.margins()
            st.metric("Holder", profile.get("user_name", "N/A"))
            st.metric("Equity Margin", f"₹{margins.get('equity', {}).get('available', {}).get('live_balance', 0):,.2f}")
        except Exception as e:
            st.warning(f"Error: {e}")

    with col2:
        st.subheader("NIFTY 50")
        if api_key and access_token:
            nifty_ltp_data = get_ltp_price_cached(api_key, access_token, "NIFTY 50", DEFAULT_EXCHANGE)
            if nifty_ltp_data and "_error" not in nifty_ltp_data:
                nifty_ltp = nifty_ltp_data.get("last_price", 0.0)
                st.metric("LTP", f"₹{nifty_ltp:,.2f}")
            else:
                st.warning("Could not fetch NIFTY 50.")

    with col3:
        st.subheader("Quick Perf.")
        if st.session_state.get("last_fetched_symbol") and not st.session_state.get("historical_data", pd.DataFrame()).empty:
            last_symbol = st.session_state["last_fetched_symbol"]
            returns = st.session_state["historical_data"]["close"].pct_change().dropna() * 100
            if not returns.empty:
                perf = calculate_performance_metrics(returns)
                st.write(f"**{last_symbol}**")
                st.metric("Total Return", f"{perf.get('Total Return (%)', 0):.2f}%")
            else:
                st.info("Insufficient data.")
        else:
            st.info("Fetch historical data first.")

def render_portfolio_tab(kite_client: KiteConnect | None):
    st.header("Portfolio")
    if not kite_client:
        st.info("Login required.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch Holdings", key="port_hold_btn"):
            try:
                holdings = kite_client.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success("Updated.")
            except Exception as e: st.error(str(e))
        if not st.session_state.get("holdings_data", pd.DataFrame()).empty:
            st.dataframe(st.session_state["holdings_data"], use_container_width=True)

    with col2:
        if st.button("Fetch Positions", key="port_pos_btn"):
            try:
                positions = kite_client.positions()
                st.session_state["net_positions"] = pd.DataFrame(positions.get("net", []))
                st.success("Updated.")
            except Exception as e: st.error(str(e))
        if not st.session_state.get("net_positions", pd.DataFrame()).empty:
            st.dataframe(st.session_state["net_positions"], use_container_width=True)

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Market Data")
    if not kite_client:
        st.info("Login required.")
        return

    st.subheader("Single Symbol Analysis")
    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="hist_sym_tab_input")
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=90), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["minute", "5minute", "30minute", "day", "week", "month"], index=3, key="hist_interval_selector")

        if st.button("Fetch Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} data..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange)
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.success(f"Fetched {len(df_hist)} records.")
                else:
                    st.error(f"Error: {df_hist.get('_error', 'Unknown error')}")

    with col_hist_plot:
        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
            fig.update_layout(title_text=f"{st.session_state['last_fetched_symbol']}", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

def render_ml_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Machine Learning Analysis")
    if not kite_client: return st.info("Login required.")
    
    df = st.session_state.get("historical_data", pd.DataFrame())
    if df.empty: return st.warning("Fetch historical data first.")

    st.subheader("Model Training")
    df = add_technical_indicators(df)
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    features = ['SMA_Short', 'SMA_Long', 'RSI', 'MACD', 'Bollinger_Mid']
    X = df[features]
    y = df['target']
    
    if len(X) > 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        st.metric("R2 Score", f"{r2_score(y_test, preds):.4f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual'))
        fig.add_trace(go.Scatter(x=y_test.index, y=preds, name='Predicted'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data points.")

def render_risk_stress_testing_tab(kite_client: KiteConnect | None):
    st.header("Risk Analysis")
    df = st.session_state.get("historical_data", pd.DataFrame())
    if df.empty: return st.warning("Fetch historical data first.")
    
    returns = df['close'].pct_change().dropna()
    var_95 = np.percentile(returns, 5) * 100
    st.metric("VaR (95%)", f"{var_95:.2f}%")
    
    fig = go.Figure(go.Histogram(x=returns, nbinsx=50, name='Returns'))
    st.plotly_chart(fig, use_container_width=True)

def render_performance_analysis_tab(kite_client: KiteConnect | None):
    st.header("Performance")
    df = st.session_state.get("historical_data", pd.DataFrame())
    if df.empty: return st.warning("Fetch historical data first.")
    
    returns = df['close'].pct_change().dropna() * 100
    metrics = calculate_performance_metrics(returns)
    col1, col2 = st.columns(2)
    for k, v in metrics.items():
        col1.metric(k, f"{v:.2f}")

    cum_ret = (1 + returns/100).cumprod()
    st.line_chart(cum_ret)

def render_multi_asset_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Multi-Asset Correlation")
    if not kite_client: return st.info("Login required.")
    
    symbols = st.text_input("Symbols (comma separated)", "INFY,TCS,RELIANCE").split(',')
    if st.button("Analyze"):
        data = {}
        for s in symbols:
            s = s.strip().upper()
            df = get_historical_data_cached(api_key, access_token, s, datetime.now().date()-timedelta(days=365), datetime.now().date(), "day")
            if not df.empty and "_error" not in df:
                data[s] = df['close']
        
        if len(data) > 1:
            res = pd.DataFrame(data).pct_change().corr()
            st.write(res.style.background_gradient(cmap='RdBu'))
        else:
            st.error("Need at least 2 valid symbols.")

def render_websocket_tab(kite_client: KiteConnect | None):
    st.header("Live Ticks (WebSocket)")
    if not kite_client: return st.info("Login required.")
    
    token_input = st.text_input("Instrument Token(s)", "256265") # NIFTY 50
    if st.button("Start"):
        if not st.session_state["kt_running"]:
            kt = KiteTicker(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
            st.session_state["kt_ticker"] = kt
            st.session_state["kt_running"] = True
            
            def on_ticks(ws, ticks):
                for t in ticks:
                    t["_ts"] = datetime.utcnow().isoformat()
                    st.session_state["kt_ticks"].append(t)
                    if 'last_price' in t:
                        new_row = pd.DataFrame([{'timestamp': datetime.now(), 'last_price': t['last_price'], 'instrument_token': t['instrument_token']}])
                        st.session_state["kt_live_prices"] = pd.concat([st.session_state["kt_live_prices"], new_row], ignore_index=True)
                if len(st.session_state["kt_ticks"]) > 100: st.session_state["kt_ticks"] = st.session_state["kt_ticks"][-100:]
                st.session_state["_rerun_ws"] = True

            kt.on_ticks = on_ticks
            kt.on_connect = lambda ws, r: ws.subscribe([int(x) for x in token_input.split(',')])
            
            t = threading.Thread(target=lambda: kt.connect(daemon=True), daemon=True)
            t.start()
            st.session_state["kt_thread"] = t
            st.rerun()

    if st.button("Stop"):
        if st.session_state["kt_running"]:
            st.session_state["kt_ticker"].disconnect()
            st.session_state["kt_running"] = False
            st.rerun()

    if st.session_state["_rerun_ws"]:
        st.session_state["_rerun_ws"] = False
        st.dataframe(pd.DataFrame(st.session_state["kt_ticks"]).tail(10))
        if st.session_state["kt_running"]:
            time.sleep(1)
            st.rerun()

def render_instruments_utils_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Instruments & Bulk Data Downloader")
    if not kite_client: return st.info("Login required.")
    if not api_key or not access_token: return st.info("Auth required.")

    # --- 1. Load Instruments ---
    inst_exchange = st.selectbox("Select Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], key="inst_bulk_ex")
    if st.button("Load Instruments", key="inst_bulk_load_btn"):
        df_instruments = load_instruments_cached(api_key, access_token, inst_exchange)
        if not df_instruments.empty and "_error" not in df_instruments.columns:
            st.session_state["instruments_df"] = df_instruments
            st.success(f"Loaded {len(df_instruments)} instruments.")
        else:
            st.error(f"Failed: {df_instruments.get('_error', 'Unknown')}")

    df_inst = st.session_state.get("instruments_df", pd.DataFrame())
    
    # --- 2. Instrument List CSV Download ---
    if not df_inst.empty:
        st.subheader("1. Download Instrument List")
        st.write(f"Available Instruments: {len(df_inst)}")
        csv_data = df_inst.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {inst_exchange} Instruments CSV",
            data=csv_data,
            file_name=f"{inst_exchange}_instruments.csv",
            mime="text/csv",
            key="download_all_inst_csv"
        )
        
        with st.expander("View Instruments"):
            st.dataframe(df_inst.head(100))

        st.markdown("---")

        # --- 3. Bulk Historical Data Downloader ---
        st.subheader("2. Bulk Historical Data Downloader (5 Years)")
        st.markdown(f"**Warning:** Fetching data for {len(df_inst)} instruments is intensive. Rate limits are applied automatically.")
        
        col_bulk_1, col_bulk_2 = st.columns(2)
        with col_bulk_1:
            bulk_end_date = datetime.now().date()
            bulk_start_date = bulk_end_date - timedelta(days=365*5) # 5 Years
            st.write(f"**Range:** {bulk_start_date} to {bulk_end_date}")
            
        with col_bulk_2:
            # Safety limiter
            max_inst = len(df_inst)
            limit_fetch = st.number_input("Limit number of instruments (0 = All)", min_value=0, max_value=max_inst, value=10, help="Set to 0 to fetch ALL. Start small to test.")
        
        if st.button("Start Bulk Download", key="start_bulk_download"):
            
            # Prepare list
            targets = df_inst if limit_fetch == 0 else df_inst.head(limit_fetch)
            total_targets = len(targets)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_bulk_data = []
            
            # KiteConnect instance for the loop
            kc = get_authenticated_kite_client(api_key, access_token)
            
            start_ts = datetime.combine(bulk_start_date, datetime.min.time())
            end_ts = datetime.combine(bulk_end_date, datetime.max.time())
            
            for i, row in targets.iterrows():
                symbol = row['tradingsymbol']
                token = row['instrument_token']
                
                status_text.text(f"Fetching {i+1}/{total_targets}: {symbol} ({token})")
                
                try:
                    # Direct API call to avoid cache overhead for massive unique fetches
                    records = kc.historical_data(token, from_date=start_ts, to_date=end_ts, interval="day")
                    if records:
                        temp_df = pd.DataFrame(records)
                        temp_df['symbol'] = symbol
                        temp_df['instrument_token'] = token
                        # Keep it lean
                        cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'instrument_token']
                        all_bulk_data.append(temp_df[cols])
                except Exception as e:
                    # Log error but don't stop the whole process
                    print(f"Error fetching {symbol}: {e}")
                
                # Update progress
                progress_bar.progress((i + 1) / total_targets)
                
                # RATE LIMITING: Sleep to respect ~3 req/sec
                time.sleep(0.4) 
            
            status_text.text("Processing complete. Compiling CSV...")
            
            if all_bulk_data:
                master_bulk_df = pd.concat(all_bulk_data, ignore_index=True)
                st.success(f"Successfully fetched {len(master_bulk_df)} rows for {total_targets} instruments.")
                
                csv_bulk = master_bulk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Bulk Historical Data (ZIP/CSV)",
                    data=csv_bulk,
                    file_name=f"bulk_history_{inst_exchange}_5years.csv",
                    mime="text/csv",
                    key="download_bulk_final"
                )
            else:
                st.warning("No data fetched. Market closed or tokens invalid?")
                
    else:
        st.info("Load instruments first to enable download options.")

# --- Run Logic ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

with tab_dashboard: render_dashboard_tab(k, api_key, access_token)
with tab_portfolio: render_portfolio_tab(k)
with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_ml: render_ml_analysis_tab(k, api_key, access_token)
with tab_risk: render_risk_stress_testing_tab(k)
with tab_performance: render_performance_analysis_tab(k)
with tab_multi_asset: render_multi_asset_analysis_tab(k, api_key, access_token)
with tab_ws: render_websocket_tab(k)
with tab_inst: render_instruments_utils_tab(k, api_key, access_token)
