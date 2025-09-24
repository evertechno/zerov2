# streamlit_kite_app_prod.py
import streamlit as st
from kiteconnect import KiteConnect
from supabase import create_client, Client
import pandas as pd
import json
import time
from typing import Optional, Dict, Any, List

st.set_page_config(page_title="Kite Connect + Supabase Index", layout="wide")
st.title("📊 Kite Connect (Zerodha) + Supabase Index (Prod-ready)")

# -------------------------
# Load config from secrets
# -------------------------
def load_kite_conf():
    try:
        kite_conf = st.secrets["kite"]
        return kite_conf.get("api_key"), kite_conf.get("api_secret"), kite_conf.get("redirect_uri")
    except Exception:
        return None, None, None

def load_supabase_conf():
    try:
        sb = st.secrets["supabase"]
        return sb.get("url"), sb.get("anon_key")
    except Exception:
        return None, None

API_KEY, API_SECRET, REDIRECT_URI = load_kite_conf()
SUPABASE_URL, SUPABASE_KEY = load_supabase_conf()

if not (API_KEY and API_SECRET and REDIRECT_URI):
    st.error("Missing Kite credentials in Streamlit secrets. Add [kite] api_key, api_secret and redirect_uri.")
    st.stop()

if not (SUPABASE_URL and SUPABASE_KEY):
    st.error("Missing Supabase credentials in Streamlit secrets. Add [supabase] url and anon_key.")
    st.stop()

# -------------------------
# Initialize clients
# -------------------------
kite_client = KiteConnect(api_key=API_KEY)
kite_login_url = kite_client.login_url()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Helper functions
# -------------------------
def supabase_current_user():
    """Return logged-in supabase user (or None)."""
    try:
        session = supabase.auth.session()
        if session and "user" in session and session["user"]:
            return session["user"]
        return st.session_state.get("supabase_user")
    except Exception:
        return st.session_state.get("supabase_user")

def pretty_error(e: Exception) -> str:
    return str(e)

def fetch_last_price(kite: KiteConnect, symbol: str, exchange_prefix="NSE") -> Optional[float]:
    try:
        q = kite.quote(f"{exchange_prefix}:{symbol}")
        key = f"{exchange_prefix}:{symbol}"
        if q and key in q and "last_price" in q[key]:
            return float(q[key]["last_price"])
        return None
    except Exception:
        return None

def batch_fetch_prices(kite: KiteConnect, syms: List[str], exchange_prefix="NSE", sleep_between=0.15) -> Dict[str, Optional[float]]:
    out = {}
    for s in syms:
        out[s] = fetch_last_price(kite, s, exchange_prefix=exchange_prefix)
        time.sleep(sleep_between)
    return out

# -------------------------
# Sidebar: Supabase Auth
# -------------------------
st.sidebar.header("Authentication")

if "supabase_session" not in st.session_state:
    st.session_state["supabase_session"] = None

if not supabase_current_user():
    st.sidebar.subheader("Supabase Login / Sign up")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Login"):
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            user = res.get("user") if isinstance(res, dict) else (res.get("data") or {}).get("user")
            if user:
                st.session_state["supabase_session"] = res
                st.session_state["supabase_user"] = user
                st.experimental_rerun()
            else:
                st.sidebar.error("Login response did not contain user.")
        except Exception as e:
            st.sidebar.error(f"Login failed: {pretty_error(e)}")

    if col2.button("Sign up"):
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            st.sidebar.info("Signup created. Confirm email before login.")
        except Exception as e:
            st.sidebar.error(f"Sign up failed: {pretty_error(e)}")

else:
    user = supabase_current_user()
    user_email = user.get("email") if isinstance(user, dict) else None
    st.sidebar.success(f"Signed in: {user_email}")
    if st.sidebar.button("Logout"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.pop("supabase_session", None)
        st.session_state.pop("supabase_user", None)
        st.experimental_rerun()

# -------------------------
# Kite Login
# -------------------------
st.markdown("### Step 1 — Kite Login (Zerodha)")
st.write("Click the Kite login link below and complete the flow. After login you'll be redirected to your redirect_uri with a `request_token` in query params.")
st.markdown(f"[🔗 Open Kite login]({kite_login_url})", unsafe_allow_html=True)

query_request_token = st.experimental_get_query_params().get("request_token")
request_token = query_request_token[0] if query_request_token else None

if request_token and "kite_access_token" not in st.session_state:
    st.info("Received request_token — exchanging for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if access_token:
            st.session_state["kite_access_token"] = access_token
            st.success("Kite access token saved in session.")
        else:
            st.error("No access token returned from Kite.")
    except Exception as e:
        st.error(f"Failed to generate Kite session: {pretty_error(e)}")

if "kite_access_token" not in st.session_state:
    st.warning("Please login to Kite using the link above to fetch live quotes.")
    st.stop()

k = KiteConnect(api_key=API_KEY)
k.set_access_token(st.session_state["kite_access_token"])

# -------------------------
# Main Tabs
# -------------------------
tab_dashboard, tab_create, tab_saved, tab_account = st.tabs(["Dashboard", "Create Index", "Saved Indices", "Account"])

# -------------------------
# Dashboard
# -------------------------
with tab_dashboard:
    st.header("📊 Dashboard")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("👤 Fetch Kite profile"):
            try:
                st.json(k.profile())
            except Exception as e:
                st.error(f"Error: {pretty_error(e)}")
        if st.button("📈 Get positions"):
            try:
                positions = k.positions()
                st.dataframe(pd.DataFrame(positions.get("net", []) if isinstance(positions, dict) else positions))
            except Exception as e:
                st.error(f"Error: {pretty_error(e)}")
    with col2:
        symbol = st.text_input("Quick symbol", value="INFY", key="quick_sym")
        if st.button("Get quote"):
            price = fetch_last_price(k, symbol)
            if price is None:
                st.warning(f"No price for {symbol}")
            else:
                st.success(f"{symbol} last price: {price}")

# -------------------------
# Create Index
# -------------------------
with tab_create:
    st.header("🧩 Create & Save Custom Index")
    uploaded_file = st.file_uploader("Upload CSV with columns: symbol, Name, Weights", type=["csv"], key="upload_index_csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV read failed: {pretty_error(e)}")
            df = None

        if df is not None:
            required = {"symbol", "Name", "Weights"}
            if not required.issubset(set(df.columns)):
                st.error(f"CSV must contain columns: {required}")
            else:
                df = df.copy()
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
                df["Weights"] = pd.to_numeric(df["Weights"], errors="coerce")
                if df["Weights"].isnull().any():
                    st.error("Some Weights invalid.")
                else:
                    total_weight = df["Weights"].sum()
                    if total_weight == 0:
                        st.error("Sum of Weights is 0.")
                    else:
                        df["Weights"] /= total_weight
                        st.info("Fetching live prices...")
                        prices = batch_fetch_prices(k, df["symbol"].tolist(), sleep_between=0.12)
                        df["Last Price"] = df["symbol"].map(prices)
                        missing = df[df["Last Price"].isnull()]
                        if not missing.empty:
                            st.warning(f"Missing prices: {', '.join(missing['symbol'].tolist())}")
                        df["Weighted Price"] = df["Last Price"] * df["Weights"]
                        index_value = float(df["Weighted Price"].sum(skipna=True))
                        st.subheader("Index Constituents")
                        st.dataframe(df)
                        st.markdown(f"### Current Index Value: **{index_value:.4f}**")

                        user = supabase_current_user()
                        if user:
                            idx_name = st.text_input("Index name", value="My Custom Index", key="index_name_input")
                            col_save1, col_save2 = st.columns([1,1])
                            if col_save1.button("💾 Save Index & Snapshot"):
                                try:
                                    symbols_json = df[["symbol", "Name", "Weights"]].to_dict(orient="records")
                                    insert_idx = supabase.table("indices").insert({
                                        "user_id": user.get("id"),
                                        "name": idx_name,
                                        "symbols": json.dumps(symbols_json),
                                        "last_value": index_value
                                    }).execute()
                                    idx_data = (insert_idx.data or [None])[0] if hasattr(insert_idx, "data") else (insert_idx.get("data") or [None])[0]
                                    if not idx_data or "id" not in idx_data:
                                        st.error("Failed to create index.")
                                    else:
                                        index_id = idx_data["id"]
                                        calc_details = df[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                        supabase.table("index_calculations").insert({
                                            "index_id": index_id,
                                            "value": index_value,
                                            "details": json.dumps(calc_details)
                                        }).execute()
                                        st.success("Index and snapshot saved.")
                                except Exception as e:
                                    st.error(f"Save failed: {pretty_error(e)}")

# -------------------------
# Saved Indices
# -------------------------
with tab_saved:
    st.header("💾 Saved Indices")
    user = supabase_current_user()
    if user:
        try:
            resp = supabase.table("indices").select("*").eq("user_id", user.get("id")).order("created_at", desc=True).execute()
            rows = resp.data if hasattr(resp, "data") else resp.get("data", [])
            if not rows:
                st.info("No saved indices.")
            else:
                for r in rows:
                    with st.expander(f"{r['name']} — {r.get('last_value')}"):
                        st.json(r["symbols"] if isinstance(r["symbols"], (dict,list)) else json.loads(r["symbols"]))
        except Exception as e:
            st.error(f"Load failed: {pretty_error(e)}")
    else:
        st.info("Login to view saved indices.")

# -------------------------
# Account
# -------------------------
with tab_account:
    st.header("👤 Account Info")
    user = supabase_current_user()
    if user:
        st.json(user)
    else:
        st.info("Not logged in.")
