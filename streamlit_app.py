# streamlit_kite_app.py
import streamlit as st
from kiteconnect import KiteConnect
from supabase import create_client, Client
import pandas as pd
import json
import time
from typing import Optional, Dict, List

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
    st.error("Missing Kite credentials in Streamlit secrets.")
    st.stop()

if not (SUPABASE_URL and SUPABASE_KEY):
    st.error("Missing Supabase credentials in Streamlit secrets.")
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
    try:
        session = st.session_state.get("supabase_session")
        if session and hasattr(session, "user"):
            return session.user
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
# Sidebar - Supabase Auth
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
            user = (res.data or {}).get("user") if hasattr(res, "data") else None
            if user:
                st.session_state["supabase_session"] = res
                st.session_state["supabase_user"] = user
                st.experimental_rerun()
            else:
                err_msg = (res.error.message if hasattr(res, "error") and res.error else "Unknown error")
                st.sidebar.error(f"Login failed: {err_msg}")
        except Exception as e:
            st.sidebar.error(f"Login failed: {pretty_error(e)}")

    if col2.button("Sign up"):
        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            if hasattr(res, "error") and res.error:
                st.sidebar.error(f"Signup failed: {res.error.message}")
            else:
                st.sidebar.info("Signup created. Confirm email before login (check inbox).")
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
st.markdown(f"[🔗 Open Kite login]({kite_login_url})", unsafe_allow_html=True)

query_request_token = st.experimental_get_query_params().get("request_token")
request_token = query_request_token[0] if query_request_token else None

if request_token and "kite_access_token" not in st.session_state:
    st.info("Exchanging request_token for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if access_token:
            st.session_state["kite_access_token"] = access_token
            st.success("Kite access token saved in session.")
        else:
            st.error("No access token returned.")
    except Exception as e:
        st.error(f"Failed to generate Kite session: {pretty_error(e)}")

if "kite_access_token" not in st.session_state:
    st.warning("Login to Kite to fetch live quotes.")
    st.stop()

k = KiteConnect(api_key=API_KEY)
k.set_access_token(st.session_state["kite_access_token"])

# -------------------------
# Tabs
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
                st.error(f"Error fetching profile: {pretty_error(e)}")

        if st.button("📈 Get positions"):
            try:
                positions = k.positions()
                st.dataframe(pd.DataFrame(positions.get("net", []) if isinstance(positions, dict) else positions))
            except Exception as e:
                st.error(f"Error fetching positions: {pretty_error(e)}")

    with col2:
        symbol = st.text_input("Quick symbol (eg: INFY)", value="INFY", key="quick_sym")
        if st.button("Get quote"):
            price = fetch_last_price(k, symbol)
            if price is None:
                st.warning(f"No price available for {symbol}")
            else:
                st.success(f"{symbol} last price: {price}")

# -------------------------
# Create Index
# -------------------------
with tab_create:
    st.header("🧩 Create & Save Custom Index")
    uploaded_file = st.file_uploader("Upload CSV with columns: symbol, Name, Weights", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required = {"symbol", "Name", "Weights"}
            if not required.issubset(set(df.columns)):
                st.error(f"CSV must contain columns: {required}")
            else:
                df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
                df["Weights"] = pd.to_numeric(df["Weights"], errors="coerce")
                if df["Weights"].isnull().any():
                    st.error("Some Weights could not be parsed.")
                else:
                    df["Weights"] = df["Weights"] / df["Weights"].sum()
                    st.info("Fetching live prices...")
                    prices = batch_fetch_prices(k, df["symbol"].tolist(), sleep_between=0.12)
                    df["Last Price"] = df["symbol"].map(prices)
                    df["Weighted Price"] = df["Last Price"] * df["Weights"]
                    index_value = float(df["Weighted Price"].sum(skipna=True))
                    st.dataframe(df)
                    st.markdown(f"### Current Index Value: **{index_value:.4f}**")

                    user = supabase_current_user()
                    if user:
                        idx_name = st.text_input("Index name", value="My Custom Index")
                        col_save1, col_save2 = st.columns([1,1])
                        if col_save1.button("💾 Save Index & Snapshot"):
                            try:
                                symbols_json = df[["symbol","Name","Weights"]].to_dict(orient="records")
                                idx_res = supabase.table("indices").insert({
                                    "user_id": user.get("id"),
                                    "name": idx_name,
                                    "symbols": json.dumps(symbols_json),
                                    "last_value": index_value
                                }).execute()
                                index_id = (idx_res.get("data") or [None])[0]["id"]
                                calc_details = df[["symbol","Last Price","Weights","Weighted Price"]].to_dict(orient="records")
                                supabase.table("index_calculations").insert({
                                    "index_id": index_id,
                                    "value": index_value,
                                    "details": json.dumps(calc_details)
                                }).execute()
                                st.success("Index & snapshot saved.")
                            except Exception as e:
                                st.error(f"Failed to save index: {pretty_error(e)}")
                    else:
                        st.warning("Login to Supabase to save indices.")

# -------------------------
# Saved Indices
# -------------------------
with tab_saved:
    st.header("💾 Saved Indices")
    user = supabase_current_user()
    if user:
        try:
            resp = supabase.table("indices").select("*").eq("user_id", user.get("id")).order("created_at", desc=True).execute()
            rows = resp.get("data", [])
            if not rows:
                st.info("No saved indices yet.")
            else:
                for r in rows:
                    with st.expander(f"{r['name']} — {r.get('last_value')}"):
                        st.json(json.loads(r["symbols"]))
                        colv1, colv2, colv3 = st.columns([1,1,1])
                        try:
                            if colv1.button("View History", key=f"hist_{r['id']}"):
                                calcs = supabase.table("index_calculations").select("*").eq("index_id", r["id"]).order("calculated_at", desc=True).limit(50).execute().get("data", [])
                                for c in calcs:
                                    st.write(f"Value: {c['value']} at {c['calculated_at']}")
                                    st.dataframe(pd.DataFrame(json.loads(c.get("details", "[]"))))
                        except Exception as e:
                            st.error(f"Failed to fetch history: {pretty_error(e)}")

                        try:
                            if colv2.button("Recalculate now (live)", key=f"recalc_{r['id']}"):
                                symbols = json.loads(r["symbols"])
                                syms = [s["symbol"] for s in symbols]
                                prices = batch_fetch_prices(k, syms, sleep_between=0.12)
                                rows_df = pd.DataFrame(symbols)
                                rows_df["Last Price"] = rows_df["symbol"].map(prices)
                                rows_df["Weighted Price"] = rows_df["Last Price"] * rows_df["Weights"]
                                value = float(rows_df["Weighted Price"].sum(skipna=True))
                                calc_details = rows_df[["symbol","Last Price","Weights","Weighted Price"]].to_dict(orient="records")
                                supabase.table("index_calculations").insert({
                                    "index_id": r["id"],
                                    "value": value,
                                    "details": json.dumps(calc_details)
                                }).execute()
                                supabase.table("indices").update({"last_value": value, "updated_at": "now()"}).eq("id", r["id"]).execute()
                                st.success(f"Recalculated & saved snapshot — {value:.4f}")
                        except Exception as e:
                            st.error(f"Recalc failed: {pretty_error(e)}")

                        try:
                            if colv3.button("Delete Index", key=f"del_{r['id']}"):
                                supabase.table("indices").delete().eq("id", r["id"]).execute()
                                st.success("Deleted index & snapshots. Refresh to update list.")
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {pretty_error(e)}")
        except Exception as e:
            st.error(f"Failed to fetch saved indices: {pretty_error(e)}")
    else:
        st.info("Login to Supabase to view saved indices.")

# -------------------------
# Account Tab
# -------------------------
with tab_account:
    st.header("👤 Account & App Info")
    user = supabase_current_user()
    st.json(user if user else "Not logged in.")
    st.markdown("---")
    st.write("- RLS expected on Supabase tables as per SQL.")
    st.write("- Encrypt any persisted Kite access tokens if stored.")
    st.write("- For heavy usage, consider batch quote APIs and retry/backoff logic.")
