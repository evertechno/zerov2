# streamlit_kite_app.py
"""
PROD-READY Streamlit app integrating:
- Kite Connect (Zerodha) for market quotes
- Supabase for Auth + storing indices + calculation snapshots

SQL to run once in Supabase SQL editor (example):
------------------------------------------------
-- Enable pgcrypto extension for gen_random_uuid() if not present:
create extension if not exists "pgcrypto";

create table public.indices (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  name text not null,
  symbols jsonb not null, -- array of {symbol, Name, Weights}
  last_value numeric,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.index_calculations (
  id uuid primary key default gen_random_uuid(),
  index_id uuid references public.indices(id) on delete cascade,
  value numeric not null,
  details jsonb, -- array of {symbol, LastPrice, Weights, WeightedPrice}
  calculated_at timestamptz default now()
);

alter table public.indices enable row level security;
alter table public.index_calculations enable row level security;

create policy "users can manage their indices"
on public.indices
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "users can manage their index_calculations"
on public.index_calculations
for all
using (
  exists (
    select 1 from public.indices i
    where i.id = index_id and i.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1 from public.indices i
    where i.id = index_id and i.user_id = auth.uid()
  )
);
------------------------------------------------
"""

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
        # supabase.auth.get_session / get_user depends on versions.
        session = supabase.auth.session()
        if session and "user" in session and session["user"]:
            return session["user"]
        # fallback: sign_in response stored in session_state
        return st.session_state.get("supabase_user")
    except Exception:
        return st.session_state.get("supabase_user")

def pretty_error(e: Exception) -> str:
    return str(e)

def fetch_last_price(kite: KiteConnect, symbol: str, exchange_prefix="NSE") -> Optional[float]:
    """Fetch last price for a single symbol; returns None on failure."""
    try:
        # Kite quote returns nested dict with key like "NSE:INFY"
        q = kite.quote(f"{exchange_prefix}:{symbol}")
        key = f"{exchange_prefix}:{symbol}"
        if q and key in q and "last_price" in q[key]:
            return float(q[key]["last_price"])
        return None
    except Exception:
        return None

def batch_fetch_prices(kite: KiteConnect, syms: List[str], exchange_prefix="NSE", sleep_between=0.15) -> Dict[str, Optional[float]]:
    """Simple rate-friendly batch fetcher (one-by-one with tiny sleep to avoid throttle)."""
    out = {}
    for s in syms:
        out[s] = fetch_last_price(kite, s, exchange_prefix=exchange_prefix)
        time.sleep(sleep_between)  # throttling guard
    return out

# -------------------------
# UI: Sidebar - Auth
# -------------------------
st.sidebar.header("Authentication")

# Supabase Auth UI (email/password)
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
            # sign_in_with_password returns dict-like; try to extract user
            user = res.get("user") if isinstance(res, dict) else None
            if not user:
                # some versions return {'data': {'user': ...}, 'error': None}
                user = (res.get("data") or {}).get("user") if isinstance(res, dict) else None
            if user:
                st.session_state["supabase_session"] = res
                st.session_state["supabase_user"] = user
                st.experimental_rerun()
            else:
                st.sidebar.error("Login response did not contain user. Check credentials.")
        except Exception as e:
            st.sidebar.error(f"Login failed: {pretty_error(e)}")

    if col2.button("Sign up"):
        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            st.sidebar.info("Signup created. Confirm email before login (check your inbox).")
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
# Kite Login (main area)
# -------------------------
st.markdown("### Step 1 — Kite Login (Zerodha)")
st.write("Click the Kite login link below and complete the flow. After login you'll be redirected to your redirect_uri with a `request_token` in query params.")
st.markdown(f"[🔗 Open Kite login]({kite_login_url})", unsafe_allow_html=True)

# After redirect, streamlit exposes query params; look for request_token
query_request_token = st.experimental_get_query_params().get("request_token")
if query_request_token:
    # query_request_token is a list
    request_token = query_request_token[0]
else:
    request_token = None

if request_token and "kite_access_token" not in st.session_state:
    st.info("Received request_token — exchanging for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if access_token:
            st.session_state["kite_access_token"] = access_token
            st.success("Kite access token saved in session.")
            # Optionally persist token to Supabase (if you want) - ensure encryption & security if storing
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
# Main UI: Tabs
# -------------------------
tab = st.tabs(["Dashboard", "Create Index", "Saved Indices", "Account"])[0]  # quick default tab variable

# Use an actual tab UI
tab_dashboard, tab_create, tab_saved, tab_account = st.tabs(["Dashboard", "Create Index", "Saved Indices", "Account"])

# -------------------------
# Dashboard tab
# -------------------------
with tab_dashboard:
    st.header("📊 Dashboard")
    st.write("Quick quote check / kite account basics.")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("👤 Fetch Kite profile"):
            try:
                profile = k.profile()
                st.json(profile)
            except Exception as e:
                st.error(f"Error fetching profile: {pretty_error(e)}")

        if st.button("📈 Get positions"):
            try:
                positions = k.positions()
                st.write("Net positions")
                st.dataframe(pd.DataFrame(positions.get("net", []) if isinstance(positions, dict) else positions))
            except Exception as e:
                st.error(f"Error fetching positions: {pretty_error(e)}")

    with col2:
        symbol = st.text_input("Quick symbol (eg: INFY)", value="INFY", key="quick_sym")
        if st.button("Get quote"):
            try:
                price = fetch_last_price(k, symbol)
                if price is None:
                    st.warning(f"No price available for {symbol}")
                else:
                    st.success(f"{symbol} last price: {price}")
            except Exception as e:
                st.error(f"Quote error: {pretty_error(e)}")

# -------------------------
# Create Index tab
# -------------------------
with tab_create:
    st.header("🧩 Create & Save Custom Index")

    uploaded_file = st.file_uploader("Upload CSV with columns: symbol, Name, Weights", type=["csv"], key="upload_index_csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {pretty_error(e)}")
            df = None

        if df is not None:
            required = {"symbol", "Name", "Weights"}
            if not required.issubset(set(df.columns)):
                st.error(f"CSV must contain columns: {required}")
            else:
                # Clean-up: uppercase symbol, numeric weights
                df = df.copy()
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
                # coerce weights numeric
                df["Weights"] = pd.to_numeric(df["Weights"], errors="coerce")
                if df["Weights"].isnull().any():
                    st.error("Some Weights could not be parsed as numbers. Fix CSV and re-upload.")
                else:
                    # Normalise weights (sum to 1)
                    total_weight = df["Weights"].sum()
                    if total_weight == 0:
                        st.error("Sum of Weights is 0. Provide non-zero weights.")
                    else:
                        df["Weights"] = df["Weights"] / total_weight

                        # Fetch live prices (with small throttle)
                        st.info("Fetching live prices (might take a moment)...")
                        prices = batch_fetch_prices(k, df["symbol"].tolist(), sleep_between=0.12)

                        df["Last Price"] = df["symbol"].map(prices)
                        # Warn if any None
                        missing = df[df["Last Price"].isnull()]
                        if not missing.empty:
                            st.warning(f"Could not fetch prices for: {', '.join(missing['symbol'].tolist())}")

                        df["Weighted Price"] = df["Last Price"] * df["Weights"]
                        index_value = float(df["Weighted Price"].sum(skipna=True))

                        st.subheader("Index Constituents")
                        st.dataframe(df)

                        st.markdown(f"### Current Index Value: **{index_value:.4f}**")

                        # Save options (requires Supabase user)
                        user = supabase_current_user()
                        if not user:
                            st.warning("Please login to Supabase (sidebar) to save indices.")
                        else:
                            idx_name = st.text_input("Index name", value="My Custom Index", key="index_name_input")
                            col_save1, col_save2 = st.columns([1, 1])
                            if col_save1.button("💾 Save Index & Snapshot"):
                                # Prepare payloads
                                try:
                                    symbols_json = df[["symbol", "Name", "Weights"]].to_dict(orient="records")
                                    # Insert index definition
                                    insert_idx = supabase.table("indices").insert({
                                        "user_id": user.get("id"),
                                        "name": idx_name,
                                        "symbols": json.dumps(symbols_json),
                                        "last_value": index_value
                                    }).execute()
                                    # handle result shapes
                                    idx_data = None
                                    if isinstance(insert_idx, dict):
                                        idx_data = (insert_idx.get("data") or [None])[0]
                                    else:
                                        idx_data = (insert_idx.data or [None])[0]
                                    if not idx_data or "id" not in idx_data:
                                        st.error("Failed to create index record in Supabase. Check logs/permissions.")
                                    else:
                                        index_id = idx_data["id"]
                                        calc_details = df[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                        supabase.table("index_calculations").insert({
                                            "index_id": index_id,
                                            "value": index_value,
                                            "details": json.dumps(calc_details)
                                        }).execute()
                                        st.success("Index and calculation snapshot saved.")
                                except Exception as e:
                                    st.error(f"Failed to save index: {pretty_error(e)}")

                            if col_save2.button("💾 Save Snapshot (to existing index)"):
                                # Let user pick an existing index to attach snapshot to
                                try:
                                    resp = supabase.table("indices").select("*").eq("user_id", user.get("id")).execute()
                                    rows = resp.data if hasattr(resp, "data") else resp.get("data", [])
                                    if not rows:
                                        st.info("No saved indices found; please create one first.")
                                    else:
                                        idx_map = {r["name"]: r["id"] for r in rows}
                                        choice = st.selectbox("Choose index", options=list(idx_map.keys()))
                                        if choice and st.button("Confirm Save Snapshot"):
                                            index_id = idx_map[choice]
                                            calc_details = df[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                            supabase.table("index_calculations").insert({
                                                "index_id": index_id,
                                                "value": index_value,
                                                "details": json.dumps(calc_details)
                                            }).execute()
                                            # update last_value on indices table
                                            supabase.table("indices").update({"last_value": index_value, "updated_at": "now()"}).eq("id", index_id).execute()
                                            st.success("Snapshot saved to existing index.")
                                except Exception as e:
                                    st.error(f"Failed saving snapshot: {pretty_error(e)}")

# -------------------------
# Saved Indices tab
# -------------------------
with tab_saved:
    st.header("💾 Saved Indices (your workspace)")
    user = supabase_current_user()
    if not user:
        st.info("Login to Supabase (sidebar) to view saved indices.")
    else:
        try:
            resp = supabase.table("indices").select("*").eq("user_id", user.get("id")).order("created_at", desc=True).execute()
            rows = resp.data if hasattr(resp, "data") else resp.get("data", [])
            if not rows:
                st.info("No saved indices yet. Create one in 'Create Index' tab.")
            else:
                # Show a simple list and allow actions
                for r in rows:
                    with st.expander(f"{r['name']} — last_value: {r.get('last_value')} — created: {r.get('created_at')}"):
                        st.json(r["symbols"] if isinstance(r["symbols"], (dict, list)) else json.loads(r["symbols"]))
                        colv1, colv2, colv3 = st.columns([1,1,1])
                        if colv1.button("View History", key=f"hist_{r['id']}"):
                            # fetch last 20 calculations
                            calc_resp = supabase.table("index_calculations").select("*").eq("index_id", r["id"]).order("calculated_at", desc=True).limit(50).execute()
                            calcs = calc_resp.data if hasattr(calc_resp, "data") else calc_resp.get("data", [])
                            if not calcs:
                                st.info("No calculations yet for this index.")
                            else:
                                for c in calcs:
                                    with st.container():
                                        st.write(f"Value: {c['value']} at {c['calculated_at']}")
                                        details = c.get("details")
                                        try:
                                            dd = details if isinstance(details, (list, dict)) else json.loads(details)
                                            st.dataframe(pd.DataFrame(dd))
                                        except Exception:
                                            st.text(str(details))
                        if colv2.button("Recalculate now (live)", key=f"recalc_{r['id']}"):
                            # Recalculate using latest symbols & save snapshot
                            try:
                                symbols = r["symbols"]
                                if isinstance(symbols, str):
                                    symbols = json.loads(symbols)
                                syms = [s["symbol"] for s in symbols]
                                prices = batch_fetch_prices(k, syms, sleep_between=0.12)
                                rows_df = pd.DataFrame(symbols)
                                rows_df["Last Price"] = rows_df["symbol"].map(prices)
                                rows_df["Weighted Price"] = rows_df["Last Price"] * rows_df["Weights"]
                                value = float(rows_df["Weighted Price"].sum(skipna=True))
                                calc_details = rows_df[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                supabase.table("index_calculations").insert({
                                    "index_id": r["id"],
                                    "value": value,
                                    "details": json.dumps(calc_details)
                                }).execute()
                                supabase.table("indices").update({"last_value": value, "updated_at": "now()"}).eq("id", r["id"]).execute()
                                st.success(f"Recalculated & saved snapshot — Value: {value:.4f}")
                            except Exception as e:
                                st.error(f"Recalc failed: {pretty_error(e)}")
                        if colv3.button("Delete Index", key=f"del_{r['id']}"):
                            # deletion (will cascade index_calculations)
                            try:
                                supabase.table("indices").delete().eq("id", r["id"]).execute()
                                st.success("Deleted index (and snapshots). Refresh to update list.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {pretty_error(e)}")
        except Exception as e:
            st.error(f"Failed to load saved indices: {pretty_error(e)}")

# -------------------------
# Account tab
# -------------------------
with tab_account:
    st.header("👤 Account & App Info")
    user = supabase_current_user()
    if not user:
        st.info("Not logged in.")
    else:
        st.json(user)

    st.markdown("---")
    st.write("Notes:")
    st.write("- RLS is expected to be configured on Supabase tables as per SQL in header.")
    st.write("- For scheduled hourly recalcs, implement an Edge Function + Cron trigger using a service_role key (do NOT put service_role in client).")
    st.write("- Consider encrypting any persisted Kite access tokens if you store them in Supabase.")
    st.write("- For heavy usage, switch to batch quote APIs (or instrument token mapping) and implement retry/backoff logic.")

# -------------------------
# End of file
# -------------------------
