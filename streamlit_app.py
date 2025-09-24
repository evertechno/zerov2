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
    st.error("Missing Kite credentials in Streamlit secrets. Please configure `kite.api_key`, `kite.api_secret`, and `kite.redirect_uri`.")
    st.stop()

if not (SUPABASE_URL and SUPABASE_KEY):
    st.error("Missing Supabase credentials in Streamlit secrets. Please configure `supabase.url` and `supabase.anon_key`.")
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
    """
    Retrieves the current Supabase user from session state.
    Handles different ways the session or user might be stored.
    """
    user = st.session_state.get("supabase_user")
    if user:
        return user
    
    # Fallback to check if a full session object was stored
    session_response = st.session_state.get("supabase_session")
    if session_response and hasattr(session_response, "user") and session_response.user:
        return session_response.user
    
    return None

def pretty_error(e: Exception) -> str:
    """Formats an exception into a readable string."""
    return str(e)

def fetch_last_price(kite: KiteConnect, symbol: str, exchange_prefix="NSE") -> Optional[float]:
    """Fetches the last traded price for a given symbol."""
    try:
        q = kite.quote(f"{exchange_prefix}:{symbol}")
        key = f"{exchange_prefix}:{symbol}"
        if q and key in q and "last_price" in q[key]:
            return float(q[key]["last_price"])
        return None
    except Exception as e:
        return None

def batch_fetch_prices(kite: KiteConnect, syms: List[str], exchange_prefix="NSE", sleep_between=0.15) -> Dict[str, Optional[float]]:
    """Fetches prices for a batch of symbols with a delay between calls."""
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
if "supabase_user" not in st.session_state:
    st.session_state["supabase_user"] = None

current_supabase_user = supabase_current_user()

if not current_supabase_user:
    st.sidebar.subheader("Supabase Login / Sign up")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    col1, col2 = st.sidebar.columns(2)

    if col1.button("Login"):
        if not email or not password:
            st.sidebar.error("Email and password cannot be empty.")
        else:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                
                if res.user:
                    st.session_state["supabase_session"] = res.session
                    st.session_state["supabase_user"] = res.user
                    st.experimental_rerun()
                else:
                    error_msg = "Unknown error during login."
                    if res.error:
                        error_msg = res.error.message
                    st.sidebar.error(f"Login failed: {error_msg}")
            except Exception as e:
                st.sidebar.error(f"Login failed: {pretty_error(e)}")

    if col2.button("Sign up"):
        if not email or not password:
            st.sidebar.error("Email and password cannot be empty.")
        else:
            try:
                res = supabase.auth.sign_up({"email": email, "password": password})
                
                if res.user:
                    st.sidebar.info("Signup created successfully. Please check your email to confirm your account before logging in.")
                else:
                    error_msg = "Unknown error during signup."
                    if res.error:
                        error_msg = res.error.message
                    st.sidebar.error(f"Signup failed: {error_msg}")
            except Exception as e:
                st.sidebar.error(f"Sign up failed: {pretty_error(e)}")
else:
    user = current_supabase_user
    user_email = user.get("email") if isinstance(user, dict) else (user.email if hasattr(user, 'email') else "N/A")
    st.sidebar.success(f"Signed in: {user_email}")
    if st.sidebar.button("Logout"):
        try:
            supabase.auth.sign_out()
        except Exception as e:
            st.sidebar.warning(f"Logout had a minor issue: {pretty_error(e)}")
        st.session_state.pop("supabase_session", None)
        st.session_state.pop("supabase_user", None)
        st.experimental_rerun()

# -------------------------
# Kite Login
# -------------------------
st.markdown("### Step 1 — Kite Login (Zerodha)")
st.markdown(f"[🔗 Open Kite login]({kite_login_url})", unsafe_allow_html=True)

# --- REPLACEMENT STARTS HERE ---
# Use st.query_params to get query parameters
query_request_token = st.query_params.get("request_token")
request_token = query_request_token # st.query_params directly returns the value, no need for [0]
# --- REPLACEMENT ENDS HERE ---

if request_token and "kite_access_token" not in st.session_state:
    st.info("Exchanging request_token for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if access_token:
            st.session_state["kite_access_token"] = access_token
            st.success("Kite access token saved in session.")
            # --- REPLACEMENT STARTS HERE ---
            # Update query_params to remove request_token
            # st.query_params.update() can be used to set/unset
            if "request_token" in st.query_params:
                del st.query_params["request_token"]
            # --- REPLACEMENT ENDS HERE ---
            st.experimental_rerun()
        else:
            st.error("No access token returned from Kite.")
    except Exception as e:
        st.error(f"Failed to generate Kite session: {pretty_error(e)}. Please try logging into Kite again.")

if "kite_access_token" not in st.session_state:
    st.warning("Login to Kite via the link above to fetch live quotes.")
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
                profile_data = k.profile()
                st.json(profile_data)
            except Exception as e:
                st.error(f"Error fetching profile: {pretty_error(e)}. Your Kite session might have expired.")

        if st.button("📈 Get positions"):
            try:
                positions = k.positions()
                if positions and "net" in positions:
                    st.dataframe(pd.DataFrame(positions["net"]))
                else:
                    st.info("No net positions found.")
            except Exception as e:
                st.error(f"Error fetching positions: {pretty_error(e)}. Your Kite session might have expired.")

    with col2:
        symbol_input = st.text_input("Quick symbol (eg: INFY)", value="INFY", key="quick_sym")
        if st.button("Get quote"):
            if symbol_input:
                price = fetch_last_price(k, symbol_input)
                if price is None:
                    st.warning(f"No price available for {symbol_input}. Check symbol or Kite session.")
                else:
                    st.success(f"{symbol_input} last price: {price:.2f}")
            else:
                st.warning("Please enter a symbol to get a quote.")

# -------------------------
# Create Index
# -------------------------
with tab_create:
    st.header("🧩 Create & Save Custom Index")
    uploaded_file = st.file_uploader("Upload CSV with columns: `symbol`, `Name`, `Weights`", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {"symbol", "Name", "Weights"}
            if not required_columns.issubset(set(df.columns)):
                st.error(f"CSV must contain all required columns: {required_columns}. Found: {set(df.columns)}")
            else:
                df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
                df["Weights"] = pd.to_numeric(df["Weights"], errors="coerce")

                if df["Weights"].isnull().any():
                    st.error("Some values in the 'Weights' column could not be parsed as numbers. Please check your CSV.")
                elif df["Weights"].sum() <= 0:
                    st.error("The sum of weights must be greater than zero.")
                else:
                    df["Weights"] = df["Weights"] / df["Weights"].sum() # Normalize weights
                    
                    st.info("Fetching live prices for index components...")
                    symbols_to_fetch = df["symbol"].tolist()
                    prices = batch_fetch_prices(k, symbols_to_fetch, sleep_between=0.12)
                    df["Last Price"] = df["symbol"].map(prices)

                    missing_prices_symbols = df[df["Last Price"].isnull()]["symbol"].tolist()
                    if missing_prices_symbols:
                        st.warning(f"Could not fetch live prices for: {', '.join(missing_prices_symbols)}. These components will not contribute to the index value.")
                    
                    df["Weighted Price"] = df["Last Price"] * df["Weights"]
                    index_value = float(df["Weighted Price"].sum(skipna=True)) # sum(skipna=True) handles None/NaN in Last Price

                    st.dataframe(df.style.format({"Weights": "{:.2%}", "Last Price": "{:,.2f}", "Weighted Price": "{:,.2f}"}))
                    st.markdown(f"### Current Index Value: **{index_value:,.4f}**")

                    user = supabase_current_user()
                    if user:
                        idx_name = st.text_input("Index name", value="My Custom Index", key="new_index_name")
                        if st.button("💾 Save Index & Snapshot", key="save_new_index"):
                            if not idx_name:
                                st.error("Index name cannot be empty.")
                            else:
                                try:
                                    symbols_for_db = df[["symbol", "Name", "Weights"]].to_dict(orient="records")
                                    
                                    # Insert index definition
                                    idx_res = supabase.table("indices").insert({
                                        "user_id": user.get("id"),
                                        "name": idx_name,
                                        "symbols": json.dumps(symbols_for_db),
                                        "last_value": index_value
                                    }).execute()

                                    index_id = None
                                    if idx_res.data and len(idx_res.data) > 0:
                                        index_id = idx_res.data[0]["id"]
                                    
                                    if index_id:
                                        calc_details = df[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                        # Insert initial calculation snapshot
                                        supabase.table("index_calculations").insert({
                                            "index_id": index_id,
                                            "value": index_value,
                                            "details": json.dumps(calc_details)
                                        }).execute()
                                        st.success(f"Index '{idx_name}' and initial snapshot saved successfully!")
                                        st.experimental_rerun()
                                    else:
                                        st.error("Failed to retrieve index ID after saving definition. Snapshot not saved.")
                                except Exception as e:
                                    st.error(f"Failed to save index: {pretty_error(e)}")
                    else:
                        st.warning("Login to Supabase to save custom indices.")
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {pretty_error(e)}. Ensure it's a valid CSV.")

# -------------------------
# Saved Indices
# -------------------------
with tab_saved:
    st.header("💾 Saved Indices")
    user = supabase_current_user()
    if user:
        try:
            # Fetch indices for the current user
            resp = supabase.table("indices").select("*").eq("user_id", user.get("id")).order("created_at", desc=True).execute()
            rows = resp.data # The actual data is in .data attribute of the response object

            if not rows:
                st.info("No saved indices yet. Go to 'Create Index' tab to create one!")
            else:
                for r in rows:
                    # Display each index in an expander
                    last_value_display = f"{r.get('last_value', 'N/A'):,.4f}" if isinstance(r.get('last_value'), (int, float)) else "N/A"
                    updated_at_display = r.get('updated_at', r['created_at'])
                    if updated_at_display:
                        updated_at_display = updated_at_display.split('.')[0].replace('T', ' ')
                    else:
                        updated_at_display = "Unknown"

                    with st.expander(f"**{r['name']}** — Current Value: {last_value_display} (Last updated: {updated_at_display})"):
                        st.subheader("Index Composition:")
                        try:
                            symbols_data = json.loads(r["symbols"])
                            symbols_df = pd.DataFrame(symbols_data)
                            if not symbols_df.empty:
                                symbols_df["Weights"] = symbols_df["Weights"].apply(lambda x: f"{x:.2%}")
                                st.dataframe(symbols_df.style.format())
                            else:
                                st.info("No symbols defined for this index.")
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for index symbols. Data might be corrupted.")
                            st.text(r["symbols"])
                        
                        st.markdown("---")
                        colv1, colv2, colv3 = st.columns([1, 1, 1])

                        # View History Button
                        with colv1:
                            if st.button("View History", key=f"hist_{r['id']}"):
                                try:
                                    calcs_resp = supabase.table("index_calculations").select("*").eq("index_id", r["id"]).order("calculated_at", desc=True).limit(50).execute()
                                    calcs = calcs_resp.data
                                    if calcs:
                                        st.subheader("Calculation History")
                                        for c in calcs:
                                            calc_time_display = c['calculated_at'].split('.')[0].replace('T', ' ')
                                            with st.expander(f"Snapshot Value: {c['value']:,.4f} at {calc_time_display}"):
                                                try:
                                                    details_df = pd.DataFrame(json.loads(c.get("details", "[]")))
                                                    st.dataframe(details_df.style.format({"Last Price": "{:,.2f}", "Weights": "{:.2%}", "Weighted Price": "{:,.2f}"}))
                                                except json.JSONDecodeError:
                                                    st.error("Invalid JSON format for calculation details.")
                                                    st.text(c.get("details", "{}"))
                                    else:
                                        st.info("No calculation history found for this index.")
                                except Exception as e:
                                    st.error(f"Failed to fetch history: {pretty_error(e)}")

                        # Recalculate Now Button
                        with colv2:
                            if st.button("Recalculate now (live)", key=f"recalc_{r['id']}"):
                                try:
                                    st.info("Recalculating live value for index...")
                                    symbols_raw = json.loads(r["symbols"])
                                    syms_for_recalc = [s["symbol"] for s in symbols_raw]
                                    
                                    prices_recalc = batch_fetch_prices(k, syms_for_recalc, sleep_between=0.12)
                                    
                                    rows_df_recalc = pd.DataFrame(symbols_raw)
                                    rows_df_recalc["Last Price"] = rows_df_recalc["symbol"].map(prices_recalc)

                                    missing_prices_recalc = rows_df_recalc[rows_df_recalc["Last Price"].isnull()]["symbol"].tolist()
                                    if missing_prices_recalc:
                                        st.warning(f"Could not fetch live prices for {', '.join(missing_prices_recalc)} during recalculation. These will not contribute.")
                                    
                                    rows_df_recalc["Weighted Price"] = rows_df_recalc["Last Price"] * rows_df_recalc["Weights"]
                                    new_index_value = float(rows_df_recalc["Weighted Price"].sum(skipna=True))
                                    
                                    calc_details_recalc = rows_df_recalc[["symbol", "Last Price", "Weights", "Weighted Price"]].to_dict(orient="records")
                                    
                                    # Insert new calculation snapshot
                                    supabase.table("index_calculations").insert({
                                        "index_id": r["id"],
                                        "value": new_index_value,
                                        "details": json.dumps(calc_details_recalc)
                                    }).execute()
                                    
                                    # Update last_value and updated_at in the main index table
                                    supabase.table("indices").update({"last_value": new_index_value, "updated_at": "now()"}).eq("id", r["id"]).execute()
                                    
                                    st.success(f"Recalculated & saved snapshot — New Index Value: {new_index_value:,.4f}")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Recalculation failed: {pretty_error(e)}")

                        # Delete Index Button
                        with colv3:
                            if st.button("Delete Index", key=f"del_{r['id']}"):
                                try:
                                    st.warning("Deleting index and all its historical snapshots...")
                                    supabase.table("index_calculations").delete().eq("index_id", r["id"]).execute()
                                    supabase.table("indices").delete().eq("id", r["id"]).execute()
                                    st.success(f"Index '{r['name']}' and all its snapshots deleted.")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete index: {pretty_error(e)}. Check RLS policies.")
        except Exception as e:
            st.error(f"Failed to fetch saved indices: {pretty_error(e)}. Ensure Supabase RLS policies allow read access for `indices` and `index_calculations` tables for the logged-in user.")
    else:
        st.info("Login to Supabase to view your saved indices.")

# -------------------------
# Account Tab
# -------------------------
with tab_account:
    st.header("👤 Account & App Info")
    user = supabase_current_user()
    if user:
        st.subheader("Supabase User Info")
        if isinstance(user, dict):
            st.json(user)
        elif hasattr(user, '__dict__'):
            st.json(user.__dict__)
        else:
            st.write(str(user))
    else:
        st.info("Not logged in to Supabase.")
    
    st.markdown("---")
    st.subheader("Important Notes & Considerations")
    st.markdown("""
    - **Supabase Row Level Security (RLS):** It is **critical** to set up RLS policies on your `indices` and `index_calculations` tables in Supabase. This ensures users can only access their own data. Example policy for `indices` table (for `SELECT`): `(user_id = auth.uid())`. Apply similar policies for `INSERT`, `UPDATE`, `DELETE`.
    - **Kite Access Token Security:** The Kite access token is currently stored in `st.session_state` (in-memory). If you need to persist it (e.g., across sessions or in your database), it **must be encrypted at rest** using strong encryption methods.
    - **API Usage Limits:** Zerodha KiteConnect has API rate limits. For heavy usage (many indices, frequent recalculations), consider their batch quote APIs (if applicable) and implement robust retry/backoff logic to avoid hitting limits. The current `sleep_between` in `batch_fetch_prices` is a simple mitigation.
    - **Error Handling:** While improved, comprehensive error handling often involves logging, user-friendly messages for specific error codes, and mechanisms to alert administrators.
    - **Data Consistency:** The `updated_at` field in your `indices` table should ideally be a `timestamp with time zone` with a default value of `now()` and an `ON UPDATE` trigger to `now()` in your Supabase database definition for automatic updates.
    - **Index Value Precision:** The display format `:.4f` might be adjusted based on desired precision for financial values.
    - **User Experience:** Consider adding loading spinners (`st.spinner`) during long-running operations like `batch_fetch_prices`.
    """)
