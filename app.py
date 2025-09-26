# app.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import traceback

# --------------------------
# CONFIG
# --------------------------
OUTPUTS_DIR = "outputs"
SALES_PATH = "sales.csv"       # adjust if needed
INV_PATH = "Inventary.csv"     # adjust if needed

st.set_page_config(page_title="Quick-Commerce Forecasting", layout="wide")
st.title("📦 Quick-Commerce — Demand Forecast & Alerts")

# --------------------------
# Load Models & Artifacts
# --------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load {path}: {e}")
        return None

lgb_model = load_model(os.path.join(OUTPUTS_DIR, "model_lgb_final.joblib"))
cat_model = load_model(os.path.join(OUTPUTS_DIR, "model_cat_final.joblib"))
meta_model = load_model(os.path.join(OUTPUTS_DIR, "meta_learner.joblib"))

# Load feature list
feature_list_path = os.path.join(OUTPUTS_DIR, "feature_list.json")
feature_list = None
if os.path.exists(feature_list_path):
    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)
else:
    st.error("❌ feature_list.json not found. Please run pipeline first.")
    st.stop()

st.sidebar.header("Models Loaded")
st.sidebar.write({
    "LightGBM": bool(lgb_model),
    "CatBoost": bool(cat_model),
    "Meta Learner": bool(meta_model),
    "Features": len(feature_list) if feature_list else 0
})

# --------------------------
# Utilities
# --------------------------
def align_features(df_in, feature_list):
    """Aligns columns with feature_list, fills missing with 0"""
    df = df_in.copy()
    for f in feature_list:
        if f not in df.columns:
            df[f] = 0.0
    # keep only feature_list order if possible
    cols = [c for c in feature_list if c in df.columns]
    if cols:
        df = df[cols]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def predict_with_models(X):
    preds = {}
    if lgb_model is not None:
        try:
            preds["lgb"] = lgb_model.predict(X)
        except Exception as e:
            st.error(f"LightGBM failed: {e}")

    if cat_model is not None:
        try:
            preds["cat"] = cat_model.predict(X)
        except Exception as e:
            st.error(f"CatBoost failed: {e}")

    final = None
    if meta_model is not None and preds:
        try:
            meta_X = pd.DataFrame(preds)
            final = meta_model.predict(meta_X)
        except Exception as e:
            st.error(f"Meta learner failed: {e}")
    elif preds:
        try:
            arrs = np.vstack(list(preds.values()))
            final = np.mean(arrs, axis=0)
        except Exception as e:
            st.error(f"Ensembling failed: {e}")

    return preds, final

# --------------------------
# Groq LLM helper
# --------------------------
# Requires: pip install groq
try:
    from groq import Groq
    groq_available = True
except Exception:
    Groq = None
    groq_available = False

EXPLAIN_PROMPT = """
You are an assistant that explains demand-forecast outputs to a supply planner.
Given:
- City: {city}
- Product: {product}
- Predicted units (next 7 days): {pred}
- Model contributions:
{contributions}

Recent sales (last rows):
{recent_sales}

Please:
1. Give a short plain-language summary (2-3 lines).
2. Explain why the model might predict that amount (point out contribution magnitudes).
3. Suggest 3 practical actions (e.g., allocate from warehouse X, reorder suggestions, safety stock).
Use bullet points and keep it concise.
"""

# Improved Groq helper (paste into app.py replacing previous groq helper)
@st.cache_resource
def get_groq_client():
    try:
        from groq import Groq
    except Exception:
        return None, "groq package not installed. pip install groq"
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY not set in environment."
    try:
        return Groq(api_key=api_key), None
    except Exception as e:
        return None, f"Failed to init Groq client: {e}"

def explain_with_groq_api(prompt_text, model="mixtral-8x7b-32768", max_tokens=300, temperature=0.2):
    client, err = get_groq_client()
    if err:
        return None, None, f"init error: {err}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return None, None, f"API call failed: {e}"

    # robust parsing
    raw = resp
    try:
        # Common shape: resp.choices[0].message.content
        text = None
        if hasattr(resp, "choices"):
            choice0 = resp.choices[0]
            if hasattr(choice0, "message") and hasattr(choice0.message, "content"):
                text = choice0.message.content
            elif isinstance(choice0, dict):
                # dict shaped
                msg = choice0.get("message", {})
                if isinstance(msg, dict):
                    text = msg.get("content") or msg.get("text") or None
        # fallback: resp.get('output') or resp.get('text')
        if text is None and isinstance(resp, dict):
            text = resp.get("output") or resp.get("text") or resp.get("generated_text")
        # final fallback: stringify
        if text is None:
            text = str(resp)
    except Exception as e:
        return None, raw, f"Parsing response failed: {e}"

    return text, raw, None

# --------------------------
# Input Mode
# --------------------------
st.header("Prediction Mode")
mode = st.radio("Choose mode:", ["Upload CSV", "Custom Input (City + Product)"])

# Sidebar: model info & quick instructions
st.sidebar.markdown("**Explain model**: Groq")
st.sidebar.markdown("Set `GROQ_API_KEY` env var and install `groq` package.")
st.sidebar.text("Model used: mixtral-8x7b-32768 (change in code if needed)")

df_proc = None
city = None
product = None

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df_proc = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_proc)} rows")
        st.dataframe(df_proc.head(10))

elif mode == "Custom Input (City + Product)":
    try:
        sales = pd.read_csv(SALES_PATH)
        # normalize product id
        if "product_id" in sales.columns:
            sales["product_id"] = sales["product_id"].astype(str)

        cities = sales["city_name"].dropna().unique().tolist() if "city_name" in sales.columns else []
        products = sales["product_id"].dropna().unique().tolist() if "product_id" in sales.columns else []

        city = st.selectbox("Select City", sorted(cities))
        product = st.selectbox("Select Product", sorted(products))

        df_latest = sales[(sales["city_name"] == city) & (sales["product_id"] == product)].copy()
        df_latest = df_latest.sort_values("date") if "date" in df_latest.columns else df_latest

        if df_latest.empty:
            st.error("No data found for this city/product.")
        else:
            st.success(f"Found {len(df_latest)} records for product {product} in {city}")
            st.dataframe(df_latest.tail(10))
            df_proc = df_latest.copy()
    except Exception as e:
        st.error(f"Failed to load sales data: {e}")

# --------------------------
# Run Prediction
# --------------------------
if df_proc is not None:
    X = align_features(df_proc, feature_list)

    if st.button("Run Prediction"):
        if mode == "Custom Input (City + Product)":
            # Use only the latest row for forecast
            X_latest = X.tail(1)
            base_preds, final_pred = predict_with_models(X_latest)

            if final_pred is not None:
                units_needed = float(final_pred[0])
                st.success(f"📦 Predicted demand for next 7 days: **{units_needed:.0f} units**")

                # Show breakdown
                st.write("🔎 Model contributions:")
                for k, v in base_preds.items():
                    st.write(f"- {k}: {v[0]:.2f}")

                # Download result
                result = pd.DataFrame({
                    "city": [city],
                    "product": [product],
                    "predicted_units_next_7d": [units_needed]
                })
                st.download_button(
                    "Download Prediction",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                # Allocate placeholder (replace with inventory allocation logic if you want)
                if st.button("Allocate"):
                    st.info("Allocation logic goes here. (Implement your allocation function to use INV_PATH)")

                # Explain with Groq
                if st.button("Explain with Groq"):
                    with st.spinner("Generating explanation with Groq..."):
                        contributions = {k: (v[0] if hasattr(v, "__len__") else float(v)) for k, v in base_preds.items()}
                        # pick recent rows for context (best-effort)
                        recent = df_proc[[c for c in df_proc.columns if c in ["date","qty","sales"]]].tail(10) if any([c in df_proc.columns for c in ["qty","sales","date"]]) else df_proc.tail(10)
                        explanation = explain_prediction_with_llm(city, product, units_needed, contributions, recent_sales_df=recent)
                        # show explanation + prompt in expanders for debugging/visibility
                        with st.expander("Grok Explanation", expanded=True):
                            st.write(explanation)
                        # also show the prompt for debugging
                        with st.expander("Prompt sent to Groq (debug)", expanded=False):
                            try:
                                contrib_lines = "\n".join([f"- {k}: {float(v):.2f}" for k, v in contributions.items()])
                            except Exception:
                                contrib_lines = str(contributions)
                            try:
                                prompt = EXPLAIN_PROMPT.format(city=city, product=product, pred=float(units_needed), contributions=contrib_lines, recent_sales=recent.tail(8).to_csv(index=False))
                            except Exception:
                                prompt = "Could not build prompt for display."
                            st.code(prompt)

            else:
                st.error("❌ No prediction generated.")
        else:
            # Batch prediction on CSV
            base_preds, final_pred = predict_with_models(X)
            if final_pred is not None:
                df_result = df_proc.copy()
                for k, v in base_preds.items():
                    df_result[f"pred_{k}"] = v
                df_result["pred_final"] = final_pred

                st.success("✅ Predictions generated")
                st.dataframe(df_result.head(20))

                st.download_button(
                    "Download Predictions",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                # Allocate for batch placeholder
                if st.button("Allocate for Batch"):
                    st.info("Batch allocation logic goes here.")

                # Explain selected row in batch
                if not df_result.empty:
                    idx_to_explain = st.number_input("Row index to explain (batch)", min_value=0, max_value=max(0, len(df_result)-1), value=0, step=1)
                    if st.button("Explain selected row with Groq"):
                        row = df_result.iloc[int(idx_to_explain)]
                        city_b = row.get("city_name") or row.get("city")
                        product_b = row.get("product_id") or row.get("product")
                        pred_b = row.get("pred_final")
                        contributions = {}
                        for ccol in df_result.columns:
                            if ccol.startswith("pred_") and ccol != "pred_final":
                                contributions[ccol] = row.get(ccol, 0)
                        if not contributions:
                            contributions = {"pred_final": pred_b}
                        with st.spinner("Generating explanation with Groq..."):
                            recent = df_proc[df_proc.get("city_name", df_proc.columns[0])==city_b].tail(10) if "date" in df_proc.columns else df_proc.tail(10)
                            explanation = explain_prediction_with_llm(city_b, product_b, pred_b, contributions, recent_sales_df=recent)
                            with st.expander(f"Grok Explanation — row {idx_to_explain}", expanded=True):
                                st.write(explanation)
            else:
                st.error("❌ No predictions generated.")
