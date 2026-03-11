import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Group 6 | House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { background-color: #f0f4f8; color: #1A1A1A; }
    h1, h2, h3, h4, p, span, div, label { color: #1A1A1A; }
    .stMarkdown p, .stMarkdown h3 { color: #1A1A1A !important; }
    .stNumberInput label, .stSelectbox label, .stSlider label { color: #1A1A1A !important; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { color: #1A1A1A !important; font-size: 1rem; font-weight: 600; }
    .header-box {
        background: linear-gradient(135deg, #0A2342 0%, #1565C0 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-bottom: 5px solid #0097A7;
    }
    .header-box h1 { color: white; margin: 0; font-size: 2.2rem; }
    .header-box p  { color: #90CAF9; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #0097A7;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 800; color: #0A2342; }
    .metric-card .label { font-size: 0.78rem; color: #607D8B; text-transform: uppercase; letter-spacing: 0.05em; }
    .result-box {
        background: linear-gradient(135deg, #0A2342, #1565C0);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-box .price { font-size: 3rem; font-weight: 900; color: #F9A825; }
    .result-box .sub   { font-size: 0.9rem; color: #90CAF9; margin-top: 0.3rem; }
    .info-row {
        background: white;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .section-header {
        background: #0A2342;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        margin: 1rem 0 0.5rem 0;
    }
    .stTabs [aria-selected="true"] { color: #0097A7 !important; border-bottom-color: #0097A7 !important; }
    .stCaption { color: #546E7A !important; }
    .stForm { color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🏠 House Price Predictor</h1>
    <p>Group 6 &nbsp;|&nbsp; Ames, Iowa Housing Dataset &nbsp;|&nbsp; XGBoost Model &nbsp;|&nbsp; RMSLE: 0.1108 &nbsp;|&nbsp; R²: 93.1%</p>
</div>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("xgb_model.pkl")
    encoders     = joblib.load("label_encoders.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, encoders, feature_cols

@st.cache_data
def load_test_data():
    return pd.read_csv("test_clean.csv")

try:
    model, label_encoders, feature_cols = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"⚠️ Model files not found: {e}\n\nMake sure xgb_model.pkl, label_encoders.pkl, feature_cols.pkl are in the same folder.")
    artifacts_loaded = False
    st.stop()

try:
    test_clean = load_test_data()
    test_loaded = True
except:
    test_loaded = False

# ── Stat cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="value">2,919</div><div class="label">Total Houses</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="value">81</div><div class="label">Features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="value">0.1108</div><div class="label">Best RMSLE</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="value">93.1%</div><div class="label">R² Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Predict by House ID", "✏️  Predict by Features"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — House ID Lookup
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter a House ID from the test set (1461 – 2919)")

    col_input, col_btn = st.columns([2, 1])
    with col_input:
        house_id = st.number_input("House ID", min_value=1461, max_value=2919, value=1461, step=1, label_visibility="collapsed")
    with col_btn:
        predict_id = st.button("🔮 Predict Price", use_container_width=True, key="btn_id")

    if predict_id:
        if not test_loaded:
            st.error("test_clean.csv not found. Please add it to the repository.")
        else:
            match = test_clean[test_clean["Id"] == house_id].copy()
            if match.empty:
                st.error(f"House ID {house_id} not found in the test set.")
            else:
                row_orig = match.iloc[0].copy()

                # Encode categoricals
                for col, le in label_encoders.items():
                    if col in match.columns:
                        try:
                            match[col] = le.transform(match[col].astype(str))
                        except:
                            match[col] = 0

                # Ensure all feature cols present
                for col in feature_cols:
                    if col not in match.columns:
                        match[col] = 0

                X_pred = match[feature_cols]
                log_pred = model.predict(X_pred)[0]
                price = np.expm1(log_pred)

                # Result
                left, right = st.columns([1, 1])

                with left:
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size:1rem; color:#90CAF9; margin-bottom:0.3rem;">🏠 HOUSE ID {house_id}</div>
                        <div class="price">${price:,.0f}</div>
                        <div class="sub">Predicted Sale Price</div>
                    </div>
                    """, unsafe_allow_html=True)

                with right:
                    st.markdown('<div class="section-header">📋 House Details</div>', unsafe_allow_html=True)
                    details = {
                        "Neighborhood":  row_orig.get("Neighborhood", "N/A"),
                        "Overall Quality": f"{int(row_orig.get('OverallQual', 0))} / 10",
                        "Total SF":       f"{int(row_orig.get('TotalSF', 0)):,} sq ft",
                        "Year Built":     int(row_orig.get("YearBuilt", 0)),
                        "Total Bathrooms": row_orig.get("TotalBath", 0),
                        "Garage Cars":    int(row_orig.get("GarageCars", 0)),
                        "House Age":      f"{int(row_orig.get('HouseAge', 0))} years",
                    }
                    for k, v in details.items():
                        st.markdown(f"""
                        <div class="info-row">
                            <span style="color:#546E7A; font-weight:600">{k}</span>
                            <span style="font-weight:700; color:#0A2342">{v}</span>
                        </div>
                        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual Feature Input
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Enter house features manually")
    st.caption("Fill in the key characteristics of any house to get a price estimate.")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏗 Size**")
            gr_liv_area   = st.number_input("Above-Ground Living Area (sq ft)", 400, 6000, 1500)
            total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
            first_flr_sf  = st.number_input("1st Floor (sq ft)", 400, 4000, 1000)
            second_flr_sf = st.number_input("2nd Floor (sq ft)", 0, 2000, 0)
            lot_area      = st.number_input("Lot Area (sq ft)", 1000, 220000, 9000)
            garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, 480)

        with col2:
            st.markdown("**⭐ Quality & Condition**")
            overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
            overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)
            kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
            exter_qual   = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
            bsmt_qual    = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"])
            fireplaces   = st.number_input("Number of Fireplaces", 0, 4, 0)

        with col3:
            st.markdown("**📅 Time & Location**")
            year_built     = st.number_input("Year Built", 1870, 2010, 1990)
            year_remod     = st.number_input("Year Remodeled", 1950, 2010, 1990)
            yr_sold        = st.number_input("Year Sold", 2006, 2010, 2008)
            neighborhood   = st.selectbox("Neighborhood", [
                "NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt","Gilbert",
                "Sawyer","NWAmes","SawyerW","BrkSide","Crawfor","Mitchel","NoRidge",
                "Timber","IDOTRR","ClearCr","StoneBr","SWISU","Blmngtn","MeadowV",
                "BrDale","Veenker","NPkVill","Blueste"
            ])
            garage_cars    = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
            full_bath      = st.number_input("Full Bathrooms (above grade)", 0, 4, 2)
            half_bath      = st.number_input("Half Bathrooms (above grade)", 0, 2, 0)
            bsmt_full_bath = st.number_input("Basement Full Bathrooms", 0, 3, 0)
            bsmt_half_bath = st.number_input("Basement Half Bathrooms", 0, 2, 0)

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        # Engineered features
        total_sf           = total_bsmt_sf + first_flr_sf + second_flr_sf
        total_bath         = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
        house_age          = yr_sold - year_built
        years_since_remod  = yr_sold - year_remod
        was_remodeled      = 1 if year_remod != year_built else 0
        is_new             = 1 if yr_sold == year_built else 0

        # Build a row with all feature_cols set to 0 first
        row = {col: 0 for col in feature_cols}

        # Fill numerical values
        numerical_vals = {
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "LotArea": lot_area,
            "GarageArea": garage_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "Fireplaces": fireplaces,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "YrSold": yr_sold,
            "GarageCars": garage_cars,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BsmtFullBath": bsmt_full_bath,
            "BsmtHalfBath": bsmt_half_bath,
            "TotalSF": total_sf,
            "TotalBath": total_bath,
            "HouseAge": house_age,
            "YearsSinceRemodel": years_since_remod,
            "WasRemodeled": was_remodeled,
            "IsNew": is_new,
        }
        for k, v in numerical_vals.items():
            if k in row:
                row[k] = v

        # Encode categorical columns
        categorical_vals = {
            "Neighborhood": neighborhood,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "BsmtQual": bsmt_qual,
        }
        for col, val in categorical_vals.items():
            if col in label_encoders and col in row:
                try:
                    row[col] = label_encoders[col].transform([str(val)])[0]
                except:
                    row[col] = 0

        X_manual = pd.DataFrame([row])[feature_cols]
        log_pred = model.predict(X_manual)[0]
        price    = np.expm1(log_pred)

        # Confidence range (±1 MAE)
        low  = price - 14218
        high = price + 14218

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:1rem; color:#90CAF9; margin-bottom:0.3rem;">✏️ MANUAL PREDICTION</div>
            <div class="price">${price:,.0f}</div>
            <div class="sub">Estimated Sale Price &nbsp;|&nbsp; Typical range: ${max(0,low):,.0f} – ${high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">📐 Engineered Features Used</div>', unsafe_allow_html=True)
            eng = {
                "TotalSF": f"{total_sf:,} sq ft",
                "TotalBath": total_bath,
                "HouseAge": f"{house_age} years",
                "YearsSinceRemodel": f"{years_since_remod} years",
                "WasRemodeled": "Yes" if was_remodeled else "No",
                "IsNew": "Yes" if is_new else "No",
            }
            for k, v in eng.items():
                st.markdown(f"""
                <div class="info-row">
                    <span style="color:#546E7A; font-weight:600">{k}</span>
                    <span style="font-weight:700; color:#0097A7">{v}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-header">🔑 Top Factors in This Prediction</div>', unsafe_allow_html=True)
            factors = [
                ("Overall Quality", f"{overall_qual}/10", overall_qual >= 7),
                ("Total SF", f"{total_sf:,} sq ft", total_sf >= 2000),
                ("Neighborhood", neighborhood, neighborhood in ["NoRidge","NridgHt","StoneBr","Timber","Veenker"]),
                ("Garage Capacity", f"{garage_cars} car(s)", garage_cars >= 2),
                ("Kitchen Quality", kitchen_qual, kitchen_qual in ["Ex","Gd"]),
            ]
            for name, val, positive in factors:
                icon = "🟢" if positive else "🔵"
                st.markdown(f"""
                <div class="info-row">
                    <span style="color:#546E7A; font-weight:600">{icon} {name}</span>
                    <span style="font-weight:700; color:#0A2342">{val}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#90A4AE; font-size:0.8rem;">
    Group 6 &nbsp;|&nbsp; House Prices: Advanced Regression Techniques &nbsp;|&nbsp;
    XGBoost · Scikit-learn · Python &nbsp;|&nbsp; Kaggle Competition
</div>
""", unsafe_allow_html=True)
