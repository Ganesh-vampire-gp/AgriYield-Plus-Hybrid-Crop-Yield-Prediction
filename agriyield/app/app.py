import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="AgriYield+", layout="wide")


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "agriyield" / "models"
DATA_DIR = BASE_DIR / "agriyield" / "data"


PREPROC_PATH = MODELS_DIR / "hybrid_preprocessor.pkl"
XGB_PATH = MODELS_DIR / "hybrid_xgb.pkl"
CAT_PATH = MODELS_DIR / "hybrid_cat.pkl"
LSTM_PATH = MODELS_DIR / "hybrid_lstm.keras"
REC_PATH = MODELS_DIR / "crop_recommender.pkl"
SOIL_LIST_PATH = MODELS_DIR / "soil_types_list.pkl"


DATA_PATH = BASE_DIR / "agriyield" / "data" / "raw" / "season_based_crop.csv"

@st.cache_resource
def load_resources():
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        col_map = {
            "State_Name": "State", "District_Name": "District", "Crop_Year": "Year",
            "Crop_Name": "Crop", "Area": "Area", "Production": "Production",
            "Season": "Season"
        }
        df = df.rename(columns=col_map)
        
        
        for col in ["State", "District", "Season", "Crop"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        
        if "target_yield" not in df.columns and "Production" in df.columns:
            df = df[df["Area"] > 0]
            df["target_yield"] = df["Production"] / df["Area"]
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()

    
    try:
        preprocessor = joblib.load(PREPROC_PATH)
        xgb_model = joblib.load(XGB_PATH)
        cat_model = joblib.load(CAT_PATH)
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
        explainer = shap.TreeExplainer(xgb_model)
    except:
        preprocessor, xgb_model, cat_model, lstm_model, explainer = None, None, None, None, None

    
    try:
        recommender = joblib.load(REC_PATH)
        soil_types = joblib.load(SOIL_LIST_PATH)
    except:
        recommender = None
        soil_types = ["CLAYEY", "LOAMY", "SANDY", "BLACK", "RED"]

    return df, preprocessor, xgb_model, cat_model, lstm_model, explainer, recommender, soil_types


df, preprocessor, xgb_model, cat_model, lstm_model, explainer, recommender, soil_types = load_resources()

if df.empty:
    st.error("Data could not be loaded. Please check 'season_based_crop.csv' exists in data/raw/.")
    st.stop()

st.title("🌱 AgriYield+: Intelligent Agriculture System")


tab1, tab2 = st.tabs(["📊 Crop Yield Prediction", "🌾 Crop Recommendation"])

# TAB 1: YIELD PREDICTION
with tab1:
    st.header("Predict Crop Yield")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        states = sorted(df["State"].unique())
        sel_state = st.selectbox("State", states, key="yield_state")
        
        districts = sorted(df[df["State"] == sel_state]["District"].unique())
        sel_dist = st.selectbox("District", districts, key="yield_dist")
        
        sel_year = st.slider("Year", 2000, 2050, 2024, key="yield_year")
        
        
        if "Season" in df.columns:
            seasons = sorted(df["Season"].unique())
        else:
            seasons = ["KHARIF", "RABI", "WHOLE YEAR"]
        sel_season = st.selectbox("Season", seasons, key="yield_season")

    with col2:
        
        if "Crop" in df.columns:
            crops = sorted(df["Crop"].unique())
        else:
            crops = ["RICE", "MAIZE", "WHEAT"]
        sel_crop = st.selectbox("Crop Type", crops, key="yield_crop")
        
        
        dist_data = df[df["District"] == sel_dist]
        avg_area = dist_data["Area"].mean() if not dist_data.empty else 10.0
        area_inp = st.number_input("Cultivation Area (Acres)", value=float(avg_area) if avg_area > 0 else 10.0)
        
        
        rain_inp = st.number_input("Rainfall (mm)", value=120.0)
        temp_inp = st.number_input("Temperature (°C)", value=25.0)
        soil_inp = st.selectbox("Soil Type", soil_types, key="yield_soil")
        ndvi_inp = st.number_input("NDVI (Vegetation Index)", value=0.21)

    if st.button("Predict Yield", key="btn_yield"):
        if xgb_model is None:
            st.error("Yield models not loaded. Please train them first.")
        else:
            
            hist_yield = df[df["District"] == sel_dist]["target_yield"].mean()
            if pd.isna(hist_yield): hist_yield = 2.0
            
            
            input_df = pd.DataFrame({
                "Year": [sel_year], 
                "State": [sel_state], 
                "District": [sel_dist],
                "Season": [sel_season],
                "Crop": [sel_crop],
                "Area": [area_inp], 
                "yield_calculated": [hist_yield], 
                "rainfall": [rain_inp], 
                "temperature": [temp_inp], 
                "ndvi": [ndvi_inp],
                "Soil_Type": [soil_inp]
            })

            try:
                X_proc = preprocessor.transform(input_df)
                
                
                p_xgb = xgb_model.predict(X_proc)[0]
                p_cat = cat_model.predict(X_proc)[0]
                
                X_lstm = X_proc.reshape((X_proc.shape[0], 1, X_proc.shape[1]))
                if hasattr(X_proc, "toarray"): X_proc = X_proc.toarray() 
                p_lstm = lstm_model.predict(X_lstm, verbose=0).flatten()[0]
                
                final_pred = (p_xgb * 0.4) + (p_cat * 0.4) + (p_lstm * 0.2)
                total_prod = final_pred * area_inp

                st.success(f"Predicted Yield: **{final_pred:.2f} tons/acre**")
                st.info(f"Total Expected Production: **{total_prod:.2f} tons**")
                
                
                # Explainable AI (SHAP)
                st.subheader("Why this prediction?")
                
                
                X_proc_user = preprocessor.transform(input_df)
                
                
                if hasattr(X_proc_user, "toarray"): 
                    X_proc_user = X_proc_user.toarray()
                
                
                shap_vals = explainer.shap_values(X_proc_user)
                
                
                vals = shap_vals[0]
                
                
                try:
                    
                    cat_encoder = preprocessor.named_transformers_["cat"]
                    ohe_features = list(cat_encoder.get_feature_names_out())
                    
                    
                    num_features = ["Year", "Area", "rainfall", "temperature", "ndvi"] 
                    
                    
                    feature_names = ohe_features + num_features
                    
                except Exception as e:
                    
                    feature_names = [f"Feature {i}" for i in range(len(vals))]

                
                if len(feature_names) != len(vals):
                    st.warning(f"Feature name mismatch: Got {len(feature_names)} names for {len(vals)} values. Using generic names.")
                    feature_names = [f"Feature {i}" for i in range(len(vals))]

                impact_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Impact": vals
                })
                
                
                impact_df["Abs_Impact"] = impact_df["Impact"].abs()
                top_features = impact_df.sort_values("Abs_Impact", ascending=False).head(8)
                
                
                top_features["Feature"] = top_features["Feature"].str.replace("cat__", "") \
                                                                .str.replace("State_", "") \
                                                                .str.replace("District_", "") \
                                                                .str.replace("Season_", "") \
                                                                .str.replace("Crop_", "") \
                                                                .str.replace("Soil_Type_", "")

                
                colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in top_features["Impact"]]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(top_features["Feature"], top_features["Impact"], color=colors)
                ax.bar_label(bars, fmt="%.1f", padding=3)
                ax.set_xlabel("Impact on Yield (kg/ha)")
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                plt.gca().invert_yaxis() 
                st.pyplot(fig)
            except Exception as e:
                    st.error(f"Prediction Error: {e}")




    
    st.subheader("📈 Yield Trend")
    trend_df = df[(df["District"] == sel_dist) & (df["Crop"] == sel_crop)].sort_values("Year")
    
    
    with st.expander("🔍 Debug Info"):
        st.write(f"**Selected District:** {sel_dist}")
        st.write(f"**Selected Crop:** {sel_crop}")
        st.write(f"**Matching Records:** {len(trend_df)}")
        st.write(f"**Available Columns:** {list(df.columns)}")
        if not trend_df.empty:
            st.write(f"**Data Preview:**")
            st.dataframe(trend_df[["Year", "District", "Crop", "target_yield"]].head())
    
    if not trend_df.empty:
        st.line_chart(trend_df.set_index("Year")[["target_yield"]])
    else:
        st.warning(f"⚠️ No historical data for **{sel_crop}** in **{sel_dist}**.")


# TAB 2: CROP RECOMMENDATION
with tab2:
    st.header("Find Suitable Crop & Advisory")
    st.markdown("Get expert crop suggestions with fertilizer and care insights.")
    
    if recommender is None:
        st.warning("Recommender model not found. Run 'agriyield/models/train_recommender.py'.")
    else:
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_state = st.selectbox("State", states, key="rec_state")
            r_dists = sorted(df[df["State"] == r_state]["District"].unique())
            r_dist = st.selectbox("District", r_dists, key="rec_dist")
            
        with rc2:
            r_season = st.selectbox("Season", ["KHARIF", "RABI", "WHOLE YEAR", "SUMMER", "WINTER"], key="rec_season")
            r_soil = st.selectbox("Soil Type", soil_types, key="rec_soil")

        with rc3:
            
            r_ph = st.slider("Soil pH Level", 4.0, 9.0, 6.5, help="Optimal pH helps nutrient absorption.")
            r_water = st.select_slider("Water Availability", options=["Low", "Medium", "High", "Abundant"], value="Medium")

        
        if st.button("Recommend Crop", key="btn_rec", type="primary"):
            
            
            rec_input = pd.DataFrame({
                "State": [r_state], 
                "District": [r_dist], 
                "Season": [r_season], 
                "Soil_Type": [r_soil]
            })
            
            try:
                pred_crop = recommender.predict(rec_input)[0]
                probs = recommender.predict_proba(rec_input)[0]
                
                
                st.divider()
                st.subheader(f"🌟 Best Crop: :green[{pred_crop}]")
                
                
                conf_score = max(probs)
                st.progress(conf_score, text=f"Confidence Score: {conf_score*100:.1f}%")

                
                st.markdown("### 💊 Fertilizer & Care Guide")
                
                
                fertilizer_map = {
                    "RICE": "Urea (Nitrogen) & TSP. Maintain standing water level of 5cm.",
                    "COTTON": "Balanced NPK (20-20-20). Requires good drainage.",
                    "MAIZE": "Nitrogen rich fertilizer. Apply Zinc sulphate if leaves yellow.",
                    "WHEAT": "DAP (Di-ammonium Phosphate) during sowing. Irrigate at critical stages.",
                    "GROUNDNUT": "Gypsum/Calcium for pod formation. Avoid excess Nitrogen.",
                    "SUGARCANE": "High Potassium. Heavy irrigation required every 10 days.",
                    "BAJRA": "Low nutrient requirement. Apply Nitrogen in split doses.",
                    "PULSES": "Phosphorus rich fertilizer. No Nitrogen needed (Self-fixing)."
                }
                
                
                advice = fertilizer_map.get(pred_crop.upper(), "Standard NPK (10-26-26) recommended. Monitor for pests.")
                
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    st.info(f"**Fertilizer:** {advice}")
                with col_adv2:
                    
                    if r_ph < 5.5:
                        st.warning(f"**Soil Condition:** Your soil is Acidic (pH {r_ph}). Consider adding Lime.")
                    elif r_ph > 7.5:
                        st.warning(f"**Soil Condition:** Your soil is Alkaline (pH {r_ph}). Consider adding Gypsum.")
                    else:
                        st.success(f"**Soil Condition:** pH {r_ph} is optimal for most crops.")

                
                st.markdown("### 💡 Why this Recommendation?")
                reasons = []
                
                
                if r_soil in ["CLAYEY", "LOAMY"] and pred_crop in ["RICE", "SUGARCANE"]:
                    reasons.append(f"• **{r_soil} Soil** retains moisture well, which is critical for {pred_crop}.")
                elif r_soil == "SANDY" and pred_crop in ["GROUNDNUT", "MAIZE", "BAJRA"]:
                    reasons.append(f"• **{r_soil} Soil** offers good drainage, preventing root rot for {pred_crop}.")
                elif r_soil == "BLACK" and pred_crop in ["COTTON"]:
                    reasons.append(f"• **Black Soil** is famous for Cotton cultivation due to moisture holding.")
                    
                if r_season == "KHARIF":
                    reasons.append(f"• **Kharif Season** (Monsoon) provides the necessary rainfall.")
                elif r_season == "RABI":
                    reasons.append(f"• **Rabi Season** (Winter) offers the cool, dry climate needed.")

                
                if not reasons:
                    reasons.append(f"• Historical farming data in **{r_dist}** shows high success rates for **{pred_crop}**.")
                
                for r in reasons:
                    st.write(r)

                
                st.markdown("### 🔄 Alternative Options")
                
                top3_idx = np.argsort(probs)[-3:][::-1]
                top_crops = [recommender.classes_[i] for i in top3_idx]
                top_probs = [probs[i] for i in top3_idx]
                
                c_chart1, c_chart2 = st.columns([1, 2])
                
                with c_chart1:
                    
                    for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                        st.metric(f"Option {i+1}", crop, f"{prob*100:.1f}%")
                        
                with c_chart2:
                    
                    fig, ax = plt.subplots(figsize=(5, 3))
                    
                    wedges, texts, autotexts = ax.pie(top_probs, labels=top_crops, autopct='%1.1f%%', 
                                                      colors=['#2ecc71', '#3498db', '#95a5a6'], 
                                                      startangle=90, wedgeprops=dict(width=0.4))
                    ax.set_title("Probability Distribution")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
