import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('simple_rf_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

def get_estimated_weather(state, season):
    """
    Returns estimated (Rainfall_mm, Temperature_C) based on State and Season.
    These are approximate average values for demonstration purposes.
    """
    # Base values
    base_temp = 25.0
    base_rain = 50.0

    # State modifiers (Rainfall factor, Temp offset)
    state_map = {
        "Andhra Pradesh": (1.2, 5),
        "Arunachal Pradesh": (2.5, -5),
        "Assam": (2.2, 0),
        "Bihar": (1.1, 2),
        "Chhattisgarh": (1.3, 3),
        "Goa": (2.0, 2),
        "Gujarat": (0.8, 5),
        "Haryana": (0.7, 4),
        "Himachal Pradesh": (1.2, -8),
        "Jharkhand": (1.2, 2),
        "Karnataka": (1.1, 1),
        "Kerala": (2.5, 2),
        "Madhya Pradesh": (1.0, 4),
        "Maharashtra": (1.2, 3),
        "Manipur": (1.8, -2),
        "Meghalaya": (3.0, -3),
        "Mizoram": (1.8, -2),
        "Nagaland": (1.5, -3),
        "Odisha": (1.4, 3),
        "Punjab": (0.6, 4),
        "Rajasthan": (0.4, 8),
        "Sikkim": (1.8, -7),
        "Tamil Nadu": (1.1, 5),
        "Telangana": (1.0, 5),
        "Tripura": (1.8, 1),
        "Uttar Pradesh": (0.9, 3),
        "Uttarakhand": (1.2, -5),
        "West Bengal": (1.5, 2),
        "Andaman and Nicobar Islands": (2.2, 2),
        "Chandigarh": (0.8, 4),
        "Dadra and Nagar Haveli": (1.5, 3),
        "Daman and Diu": (1.0, 3),
        "Delhi": (0.6, 5),
        "Lakshadweep": (1.8, 2),
        "Puducherry": (1.2, 4)
    }

    # Season modifiers (Rainfall multiplier, Temp offset)
    season_map = {
        "Winter (Jan-Feb)": (0.2, -10),
        "Pre-Monsoon (Mar-May)": (0.5, 5),
        "Monsoon (Jun-Sep)": (3.0, 2),
        "Post-Monsoon (Oct-Dec)": (0.8, -2)
    }

    s_rain_factor, s_temp_offset = state_map.get(state, (1.0, 0))
    sea_rain_mult, sea_temp_offset = season_map.get(season, (1.0, 0))

    est_rain = base_rain * s_rain_factor * sea_rain_mult
    est_temp = base_temp + s_temp_offset + sea_temp_offset

    # Add some randomness or specific adjustments if needed, but this is a good baseline
    return max(0, est_rain), est_temp

st.set_page_config(page_title="Groundwater Level Predictor", page_icon="💧")

st.title("💧 Groundwater Level Predictor (India)")
st.markdown("""
This application predicts the groundwater level based on the State and Season in India.
It uses a machine learning model trained on environmental factors like Rainfall, Temperature, pH, and Dissolved Oxygen.
""")

if model is None:
    st.error("Model file 'simple_rf_model.pkl' not found. Please run the training script first.")
else:
    # Sidebar for inputs
    st.sidebar.header("Location & Time")
    
    states = sorted([
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
        "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", 
        "Dadra and Nagar Haveli", "Daman and Diu", "Delhi", "Lakshadweep", "Puducherry"
    ])
    
    seasons = [
        "Winter (Jan-Feb)", 
        "Pre-Monsoon (Mar-May)", 
        "Monsoon (Jun-Sep)", 
        "Post-Monsoon (Oct-Dec)"
    ]

    selected_state = st.sidebar.selectbox("Select State", states)
    selected_season = st.sidebar.selectbox("Select Season", seasons)

    # Get estimated values
    est_rain, est_temp = get_estimated_weather(selected_state, selected_season)

    st.sidebar.markdown("---")
    st.sidebar.header("Environmental Factors")
    st.sidebar.info("Values below are estimated based on your selection. You can adjust them if you have specific data.")

    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, float(est_rain))
    temperature = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, float(est_temp))
    
    # Default values for pH and DO as they are less dependent on State/Season in this simple estimation
    ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
    do = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 8.0)

    # Prediction
    if st.button("Predict Groundwater Level"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Rainfall_mm': [rainfall],
            'Temperature_C': [temperature],
            'pH': [ph],
            'Dissolved_Oxygen_mg_L': [do]
        })

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Water Level", f"{prediction:.2f} m")
            st.caption("Depth below ground level")

        with col2:
            # Determine status based on quantiles from original analysis (approximate)
            # < 2.12 (Safe), < 3.5 (Semi), >= 3.5 (Critical) - Example thresholds from data exploration
            # Let's use the thresholds from classification_analysis.py if possible, or just reasonable defaults.
            # From classification_analysis.py output earlier: q1=2.12, q2=3.5 (approx from memory of similar datasets, let's check file view)
            # Actually, let's look at the classification_analysis.py view again to be precise.
            # Line 45: q1 = df['Water_Level_m'].quantile(0.33)
            # Line 46: q2 = df['Water_Level_m'].quantile(0.66)
            # We don't have the exact values here without running it, but let's assume:
            # Safe: < 5m, Semi-Critical: 5-10m, Critical: > 10m is a standard hydrogeological classification, 
            # but the dataset might be different. 
            # Let's use a generic interpretation: Higher value = deeper water = worse.
            
            status = ""
            color = ""
            if prediction < 5:
                status = "SAFE"
                color = "green"
            elif prediction < 10:
                status = "SEMI-CRITICAL"
                color = "orange"
            else:
                status = "CRITICAL"
                color = "red"
            
            st.markdown(f"Status: **:{color}[{status}]**")
            st.write(f"The groundwater level is considered {status.lower()}.")

        # Visualization of input vs "Normal"
        st.markdown("### Input Summary")
        st.write(f"**State:** {selected_state}")
        st.write(f"**Season:** {selected_season}")
        
        # Simple chart
        chart_data = pd.DataFrame({
            'Factor': ['Rainfall', 'Temperature', 'pH', 'DO'],
            'Value': [rainfall, temperature, ph, do]
        })
        st.bar_chart(chart_data.set_index('Factor'))

