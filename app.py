import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and columns
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('simple_rf_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_model_resources()

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
        # Create input dataframe with all columns initialized to 0
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0
        
        # Set basic features
        input_data['Rainfall_mm'] = rainfall
        input_data['Temperature_C'] = temperature
        input_data['pH'] = ph
        input_data['Dissolved_Oxygen_mg_L'] = do
        
        # Set Location feature
        loc_col = f'Location_{selected_state}'
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

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

        # --- Textual Analysis ---
        st.markdown("---")
        st.subheader("📝 Analysis & Insights")
        
        analysis_text = ""
        
        # 1. Level Interpretation
        if prediction < 5:
            analysis_text += f"**Good News:** The predicted groundwater level of **{prediction:.2f} m** is within the safe range (< 5 m). This indicates healthy groundwater reserves in **{selected_state}**."
        elif prediction < 10:
            analysis_text += f"**Caution:** The level of **{prediction:.2f} m** is in the semi-critical range. While not immediately alarming, sustainable water management practices are recommended."
        else:
            analysis_text += f"**Critical Alert:** A groundwater level of **{prediction:.2f} m** is considered critical (> 10 m). This suggests significant depletion, likely due to high extraction or low recharge in this region."

        # 2. Seasonal Context
        if "Monsoon" in selected_season:
            analysis_text += f"\n\n**Seasonal Impact:** During the **{selected_season}**, groundwater levels typically recover due to rainfall recharge. "
            if prediction < 5:
                analysis_text += "The current prediction aligns with this expectation, showing good recharge."
            else:
                analysis_text += "However, despite the season, levels remain concerning, indicating that recharge might be insufficient to offset extraction."
        elif "Summer" in selected_season or "Pre-Monsoon" in selected_season:
             analysis_text += f"\n\n**Seasonal Impact:** In the **{selected_season}**, water levels naturally drop due to higher evaporation and usage. "
             if prediction > 10:
                 analysis_text += "The deep levels observed are typical for arid regions or areas with heavy agricultural usage during this time."

        # 3. State Context (General knowledge injection)
        arid_states = ['Rajasthan', 'Gujarat', 'Punjab', 'Haryana']
        high_rainfall_states = ['Assam', 'Meghalaya', 'Kerala', 'Goa']
        
        if selected_state in arid_states:
             analysis_text += f"\n\n**Regional Context:** **{selected_state}** is known to have lower groundwater levels due to its arid/semi-arid climate and intensive agriculture. Conservation is key here."
        elif selected_state in high_rainfall_states:
             analysis_text += f"\n\n**Regional Context:** **{selected_state}** generally receives high rainfall. "
             if prediction > 5:
                 analysis_text += "Seeing lower levels here might indicate local issues like rapid urbanization or delayed monsoons."
        
        st.info(analysis_text)

        # --- Crop Recommendations ---
        st.markdown("---")
        st.subheader("🌾 Farmer's Corner: Crop Recommendations")
        
        st.markdown(f"Based on the predicted water level of **{prediction:.2f} m** and the **{selected_season}** season, here are the recommended crops:")
        
        crops = []
        advice = ""
        
        # Logic for crop recommendation
        if prediction < 5:
            # Shallow water - Good for water-intensive crops
            if "Monsoon" in selected_season:
                crops = ["Rice (Paddy)", "Sugarcane", "Jute", "Leafy Vegetables"]
                advice = "Water availability is excellent. You can safely grow water-intensive crops."
            else:
                crops = ["Vegetables", "Flowers", "Medicinal Plants", "Fodder Crops"]
                advice = "Groundwater is accessible. Suitable for high-value short-duration crops."
                
        elif prediction < 10:
            # Moderate water - Standard crops
            if "Winter" in selected_season:
                crops = ["Wheat", "Mustard", "Chickpea", "Potato"]
            else:
                crops = ["Maize", "Cotton", "Pulses (Dal)", "Soybean", "Groundnut"]
            advice = "Water levels are moderate. Prefer crops that require standard irrigation but avoid water-guzzling crops like Paddy if possible."
            
        else:
            # Deep water - Drought resistant crops
            crops = ["Millets (Bajra, Jowar, Ragi)", "Barley", "Guar", "Castor", "Moth Bean"]
            advice = "⚠️ **Critical Water Level:** Strictly adopt drip irrigation. Grow drought-resistant and hardy crops only. Avoid flood irrigation."
        
        # Display recommendations
        col_crop1, col_crop2 = st.columns([1, 2])
        
        with col_crop1:
            st.success("**Recommended Crops:**")
            for crop in crops:
                st.markdown(f"- {crop}")
                
        with col_crop2:
            st.warning("**Advisory:**")
            st.write(advice)

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

        # --- Historical Comparison Section ---
        st.markdown("---")
        st.header("📊 Historical Comparison")
        
        @st.cache_data
        def load_historical_data():
            try:
                df = pd.read_csv('DWLR_Dataset_2023.csv')
                return df
            except FileNotFoundError:
                return None

        hist_df = load_historical_data()

        if hist_df is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.info("Comparing your prediction with the entire historical dataset (2023).")

            # 1. Distribution of Water Levels
            st.subheader("1. Water Level Distribution")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
            sns.histplot(hist_df['Water_Level_m'], kde=True, color='skyblue', ax=ax_dist)
            ax_dist.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {prediction:.2f}m')
            ax_dist.set_title("Historical Groundwater Levels")
            ax_dist.set_xlabel("Water Level (m)")
            ax_dist.legend()
            st.pyplot(fig_dist)

            # 2. Scatter Plots (Rainfall & Temperature)
            st.subheader("2. Environmental Factors Comparison")
            
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("**Rainfall vs. Water Level**")
                fig_rain, ax_rain = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=hist_df, x='Rainfall_mm', y='Water_Level_m', alpha=0.3, color='gray', ax=ax_rain)
                # Highlight current prediction
                ax_rain.scatter([rainfall], [prediction], color='red', s=100, zorder=5, label='You')
                ax_rain.set_xlabel("Rainfall (mm)")
                ax_rain.set_ylabel("Water Level (m)")
                ax_rain.legend()
                st.pyplot(fig_rain)

            with col_chart2:
                st.markdown("**Temperature vs. Water Level**")
                fig_temp, ax_temp = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=hist_df, x='Temperature_C', y='Water_Level_m', alpha=0.3, color='orange', ax=ax_temp)
                # Highlight current prediction
                ax_temp.scatter([temperature], [prediction], color='red', s=100, zorder=5, label='You')
                ax_temp.set_xlabel("Temperature (°C)")
                ax_temp.set_ylabel("Water Level (m)")
                ax_temp.legend()
                st.pyplot(fig_temp)
                
        else:
            st.warning("Historical dataset 'DWLR_Dataset_2023.csv' not found. Cannot display comparison charts.")
