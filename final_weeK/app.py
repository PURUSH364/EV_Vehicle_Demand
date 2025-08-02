import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Adoption Forecaster",
    page_icon="img/favicon.png",  # Custom favicon (optional)
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    return joblib.load('forecasting_ev_model.pkl')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def generate_forecast(df, model, counties, forecast_horizon):
    all_forecast_dfs = []
    for county in counties:
        county_df = df[df['County'] == county].sort_values("Date").reset_index(drop=True)
        if county_df.empty:
            continue

        last_known_data = county_df.tail(1).copy()
        last_cumulative_value = last_known_data['cumulative_ev'].iloc[0]
        forecast_list, current_input_data = [], last_known_data

        for _ in range(forecast_horizon):
            next_date = current_input_data['Date'].iloc[0] + relativedelta(months=1)
            pred_features = pd.DataFrame(index=[0])
            pred_features['months_since_start'] = current_input_data['months_since_start'].iloc[0] + 1
            pred_features['county_encoded'] = current_input_data['county_encoded'].iloc[0]
            pred_features['ev_total_lag1'] = current_input_data['Electric Vehicle (EV) Total'].iloc[0]
            pred_features['ev_total_lag2'] = current_input_data['ev_total_lag1'].iloc[0]
            pred_features['ev_total_lag3'] = current_input_data['ev_total_lag2'].iloc[0]

            past_values = np.append(county_df['Electric Vehicle (EV) Total'].values, [f['Electric Vehicle (EV) Total'].iloc[0] for f in forecast_list])
            pred_features['ev_total_roll_mean_3'] = np.mean(past_values[-3:]) if len(past_values) >= 3 else np.mean(past_values)
            pred_features['ev_total_pct_change_1'] = (pred_features['ev_total_lag1'].iloc[0] - pred_features['ev_total_lag2'].iloc[0]) / pred_features['ev_total_lag2'].iloc[0] if pred_features['ev_total_lag2'].iloc[0] > 0 else 0
            pred_features['ev_total_pct_change_3'] = (pred_features['ev_total_lag1'].iloc[0] - pred_features['ev_total_lag3'].iloc[0]) / pred_features['ev_total_lag3'].iloc[0] if pred_features['ev_total_lag3'].iloc[0] > 0 else 0
            pred_features['ev_growth_slope'] = current_input_data['ev_growth_slope'].iloc[0]

            prediction = model.predict(pred_features[model.feature_names_in_])[0]

            new_row = pred_features.copy()
            new_row['Date'] = next_date
            new_row['County'] = county
            new_row['Electric Vehicle (EV) Total'] = max(0, round(prediction))
            forecast_list.append(new_row)
            current_input_data = new_row

        if not forecast_list:
            continue

        forecast_df_for_county = pd.concat(forecast_list, ignore_index=True)
        forecast_df_for_county['cumulative_ev'] = forecast_df_for_county['Electric Vehicle (EV) Total'].cumsum() + last_cumulative_value
        all_forecast_dfs.append(forecast_df_for_county)

    return pd.concat(all_forecast_dfs, ignore_index=True) if all_forecast_dfs else pd.DataFrame()

def display_results(historical_df, forecast_df, counties, forecast_years):
    last_hist_date = historical_df['Date'].max()
    historical_plot_df = historical_df[historical_df['County'].isin(counties)].copy()
    historical_plot_df['Type'] = 'Historical'
    forecast_plot_df = forecast_df.copy()
    forecast_plot_df['Type'] = 'Forecast'
    plot_df = pd.concat([historical_plot_df, forecast_plot_df])

    st.markdown("### Forecast Range: {} to {}".format(
        forecast_df['Date'].min().strftime("%b %Y"), forecast_df['Date'].max().strftime("%b %Y")))

    fig = px.line(plot_df, x='Date', y='cumulative_ev', color='County', line_dash='Type',
                  title='EV Adoption Trends: Historical vs. Forecast',
                  labels={'cumulative_ev': 'Cumulative EV Count'}, template='plotly_dark')
    fig.update_layout(title_font=dict(size=22, color='white'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.add_vline(x=last_hist_date.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF4B4B")
    st.plotly_chart(fig, use_container_width=True)

    top_growth = []
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists('img/context_image.jpg'):
            st.image('img/context_image.jpg', caption="Planning for an electric future.", use_container_width=True)
    with col2:
        if counties:
            metric_cols = st.columns(len(counties))
            for i, county in enumerate(counties):
                hist = historical_df[historical_df['County'] == county]
                fore = forecast_df[forecast_df['County'] == county]
                if hist.empty or fore.empty:
                    continue
                start_val, end_val = hist['cumulative_ev'].iloc[-1], fore['cumulative_ev'].iloc[-1]
                growth = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
                top_growth.append((county, growth))
                with metric_cols[i]:
                    st.metric(f"{county} Growth", f"{growth:.1f}%")
                    st.metric(f"{county} Total EVs", f"{int(end_val):,}")
            top3 = sorted(top_growth, key=lambda x: x[1], reverse=True)[:3]
            st.success("Top 3 Fastest Growing Counties: " + ", ".join([f"{c} ({g:.1f}%)" for c, g in top3]))

    with st.expander("📄 Forecast Data Table & CSV Export"):
        st.dataframe(forecast_df[['Date', 'County', 'Electric Vehicle (EV) Total', 'cumulative_ev']])
        st.download_button("Download Forecast CSV", convert_df_to_csv(forecast_df), f"forecast_data.csv")
        st.download_button("Download Historical CSV", convert_df_to_csv(historical_plot_df), f"historical_data.csv")

# --- Load Data ---
bg_image = get_base64_image('img/background1.jpg')
model = load_model()
df = load_data()

st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url(data:image/jpeg;base64,{bg_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .footer {{
        position: fixed; bottom: 0; width: 100%; text-align: center;
        padding: 10px; font-size: 13px; color: #a9a9a9; background-color: rgba(0,0,0,0.8);
    }}
    </style>
""", unsafe_allow_html=True)

if df is None:
    st.error("⚠️ Data or model files not found. Make sure they are in the directory.")
    st.stop()

with st.sidebar:
    st.title("⚡ Forecast Controls")
    st.markdown("Configure parameters below.")
    county_list = sorted(df['County'].dropna().unique().tolist())
    default_counties = [c for c in ["King", "Snohomish", "Pierce"] if c in county_list]
    selected_counties = st.multiselect("Select Counties", county_list, default=default_counties)
    forecast_years = st.slider("Forecast Years", 1, 5, 3)

st.title("Electric Vehicle Adoption Forecaster")
st.markdown("Predict EV trends across Washington State using machine learning.")

if os.path.exists('img/hero_image.jpg'):
    st.image('img/hero_image.jpg', use_container_width=True)

if not selected_counties:
    st.info("🔍 Please select at least one county from the sidebar to generate a forecast.")
else:
    with st.spinner(f"Generating forecast for: {', '.join(selected_counties)}"):
        forecast_months = forecast_years * 12
        forecasted_data = generate_forecast(df, model, selected_counties, forecast_months)
        if not forecasted_data.empty:
            display_results(df, forecasted_data, selected_counties, forecast_years)
        else:
            st.warning("No forecast data available for the selected counties.")

st.markdown('<div class="footer">EV Adoption Forecaster | A Project by Y PURUSHOTHAM REDDY</div>', unsafe_allow_html=True)
