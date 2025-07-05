import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="Attrition Model Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .alert-error {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-success {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #f9f9f9;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 13px;
        color: #555;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 13px;
        color: green;
    }
    .metric-delta.red {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Attrition Model Monitoring Dashboard</h1>', unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Daily prediction volume
    daily_volume = []
    for date in dates:
        base_volume = 1000 + random.randint(-200, 300)
        # Add weekly pattern (lower on weekends)
        if date.weekday() >= 5:  # Weekend
            base_volume *= 0.7
        daily_volume.append(int(base_volume))
    
    # Model performance metrics
    precision_scores = [0.82 + random.uniform(-0.04, 0.06) for _ in dates]
    recall_scores = [0.78 + random.uniform(-0.06, 0.08) for _ in dates]
    f2_scores = [0.80 + random.uniform(-0.05, 0.05) for _ in dates]
    
    # System errors
    error_counts = [random.randint(0, 15) for _ in dates]
    
    # Data drift indicators
    drift_scores = [random.uniform(0.1, 0.9) for _ in dates]
    
    return {
        'dates': dates,
        'daily_volume': daily_volume,
        'precision': precision_scores,
        'recall': recall_scores,
        'f2': f2_scores,
        'errors': error_counts,
        'drift_scores': drift_scores
    }

# Sidebar for controls
st.sidebar.header("Dashboard Controls")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Last 7 Days", "Last 14 Days", "Last 30 Days"],
    index=2
)

refresh_data = st.sidebar.button("Refresh Data")
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")

# Generate data
data = generate_sample_data()

# Filter data based on time range
if time_range == "Last 7 Days":
    data_slice = slice(-7, None)
elif time_range == "Last 14 Days":
    data_slice = slice(-14, None)
else:
    data_slice = slice(None)

# Key Metrics Row
st.markdown("---")
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_predictions = sum(data['daily_volume'][data_slice])
    delta_pred = f"{random.randint(-5, 15)}% vs last period"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Total Predictions</div>
        <div class="metric-value">{total_predictions:,}</div>
        <div class="metric-delta">⬆ {delta_pred}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_f2score = np.mean(data['f2'][data_slice])
    delta_f2 = random.uniform(-0.02, 0.02)
    delta_color = "red" if delta_f2 < 0 else "green"
    arrow = "⬇" if delta_f2 < 0 else "⬆"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Average F2-Score</div>
        <div class="metric-value">{avg_f2score:.3f}</div>
        <div class="metric-delta {delta_color}">{arrow} {delta_f2:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_errors = sum(data['errors'][data_slice])
    delta_errors = f"{random.randint(-3, 8)} vs last period"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Total Errors</div>
        <div class="metric-value">{total_errors}</div>
        <div class="metric-delta">⬆ {delta_errors}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_drift = np.mean(data['drift_scores'][data_slice])
    drift_status = "🟢 Normal" if avg_drift < 0.5 else "🟡 Warning" if avg_drift < 0.7 else "🔴 Critical"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Data Drift Status</div>
        <div class="metric-value">{drift_status}</div>
        <div class="metric-delta">⬆ Score: {avg_drift:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

# Main dashboard content
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    # Daily/Weekly Prediction Volume
    st.markdown("### Prediction Volume Trends")
    
    tab1, tab2 = st.tabs(["Daily Volume", "Weekly Aggregate"])
    
    with tab1:
        fig_volume = px.line(
            x=data['dates'][data_slice],
            y=data['daily_volume'][data_slice],
            title="Daily Prediction Volume",
            labels={'x': 'Date', 'y': 'Number of Predictions'}
        )
        fig_volume.update_layout(height=400)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        # Weekly aggregation
        df_temp = pd.DataFrame({
            'date': data['dates'][data_slice],
            'volume': data['daily_volume'][data_slice]
        })
        df_temp['week'] = df_temp['date'].dt.isocalendar().week
        weekly_volume = df_temp.groupby('week')['volume'].sum().reset_index()
        
        fig_weekly = px.bar(
            weekly_volume,
            x='week',
            y='volume',
            title="Weekly Prediction Volume",
            labels={'week': 'Week Number', 'volume': 'Total Predictions'}
        )
        fig_weekly.update_layout(height=400)
        st.plotly_chart(fig_weekly, use_container_width=True)

with col_right:
    # Recent System Errors
    st.markdown(f'<div style="margin-top:60px"></div>', unsafe_allow_html=True)
    st.markdown("#### Recent System Errors")
    
    recent_errors = data['errors'][-7:]
    recent_dates = data['dates'][-7:]
    
    error_df = pd.DataFrame({
        'Date': recent_dates,
        'Errors': recent_errors
    })
    
    # Color code based on error count
    for idx, row in error_df.iterrows():
        if row['Errors'] > 10:
            st.markdown(f'<div class="alert-error">🔴 {row["Date"].strftime("%Y-%m-%d")}: {row["Errors"]} errors</div>', unsafe_allow_html=True)
        elif row['Errors'] > 5:
            st.markdown(f'<div class="alert-warning">🟡 {row["Date"].strftime("%Y-%m-%d")}: {row["Errors"]} errors</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success">🟢 {row["Date"].strftime("%Y-%m-%d")}: {row["Errors"]} errors</div>', unsafe_allow_html=True)

cm = np.array([[85,  5],   # True‑Pos, False‑Neg
               [ 7, 103]])  # False‑Pos, True‑Neg

st.markdown("---")
st.markdown("### Model Performance Trends")

#  Line‑chart tren
fig_performance = go.Figure()

fig_performance.add_trace(go.Scatter(
    x=data['dates'][data_slice],
    y=data['precision'][data_slice],
    mode='lines+markers',
    name='Precision',
    line=dict(color='#ff7f0e', width=3)
))

fig_performance.add_trace(go.Scatter(
    x=data['dates'][data_slice],
    y=data['recall'][data_slice],
    mode='lines+markers',
    name='Recall',
    line=dict(color='#2ca02c', width=3)
))

fig_performance.add_trace(go.Scatter(
    x=data['dates'][data_slice],
    y=data['f2'][data_slice],
    mode='lines+markers',
    name='F2‑Score',
    line=dict(color='#d62728', width=3)
))

fig_performance.update_layout(
    title="Model Performance Metrics Over Time",
    xaxis_title="Date",
    yaxis_title="Score",
    height=400,
    yaxis=dict(range=[0.7, 1.0])
)

st.plotly_chart(fig_performance, use_container_width=True)

# Radar & Confusion 
col_a, col_b = st.columns(2)

# Grafik A: Radar 
with col_a:
    current_metrics = {
        'Precision': data['precision'][-1],
        'Recall':    data['recall'][-1],
        'F2‑Score':  data['f2'][-1]
    }

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(current_metrics.values()),
        theta=list(current_metrics.keys()),
        fill='toself',
        name='Current Performance',
        line=dict(color='#1f77b4')
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.7, 1.0])
        ),
        showlegend=True,
        title="Current Model Performance",
        height=400
    )

    st.plotly_chart(fig_radar, use_container_width=True)

        # Grafik B: Confusion Matrix
    with col_b:
        labels = ['Positive', 'Negative']
        z = cm

        fig_cm = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            showscale=True,
            colorscale='Blues',
            hovertemplate='Count: %{z}<extra></extra>',
        ))

        threshold = z.max() / 2.0  
        annotations = []

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                val = z[i, j]
                font_color = 'white' if val > threshold else 'black'
                annotations.append(
                    dict(
                        x=labels[j],
                        y=labels[i],
                        text=str(val),
                        showarrow=False,
                        font=dict(color=font_color, size=18)
                    )
                )

        fig_cm.update_layout(
            title="Confusion Matrix (Latest Model)",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            annotations=annotations,
            height=400
        )

        st.plotly_chart(fig_cm, use_container_width=True)


# Data Drift Indicators
st.markdown("---")
st.markdown("### Data Drift Indicators")

col1, col2 = st.columns([2.5, 1])

with col1:
    # Data drift over time
    fig_drift = go.Figure()
    
    colors = ['green' if x < 0.5 else 'orange' if x < 0.7 else 'red' for x in data['drift_scores'][data_slice]]
    
    fig_drift.add_trace(go.Scatter(
        x=data['dates'][data_slice],
        y=data['drift_scores'][data_slice],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(color=colors, size=8)
    ))
    
    # Add threshold lines
    fig_drift.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig_drift.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    
    fig_drift.update_layout(
        title="Data Drift Score Over Time",
        xaxis_title="Date",
        yaxis_title="Drift Score",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig_drift, use_container_width=True)

with col2:
    # Drift summary
    st.markdown("#### Drift Summary")
    
    recent_drift = data['drift_scores'][-7:]
    normal_days = sum(1 for x in recent_drift if x < 0.5)
    warning_days = sum(1 for x in recent_drift if 0.5 <= x < 0.7)
    critical_days = sum(1 for x in recent_drift if x >= 0.7)
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">🟢 Normal Days</div>
        <div class="metric-value">{normal_days}/7</div>
    </div>
    <div class="metric-container">
        <div class="metric-label">🟡 Warning Days</div>
        <div class="metric-value">{warning_days}/7</div>
    </div>
    <div class="metric-container">
        <div class="metric-label">🔴 Critical Days</div>
        <div class="metric-value">{critical_days}/7</div>
    </div>
    """, unsafe_allow_html=True)

# Drift recommendations
st.markdown("#### Recommendations")
if critical_days > 0:
    st.warning("Critical drift detected! Consider retraining the model.")
elif warning_days > 2:
    st.info("Moderate drift observed. Monitor closely.")
else:
    st.success("Data drift is within acceptable limits.")

# Footer
st.markdown("---")
st.markdown("### System Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Total Data Points**: {len(data['dates'][data_slice])}")
with col2:
    st.info("**Version**: v1.0.0")
with col3:
    st.info("**Status**: 🟢 Online")

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.info(f"**Data Range**: {time_range}")

# Auto-refresh functionality
if auto_refresh:
    st.rerun()