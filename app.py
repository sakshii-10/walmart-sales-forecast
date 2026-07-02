import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #0B1320; }
.block-container { padding: 1.5rem 2rem; }
section[data-testid="stSidebar"] { background-color: #0F1B2D; }
div[data-testid="metric-container"] {
    background-color: #111C2E;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
h1, h2, h3 { color: #E6EDF7; }
p, span { color: #AAB4C8; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    return df

@st.cache_data
def load_forecasts():
    fc = pd.read_csv("forecasts_precomputed.csv")
    fc['Date'] = pd.to_datetime(fc['Date'])
    return fc

df = load_data()
fc_all = load_forecasts()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio("", ["Dashboard", "Forecast", "Model Compare"])

    store_nums = sorted(df['Store'].unique())
    store_map = {f"Store {str(s).zfill(2)}": s for s in store_nums}
    store_labels = ["All Stores"] + list(store_map.keys())
    selected_store = st.selectbox("Select Store", store_labels, key="store_select")
    model_choice = st.radio("Forecast Model", ["Both", "Prophet", "SARIMA"], key="model_select")


# ── Filter ─────────────────────────────────────────────────────────────────
if selected_store == "All Stores":
    filtered = df.copy()
else:
    filtered = df[df['Store'] == store_map[selected_store]]

weekly = filtered.groupby('Date')['Weekly_Sales'].sum().reset_index().sort_values('Date')

# ── Pull precomputed forecasts for selected store ──────────────────────────
fc = fc_all[fc_all['Store'] == selected_store].reset_index(drop=True)

mp = fc['Prophet_MAPE'].iloc[0]
rp = fc['Prophet_RMSE'].iloc[0]
ms = fc['SARIMA_MAPE'].iloc[0]
rs = fc['SARIMA_RMSE'].iloc[0]
winner = fc['Best_Model'].iloc[0]

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    st.title("📊 Retail Performance Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales",      f"${weekly['Weekly_Sales'].sum():,.2f}")
    c2.metric("Avg Weekly Sales", f"${weekly['Weekly_Sales'].mean():,.2f}")
    c3.metric("Peak Week",        f"${weekly['Weekly_Sales'].max():,.2f}")
    c4.metric("Best Model",       winner)

    st.subheader("📈 Sales Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly['Date'], y=weekly['Weekly_Sales'].round(2),
        name="Actual", line=dict(color="#4DA3FF", width=3)))
    fig.update_layout(
        template="plotly_dark", height=450,
        paper_bgcolor="#0B1320", plot_bgcolor="#0B1320",
        yaxis=dict(tickformat='$,.2f'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Top 5 Stores by Sales")
        store_share = df.groupby('Store')['Weekly_Sales'].sum().nlargest(5)
        fig2 = go.Figure(data=[go.Pie(
            labels=[f"Store {str(s).zfill(2)}" for s in store_share.index],
            values=store_share.round(2).values, hole=0.6)])
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0B1320")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📊 Holiday Impact")
        hol = filtered.groupby('Holiday_Flag')['Weekly_Sales'].mean().round(2)
        fig3 = go.Figure(data=[go.Pie(
            labels=["Non-Holiday", "Holiday"],
            values=hol.values, hole=0.6)])
        fig3.update_layout(template="plotly_dark", paper_bgcolor="#0B1320")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    st.title("🔮 Forecast Engine")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly['Date'], y=weekly['Weekly_Sales'].round(2),
        name='Actual', line=dict(color='#4DA3FF', width=2)))

    if model_choice in ["Both", "Prophet"]:
        fig.add_trace(go.Scatter(
            x=fc['Date'], y=fc['Prophet_Forecast'],
            name='Prophet', line=dict(color='#f4a261', dash='dash'),
            mode='lines+markers', marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc['Date'], fc['Date'][::-1]]),
            y=pd.concat([fc['Prophet_Upper'], fc['Prophet_Lower'][::-1]]),
            fill='toself', fillcolor='rgba(244,162,97,0.12)',
            line=dict(color='rgba(0,0,0,0)'), name='Prophet CI'))

    if model_choice in ["Both", "SARIMA"]:
        fig.add_trace(go.Scatter(
            x=fc['Date'], y=fc['SARIMA_Forecast'],
            name='SARIMA', line=dict(color='#2ec4b6', dash='dash'),
            mode='lines+markers', marker=dict(size=5)))

    fig.add_vline(x=str(weekly['Date'].iloc[-13]), line_dash="dot",
                  line_color="#e63946", annotation_text="Train / Test Split",
                  annotation_font_color="#e63946")
    fig.update_layout(
        template="plotly_dark", height=450,
        paper_bgcolor="#0B1320", plot_bgcolor="#0B1320",
        yaxis=dict(tickformat='$,.2f'),
        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Data")
    st.dataframe(fc[['Date','Actual','Prophet_Forecast','SARIMA_Forecast',
                      'Prophet_Lower','Prophet_Upper']].round(2),
                 hide_index=True, use_container_width=True)

    csv = fc.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV (Power BI)", csv, "forecast_output.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Model Compare":

    st.title("⚖️ Model Comparison")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prophet MAPE", f"{mp:.2f}%")
    c2.metric("Prophet RMSE", f"${rp:,.2f}")
    c3.metric("SARIMA MAPE",  f"{ms:.2f}%", delta=f"{(mp-ms):.2f}% vs Prophet", delta_color="inverse")
    c4.metric("SARIMA RMSE",  f"${rs:,.2f}")

    st.markdown("---")
    st.success(f"🏆 Best Model: {winner}")

    if winner == "Prophet":
        reason = "Prophet captures seasonality and trend changes more effectively on this data."
    else:
        reason = "SARIMA adapts better to short-term autocorrelation patterns in this dataset."

    st.info(f"""
**What is happening here?**  
Both models were trained on 131 weeks of data and tested on the final 12 weeks.

**Result:** {winner} is more accurate for {selected_store}.

**Why?** {reason}

**MAPE** = average % error per week. Lower is better.  
**RMSE** = average $ error per week. Lower is better.
    """)

    fig = go.Figure(data=[go.Bar(
        x=["Prophet", "SARIMA"],
        y=[round(mp, 2), round(ms, 2)],
        marker_color=["#f4a261", "#2ec4b6"],
        text=[f"{mp:.2f}%", f"{ms:.2f}%"],
        textposition='auto')])
    fig.update_layout(
        template="plotly_dark",
        title="MAPE Comparison — Lower is Better",
        yaxis_title="MAPE (%)",
        paper_bgcolor="#0B1320", plot_bgcolor="#0B1320")
    st.plotly_chart(fig, use_container_width=True)
