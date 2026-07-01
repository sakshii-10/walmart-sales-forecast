import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

# ─────────────────────────────────────
# MODERN DARK UI THEME (LIKE YOUR IMAGE)
# ─────────────────────────────────────
st.markdown("""
<style>
body {
    background-color: #0B1320;
}

.block-container {
    padding: 1.5rem 2rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0F1B2D;
}

/* KPI CARDS */
div[data-testid="metric-container"] {
    background-color: #111C2E;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* HEADINGS */
h1, h2, h3 {
    color: #E6EDF7;
}

/* TEXT */
p, span {
    color: #AAB4C8;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# DATA
# ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    return df

df = load_data()

# ─────────────────────────────────────
# SIDEBAR (LIKE DASHBOARD IMAGE)
# ─────────────────────────────────────
with st.sidebar:
    st.title("📊 Navigation")

    page = st.radio("", ["Dashboard", "Forecast", "Model Compare"])

    store_nums = sorted(df['Store'].unique())
    store_map = {f"Store {str(s).zfill(2)}": s for s in store_nums}
    store_labels = ["All Stores"] + list(store_map.keys())

    selected_store = st.selectbox("Select Store", store_labels)

# ─────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────
if selected_store == "All Stores":
    filtered = df.copy()
else:
    filtered = df[df['Store'] == store_map[selected_store]]

weekly = filtered.groupby('Date')['Weekly_Sales'].sum().reset_index()
weekly = weekly.sort_values('Date')

train = weekly[:-12]
test = weekly[-12:]

# ─────────────────────────────────────
# MODELS
# ─────────────────────────────────────
@st.cache_data
def prophet_model(train_df):
    t = train_df.rename(columns={'Date':'ds','Weekly_Sales':'y'})
    m = Prophet(yearly_seasonality=True)
    m.fit(t)
    future = m.make_future_dataframe(periods=12, freq='W')
    return m.predict(future).tail(12)

@st.cache_data
def sarima_model(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,52))
    res = model.fit(disp=False)
    return res.forecast(12)

prophet_fc = prophet_model(train)
sarima_fc = sarima_model(train['Weekly_Sales'])

actual = test['Weekly_Sales'].values

# ─────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────
def metrics(y, p):
    return (
        mean_absolute_percentage_error(y, p),
        np.sqrt(mean_squared_error(y, p))
    )

mp, rp = metrics(actual, prophet_fc['yhat'])
ms, rs = metrics(actual, sarima_fc)

# ─────────────────────────────────────
# DASHBOARD PAGE (LIKE IMAGE)
# ─────────────────────────────────────
if page == "Dashboard":

    st.title("📊 Retail Performance Dashboard")

    # KPI STRIP (LIKE IMAGE TOP ROW)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${weekly['Weekly_Sales'].sum():,.0f}")
    c2.metric("Avg Weekly Sales", f"${weekly['Weekly_Sales'].mean():,.0f}")
    c3.metric("Peak Week", f"${weekly['Weekly_Sales'].max():,.0f}")
    c4.metric("Best Model", "Prophet" if mp < ms else "SARIMA")

    # MAIN CHART (CENTER FOCUS)
    st.subheader("📈 Sales Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly['Date'],
        y=weekly['Weekly_Sales'],
        name="Actual",
        line=dict(color="#4DA3FF", width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0B1320",
        plot_bgcolor="#0B1320"
    )

    st.plotly_chart(fig, use_container_width=True)

    # DONUT STYLE INSIGHTS (LIKE IMAGE RIGHT SIDE WIDGETS)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Store Contribution")
        store_share = df.groupby('Store')['Weekly_Sales'].sum().nlargest(5)

        fig2 = go.Figure(data=[go.Pie(
            labels=store_share.index.astype(str),
            values=store_share.values,
            hole=0.6
        )])
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📊 Holiday Impact")

        hol = filtered.groupby('Holiday_Flag')['Weekly_Sales'].mean()

        fig3 = go.Figure(data=[go.Pie(
            labels=["Non-Holiday", "Holiday"],
            values=hol.values,
            hole=0.6
        )])
        fig3.update_layout(template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────
# FORECAST PAGE
# ─────────────────────────────────────
elif page == "Forecast":

    st.title("🔮 Forecast Engine")

    st.line_chart(weekly.set_index('Date'))

    st.write("Prophet Forecast")
    st.dataframe(prophet_fc[['ds','yhat']])

    st.write("SARIMA Forecast")
    st.dataframe(pd.DataFrame({"Forecast": sarima_fc}))

# ─────────────────────────────────────
# ─────────────────────────────────────
# MODEL COMPARISON (EXPLAINED SIMPLY)
# ─────────────────────────────────────
elif page == "Model Compare":

    st.title("⚖️ Model Comparison (Easy Explanation)")

    st.markdown("### 📊 Performance Metrics")

    c1, c2 = st.columns(2)

    c1.metric("Prophet Error (MAPE)", f"{mp*100:.2f}%")
    c2.metric("SARIMA Error (MAPE)", f"{ms*100:.2f}%")

    st.markdown("---")

    # WINNER LOGIC
    if mp < ms:
        winner = "Prophet"
        reason = "Prophet performs better because it captures seasonality and trend changes more effectively."
    else:
        winner = "SARIMA"
        reason = "SARIMA performs better because it adapts better to short-term patterns in this dataset."

    st.success(f"🏆 Best Model: {winner}")

    st.markdown("### 🧠 What this means (Non-Technical Explanation)")

    st.info(
        f"""
        📌 **What is happening here?**  
        We tested both models on past sales and checked which one predicts better.

        📊 **Result:** {winner} is more accurate.

        💡 **Why?**  
        {reason}

        🧾 **How to use this:**  
        - Use {winner} for forecasting future sales  
        - It will give more reliable business planning results  
        """
    )

    st.markdown("### 📈 Simple Visual Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Prophet", "SARIMA"],
        y=[mp*100, ms*100],
        marker_color=["#FF9F43", "#00D084"]
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Model Error Comparison (Lower is Better)",
        yaxis_title="MAPE (%)",
        paper_bgcolor="#0B1320",
        plot_bgcolor="#0B1320"
    )

    st.plotly_chart(fig, use_container_width=True)