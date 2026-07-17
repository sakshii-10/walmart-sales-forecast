# 🛒 Walmart Retail Sales Forecasting Dashboard

A end-to-end time-series forecasting project comparing **Prophet** and **SARIMA** models on weekly retail sales data across 45 Walmart stores — deployed as a live Streamlit app with a Power BI companion dashboard.

🔗 **[Live App](https://walmart-sales-forecast-msg2nsscssauch7hcq2o7p.streamlit.app/)** &nbsp;|&nbsp; 📊 **[Power BI Dashboard](#power-bi-dashboard)**

---

## 📌 Project Overview

Retail forecasting sits at the heart of FP&A — inventory planning, budget cycles, and variance analysis all depend on getting it right. This project takes that concept and builds it from scratch: raw data → model training → bias detection → live deployment.

| | |
|---|---|
| **Data** | Walmart Recruiting Store Sales (Kaggle) |
| **Records** | 6,435 weekly sales rows · 45 stores · 2010–2012 |
| **Forecast Horizon** | 12 weeks |
| **Best Model** | SARIMA — **1.65% MAPE** · **£890,763 RMSE** |

---

## 🧠 Models Compared

| Model | MAPE | RMSE |
|-------|------|------|
| Prophet | 2.05% | £1,218,589 |
| **SARIMA** | **1.65%** | **£890,763** |

Both models were trained on 131 weeks and evaluated on a 12-week holdout. Beyond accuracy, the project identifies **forecast bias** — whether a model consistently over or under-forecasts per store — a finding directly applicable to inventory and budget planning decisions.

---

## 🚀 Features

**Streamlit App (3 pages)**
- 📊 Dashboard — total sales KPIs, store contribution donut, holiday impact
- 🔮 Forecast — Prophet vs SARIMA chart with confidence intervals, model toggle, CSV export
- ⚖️ Model Compare — MAPE bar chart, dynamic best model, plain-English explanation

**Power BI Dashboard**
- KPI cards — Actual vs Prophet vs SARIMA vs Variance
- Actual vs Forecast line chart by month
- MAPE comparison bar chart
- Conditional formatting on variance (red = over-forecast, green = under-forecast)
- Dynamic insight card — updates per store selected

---

## 🗂️ Project Structure

```
walmart-sales-forecast/
├── app.py                      # Streamlit app
├── Walmart_Sales.csv           # Raw data (Kaggle)
├── forecasts_precomputed.csv   # Pre-trained forecasts — all 45 stores
├── requirements.txt            # Dependencies
└── README.md
```

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Power BI](https://img.shields.io/badge/PowerBI-F2C811?style=flat&logo=powerbi&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)

- **Forecasting:** Prophet, SARIMA (statsmodels)
- **App:** Streamlit, Plotly
- **BI:** Power BI Desktop, DAX
- **Data:** Pandas, NumPy, Scikit-learn

---

## 🏃 Run Locally

```bash
git clone https://github.com/sakshii-10/walmart-sales-forecast
cd walmart-sales-forecast
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 📊 Power BI Dashboard

Built as a companion to the Streamlit app — targeting non-technical stakeholders who need forecast insights without code.


---

## 💡 Key Insight

> SARIMA consistently **over-forecasts** for certain stores and **under-forecasts** for others. In a real retail context, over-forecasting leads to excess inventory costs; under-forecasting leads to stockouts and missed revenue. Identifying this bias is as valuable as the MAPE score itself.

---

## 👩‍💻 Author

**Sakshi Kothari** — Data & Finance Analyst  
MSc Big Data Science · Queen Mary University of London  
📎 [LinkedIn](https://www.linkedin.com/in/sakshikothari10) · 🌐 [Portfolio](https://skothari.me) · 💻 [GitHub](https://github.com/sakshii-10)
