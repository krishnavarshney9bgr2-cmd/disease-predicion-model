# 🦟 India Disease Prediction — Streamlit App

A full-stack ML dashboard for month-by-month dengue disease prediction across 36 Indian states (2019–2023).

## 📁 Files
```
app.py                      ← Main Streamlit application
requirements.txt            ← Python dependencies
dengue_cases_in_india.csv   ← Annual state-wise dengue data
india_disease_final.csv     ← Monthly climate + dengue data (2160 rows)
```

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

App opens at **http://localhost:8501**

## ☁️ Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy** — done!

## 📊 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview & EDA | KPI cards, correlation heatmap, seasonal box plots |
| 🗺️ State Map Analysis | Folium bubble map, state heatmap, YoY trends |
| 📈 Time Series Trends | Multi-state line charts, rolling averages, violin plots |
| 🤖 ML Outbreak Classifier | Random Forest / GradBoost / Logistic classifier (98.8% accuracy) |
| 🧠 PyTorch LSTM Forecast | LSTM time-series model with future forecasting |
| 📅 Monthly Predictor | 12-month dengue case predictions with radar chart |

## 🛠️ Libraries Used
- **numpy / pandas** — data wrangling and feature engineering
- **matplotlib / seaborn** — static visualisations
- **plotly** — interactive charts
- **folium** — interactive geospatial map
- **scikit-learn** — ML classifiers & regressors
- **PyTorch** — LSTM deep learning model
- **streamlit** — web deployment

## 📐 Features Engineered
- `lag1_cases`, `lag2_cases` — 1 & 2 month case lags
- `rolling3` — 3-month rolling average
- `season_enc`, `state_enc` — label-encoded categorical features
