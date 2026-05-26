import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="India Disease Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#E63946", "#457B9D", "#2EC4B6", "#FF9F1C", "#8338EC", "#06D6A0"]

MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli": (20.1809, 73.0169),
    "Daman and Diu": (20.3974, 72.8328),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Lakshadweep": (10.5667, 72.6417),
    "Puducherry": (11.9416, 79.8083),
}

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "india_disease_final.csv"))
    dengue = pd.read_csv(os.path.join(base, "dengue_cases_in_india.csv"))

    df["month_name"] = df["month"].map(MONTH_MAP)
    df["date"] = pd.to_datetime(
        df[["year", "month"]].assign(day=1)
    )
    df["lag1_cases"] = df.groupby("state")["dengue_cases"].shift(1).fillna(0)
    df["lag2_cases"] = df.groupby("state")["dengue_cases"].shift(2).fillna(0)
    df["rolling3"] = (
        df.groupby("state")["dengue_cases"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    le = LabelEncoder()
    df["season_enc"] = le.fit_transform(df["season"])
    df["state_enc"] = le.fit_transform(df["state"])
    return df, dengue

df, dengue_df = load_data()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg",
    width=100,
)
st.sidebar.title("🦟 India Disease Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview & EDA",
        "🗺️ State Map Analysis",
        "📈 Time Series Trends",
        "🤖 ML Outbreak Classifier",
        "🧠 PyTorch LSTM Forecast",
        "📅 Monthly Predictor",
    ],
)

st.sidebar.markdown("---")
selected_state = st.sidebar.selectbox("Filter State", ["All"] + sorted(df["state"].unique().tolist()))
selected_year  = st.sidebar.selectbox("Filter Year",  ["All"] + sorted(df["year"].unique().tolist()))

fdf = df.copy()
if selected_state != "All":
    fdf = fdf[fdf["state"] == selected_state]
if selected_year != "All":
    fdf = fdf[fdf["year"] == int(selected_year)]

st.sidebar.markdown(f"**Rows shown:** {len(fdf):,}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with **Streamlit · sklearn · PyTorch · Folium · Plotly**"
)

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════
if page == "🏠 Overview & EDA":
    st.title("🦟 India Dengue Disease Dashboard")
    st.markdown("Monthly climate-driven dengue analysis across 36 Indian states (2019–2023).")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases",   f"{fdf['dengue_cases'].sum():,}")
    c2.metric("Outbreak Months", f"{fdf['outbreak'].sum():,}")
    c3.metric("Avg Temp (°C)", f"{fdf['temperature'].mean():.1f}")
    c4.metric("Avg Humidity (%)", f"{fdf['humidity'].mean():.1f}")

    st.markdown("---")

    # Distribution of dengue cases
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Case Distribution by Season")
        fig = px.box(
            fdf, x="season", y="dengue_cases", color="season",
            color_discrete_sequence=PALETTE,
            labels={"dengue_cases": "Dengue Cases", "season": "Season"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Outbreak Proportion by Season")
        outbreak_season = (
            fdf.groupby("season")["outbreak"].mean().reset_index()
        )
        outbreak_season["outbreak_pct"] = outbreak_season["outbreak"] * 100
        fig2 = px.bar(
            outbreak_season, x="season", y="outbreak_pct",
            color="season", color_discrete_sequence=PALETTE,
            labels={"outbreak_pct": "Outbreak % of Months", "season": "Season"},
            text_auto=".1f",
        )
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Feature Correlation Heatmap")
        num_cols = ["temperature", "rainfall", "humidity", "dengue_cases", "outbreak"]
        corr = fdf[num_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax3, linewidths=0.5,
        )
        ax3.set_title("Pearson Correlation", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col4:
        st.subheader("Monthly Average Cases (All States)")
        monthly_avg = fdf.groupby("month")["dengue_cases"].mean().reset_index()
        monthly_avg["month_name"] = monthly_avg["month"].map(MONTH_MAP)
        fig4 = px.line(
            monthly_avg, x="month_name", y="dengue_cases",
            markers=True, color_discrete_sequence=[PALETTE[0]],
            labels={"dengue_cases": "Avg Cases", "month_name": "Month"},
        )
        fig4.update_traces(line_width=2.5, marker_size=8)
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Dataset Preview")
    st.dataframe(fdf.head(100), use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 2 — STATE MAP ANALYSIS
# ══════════════════════════════════════════════
elif page == "🗺️ State Map Analysis":
    st.title("🗺️ State-wise Dengue Map")

    map_year = st.selectbox("Select Year for Map", sorted(df["year"].unique()), index=4)
    map_data = df[df["year"] == map_year].groupby("state")["dengue_cases"].sum().reset_index()
    map_data.columns = ["state", "total_cases"]

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    max_cases = map_data["total_cases"].max()

    for _, row in map_data.iterrows():
        state = row["state"]
        cases = row["total_cases"]
        if state in STATE_COORDS:
            lat, lon = STATE_COORDS[state]
            radius = 8 + (cases / max_cases) * 25
            intensity = int(200 * (1 - cases / max_cases))
            color = f"#{255:02x}{intensity:02x}{intensity:02x}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{state}</b><br>Cases ({map_year}): {cases:,}",
                    max_width=200,
                ),
                tooltip=f"{state}: {cases:,} cases",
            ).add_to(m)

    st_folium(m, width=900, height=500)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top 10 States — {map_year}")
        top10 = map_data.nlargest(10, "total_cases")
        fig = px.bar(
            top10, x="total_cases", y="state", orientation="h",
            color="total_cases", color_continuous_scale="Reds",
            labels={"total_cases": "Total Cases", "state": "State"},
        )
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Year-over-Year Total Cases")
        yearly = df.groupby("year")["dengue_cases"].sum().reset_index()
        fig2 = px.area(
            yearly, x="year", y="dengue_cases",
            color_discrete_sequence=[PALETTE[0]],
            labels={"dengue_cases": "Total Cases", "year": "Year"},
            markers=True,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Dengue Cases Heatmap — State × Month")
    pivot_year = st.selectbox("Year for Heatmap", sorted(df["year"].unique()), key="hm_year")
    pivot = (
        df[df["year"] == pivot_year]
        .groupby(["state", "month"])["dengue_cases"]
        .sum()
        .reset_index()
        .pivot(index="state", columns="month", values="dengue_cases")
        .fillna(0)
    )
    pivot.columns = [MONTH_MAP[c] for c in pivot.columns]
    fig3 = px.imshow(
        pivot, color_continuous_scale="YlOrRd",
        labels={"color": "Cases"},
        height=700,
    )
    fig3.update_layout(xaxis_title="Month", yaxis_title="State")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — TIME SERIES TRENDS
# ══════════════════════════════════════════════
elif page == "📈 Time Series Trends":
    st.title("📈 Time Series Analysis")

    ts_states = st.multiselect(
        "Select States to Compare",
        sorted(df["state"].unique()),
        default=["Maharashtra", "Kerala", "Delhi", "Tamil Nadu"],
    )

    if not ts_states:
        st.warning("Please select at least one state.")
        st.stop()

    ts_df = df[df["state"].isin(ts_states)]
    monthly_state = ts_df.groupby(["date", "state"])["dengue_cases"].sum().reset_index()

    fig = px.line(
        monthly_state, x="date", y="dengue_cases", color="state",
        color_discrete_sequence=PALETTE,
        labels={"dengue_cases": "Cases", "date": "Date", "state": "State"},
        title="Monthly Dengue Cases Trend",
    )
    fig.update_traces(line_width=2)
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Climate Features vs Cases")
        scatter_state = st.selectbox("State", sorted(df["state"].unique()), key="scatter")
        sdf = df[df["state"] == scatter_state]
        feature = st.selectbox("Climate Feature", ["temperature", "rainfall", "humidity"])
        fig2 = px.scatter(
            sdf, x=feature, y="dengue_cases", color="season",
            color_discrete_sequence=PALETTE,
            trendline="ols",
            labels={"dengue_cases": "Cases"},
            title=f"{feature.title()} vs Cases — {scatter_state}",
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Monthly Case Distribution (Violin)")
        violin_df = df.copy()
        violin_df["month_name"] = violin_df["month"].map(MONTH_MAP)
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        month_order = list(MONTH_MAP.values())
        sns.violinplot(
            data=violin_df[violin_df["state"].isin(ts_states)],
            x="month_name", y="dengue_cases",
            order=month_order,
            palette="husl", inner="quartile", ax=ax3,
            cut=0,
        )
        ax3.set_title("Case Distribution by Month", fontsize=11)
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Dengue Cases")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown("---")
    st.subheader("Rolling 3-Month Average")
    roll_state = st.selectbox("State for Rolling Avg", sorted(df["state"].unique()), key="roll")
    rdf = df[df["state"] == roll_state].sort_values("date")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=rdf["date"], y=rdf["dengue_cases"], name="Monthly Cases",
                          marker_color="rgba(69,123,157,0.5)"))
    fig4.add_trace(go.Scatter(x=rdf["date"], y=rdf["rolling3"], name="3-Month Rolling Avg",
                              line=dict(color=PALETTE[0], width=2.5)))
    fig4.update_layout(height=400, hovermode="x unified",
                       title=f"Rolling Average — {roll_state}",
                       xaxis_title="Date", yaxis_title="Cases")
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 4 — ML OUTBREAK CLASSIFIER
# ══════════════════════════════════════════════
elif page == "🤖 ML Outbreak Classifier":
    st.title("🤖 ML Outbreak Classifier")
    st.markdown("Predict whether a **dengue outbreak** will occur given climate & lag features.")

    FEATURES = ["temperature", "rainfall", "humidity", "season_enc", "state_enc",
                "month", "lag1_cases", "lag2_cases", "rolling3"]
    TARGET = "outbreak"

    model_data = df.dropna(subset=FEATURES + [TARGET])
    X = model_data[FEATURES].values
    y = model_data[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    col_m, col_p = st.columns([1, 2])
    with col_m:
        model_choice = st.selectbox(
            "Choose Model",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"]
        )
        n_est = st.slider("n_estimators (trees)", 50, 300, 150, 50)

    @st.cache_resource
    def train_model(choice, n_est, _X_train, _y_train):
        if choice == "Random Forest":
            m = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        elif choice == "Gradient Boosting":
            m = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
        else:
            m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(_X_train, _y_train)
        return m

    with st.spinner("Training model…"):
        clf = train_model(model_choice, n_est, X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with col_p:
        st.metric("Test Accuracy", f"{acc:.2%}")
        st.metric("Outbreak Recall", f"{report['1']['recall']:.2%}")
        st.metric("Outbreak Precision", f"{report['1']['precision']:.2%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Outbreak","Outbreak"],
                    yticklabels=["No Outbreak","Outbreak"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        if hasattr(clf, "feature_importances_"):
            st.subheader("Feature Importances")
            fi = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": clf.feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig2 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Teal")
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Predict Outbreak for New Input")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        inp_temp = st.slider("Temperature (°C)", 10.0, 45.0, 30.0, 0.5)
        inp_rain = st.slider("Rainfall (mm)", 0.0, 500.0, 150.0, 5.0)
    with pc2:
        inp_humid = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0)
        inp_month = st.selectbox("Month", range(1, 13), format_func=lambda x: MONTH_MAP[x])
    with pc3:
        inp_season = st.selectbox("Season", ["Winter","Summer","Monsoon","Post"])
        inp_state  = st.selectbox("State", sorted(df["state"].unique()), key="clf_state")
        inp_lag1   = st.number_input("Cases Last Month", 0, 10000, 100)
        inp_lag2   = st.number_input("Cases 2 Months Ago", 0, 10000, 80)

    season_map = {"Winter":3,"Summer":2,"Monsoon":1,"Post":0}
    state_vals = {s: i for i, s in enumerate(sorted(df["state"].unique()))}
    inp_rolling = (inp_lag1 + inp_lag2) / 2

    inp_vec = np.array([[inp_temp, inp_rain, inp_humid,
                         season_map[inp_season], state_vals[inp_state],
                         inp_month, inp_lag1, inp_lag2, inp_rolling]])
    inp_scaled = scaler.transform(inp_vec)

    if st.button("Predict Outbreak Risk", type="primary"):
        pred = clf.predict(inp_scaled)[0]
        prob = clf.predict_proba(inp_scaled)[0][1] if hasattr(clf, "predict_proba") else None
        if pred == 1:
            st.error(f"🚨 **OUTBREAK LIKELY!** Probability: {prob:.1%}" if prob else "🚨 Outbreak likely!")
        else:
            st.success(f"✅ **No Outbreak Expected.** Probability: {prob:.1%}" if prob else "✅ No outbreak expected.")

# ══════════════════════════════════════════════
# PAGE 5 — PYTORCH LSTM FORECAST
# ══════════════════════════════════════════════
elif page == "🧠 PyTorch LSTM Forecast":
    st.title("🧠 PyTorch LSTM Time-Series Forecasting")
    st.markdown("A deep-learning LSTM trained on historical monthly dengue cases to forecast upcoming months.")

    lstm_state = st.selectbox("Select State", sorted(df["state"].unique()), key="lstm_state")
    seq_len    = st.slider("Sequence Length (months look-back)", 3, 12, 6)
    epochs     = st.slider("Training Epochs", 30, 200, 80, 10)
    forecast_months = st.slider("Months to Forecast Ahead", 1, 12, 6)

    # Prepare state time series
    state_ts = (
        df[df["state"] == lstm_state]
        .sort_values("date")[["date", "dengue_cases"]]
        .reset_index(drop=True)
    )

    values = state_ts["dengue_cases"].values.astype(np.float32)
    # Normalize
    v_min, v_max = values.min(), values.max() + 1e-6
    normed = (values - v_min) / (v_max - v_min)

    def make_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X_seq, y_seq = make_sequences(normed, seq_len)
    split = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    X_tr_t = torch.tensor(X_tr).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr).unsqueeze(-1)
    X_te_t = torch.tensor(X_te).unsqueeze(-1)

    # LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2)
            self.fc   = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    @st.cache_resource
    def train_lstm(state_key, seq_len, epochs, _X_tr, _y_tr):
        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        dataset = TensorDataset(_X_tr, _y_tr)
        loader  = DataLoader(dataset, batch_size=16, shuffle=True)
        losses  = []
        for _ in range(epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
        return model, losses

    with st.spinner(f"Training LSTM on {lstm_state} data…"):
        lstm_model, losses = train_lstm(
            lstm_state, seq_len, epochs, X_tr_t, y_tr_t
        )

    lstm_model.eval()
    with torch.no_grad():
        test_preds = lstm_model(X_te_t).squeeze().numpy()
        train_preds = lstm_model(X_tr_t).squeeze().numpy()

    # Denormalize
    def denorm(x): return x * (v_max - v_min) + v_min
    y_te_orig   = denorm(y_te)
    preds_orig  = denorm(test_preds if test_preds.ndim > 0 else np.array([test_preds]))
    tr_pred_orig = denorm(train_preds if train_preds.ndim > 0 else np.array([train_preds]))

    rmse = np.sqrt(mean_squared_error(y_te_orig, preds_orig))
    mae  = mean_absolute_error(y_te_orig, preds_orig)
    r2   = r2_score(y_te_orig, preds_orig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Test RMSE", f"{rmse:.1f}")
    m2.metric("Test MAE",  f"{mae:.1f}")
    m3.metric("R² Score",  f"{r2:.3f}")

    st.markdown("---")

    # Forecast future months
    last_seq = normed[-seq_len:]
    forecast = []
    for _ in range(forecast_months):
        inp = torch.tensor(last_seq[-seq_len:]).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            nxt = lstm_model(inp).item()
        forecast.append(nxt)
        last_seq = np.append(last_seq, nxt)

    forecast_vals = denorm(np.array(forecast))
    last_date = state_ts["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months, freq="MS"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted + Forecast")
        dates_tr = state_ts["date"].iloc[seq_len:seq_len+len(tr_pred_orig)]
        dates_te = state_ts["date"].iloc[seq_len+len(tr_pred_orig):seq_len+len(tr_pred_orig)+len(preds_orig)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=state_ts["date"], y=state_ts["dengue_cases"],
                                 name="Actual", line=dict(color="#457B9D", width=2)))
        fig.add_trace(go.Scatter(x=dates_tr, y=tr_pred_orig,
                                 name="Train Pred", line=dict(color="#2EC4B6", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=dates_te, y=preds_orig,
                                 name="Test Pred", line=dict(color="#FF9F1C", width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals,
                                 name="Forecast", line=dict(color=PALETTE[0], width=2.5, dash="dash"),
                                 mode="lines+markers"))
        fig.update_layout(height=420, hovermode="x unified",
                          title=f"LSTM Forecast — {lstm_state}",
                          xaxis_title="Date", yaxis_title="Dengue Cases")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Training Loss Curve")
        fig2 = px.line(y=losses, labels={"y":"Loss","index":"Epoch"},
                       color_discrete_sequence=[PALETTE[0]])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(f"Forecast Next {forecast_months} Months")
        fcast_df = pd.DataFrame({
            "Month": [d.strftime("%b %Y") for d in future_dates],
            "Predicted Cases": forecast_vals.astype(int)
        })
        st.dataframe(fcast_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# PAGE 6 — MONTHLY PREDICTOR
# ══════════════════════════════════════════════
elif page == "📅 Monthly Predictor":
    st.title("📅 Month-by-Month Case Predictor")
    st.markdown(
        "Use a **Gradient Boosting Regressor** trained on all features to predict "
        "exact case counts for any state, month, and climate scenario."
    )

    FEATURES_REG = ["temperature", "rainfall", "humidity", "season_enc", "state_enc",
                    "month", "lag1_cases", "lag2_cases", "rolling3"]
    TARGET_REG = "dengue_cases"

    reg_data = df.dropna(subset=FEATURES_REG + [TARGET_REG])
    X_r = reg_data[FEATURES_REG].values
    y_r = reg_data[TARGET_REG].values

    scaler_r = StandardScaler()
    X_rs = scaler_r.fit_transform(X_r)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_rs, y_r, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_gbr(_X, _y):
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        m = GBR(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        m.fit(_X, _y)
        return m

    with st.spinner("Training regressor…"):
        gbr = train_gbr(X_tr2, y_tr2)

    preds_r = gbr.predict(X_te2)
    rmse_r = np.sqrt(mean_squared_error(y_te2, preds_r))
    r2_r   = r2_score(y_te2, preds_r)

    m1, m2 = st.columns(2)
    m1.metric("Model RMSE", f"{rmse_r:.1f} cases")
    m2.metric("Model R²",   f"{r2_r:.3f}")

    st.markdown("---")
    st.subheader("Generate Month-by-Month Predictions")

    col1, col2 = st.columns(2)
    with col1:
        pred_state  = st.selectbox("State", sorted(df["state"].unique()), key="pred_state")
        pred_year   = st.selectbox("Year", [2024, 2025, 2026], key="pred_year")
        pred_temp   = st.slider("Avg Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
        pred_rain   = st.slider("Avg Rainfall (mm)", 0.0, 400.0, 100.0, 5.0)
    with col2:
        pred_humid  = st.slider("Avg Humidity (%)", 30.0, 100.0, 65.0, 1.0)
        pred_lag1   = st.number_input("Starting Lag-1 Cases", 0, 10000, 150)
        pred_lag2   = st.number_input("Starting Lag-2 Cases", 0, 10000, 100)

    season_map2 = {1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
                   6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Post",10:"Post",
                   11:"Winter",12:"Winter"}
    season_enc_map = {"Winter":3,"Summer":2,"Monsoon":1,"Post":0}
    state_enc_val  = {s: i for i, s in enumerate(sorted(df["state"].unique()))}

    if st.button("Generate 12-Month Forecast", type="primary"):
        results = []
        lag1, lag2 = pred_lag1, pred_lag2
        for month in range(1, 13):
            season = season_map2[month]
            rolling = (lag1 + lag2) / 2
            row_vec = np.array([[pred_temp, pred_rain, pred_humid,
                                  season_enc_map[season],
                                  state_enc_val[pred_state],
                                  month, lag1, lag2, rolling]])
            row_scaled = scaler_r.transform(row_vec)
            pred_cases = max(0, int(gbr.predict(row_scaled)[0]))
            results.append({
                "Month": MONTH_MAP[month],
                "Season": season,
                "Predicted Cases": pred_cases,
                "Risk": "🔴 High" if pred_cases > 500 else ("🟡 Medium" if pred_cases > 100 else "🟢 Low")
            })
            lag2 = lag1
            lag1 = pred_cases

        results_df = pd.DataFrame(results)
        st.markdown("### 📋 Predicted Cases — All 12 Months")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig = px.bar(
            results_df, x="Month", y="Predicted Cases", color="Season",
            color_discrete_sequence=PALETTE, text="Predicted Cases",
            title=f"Monthly Dengue Predictions — {pred_state} ({pred_year})",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        months_list = results_df["Month"].tolist()
        cases_list  = results_df["Predicted Cases"].tolist()
        fig2 = go.Figure(go.Scatterpolar(
            r=cases_list + [cases_list[0]],
            theta=months_list + [months_list[0]],
            fill="toself",
            line_color=PALETTE[0],
        ))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True)),
                           showlegend=False, title="Polar Case Distribution",
                           height=430)
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"Peak month: **{results_df.loc[results_df['Predicted Cases'].idxmax(), 'Month']}** "
                   f"with {results_df['Predicted Cases'].max():,} predicted cases")
