import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# --- App Title ---
st.title("🌍 Life Expectancy Predictor (2-Feature KNN Model)")

st.markdown("""
This app predicts life expectancy using two predictors from WHO 2014 data:

- **Income Composition of Resources** (index from 0 to 1)
- **Score_Edu_HIV**: a composite score = Years of Schooling / HIV/AIDS child mortality (0-4) years old

The model uses K-Nearest Neighbors with **k=4**, trained on 2014 country-level data.
""")

# --- User Inputs ---
income_comp = st.slider("💰 Income Composition of Resources (0–1)", 0.0, 1.0, 0.6, step=0.01)
score_edu_hiv = st.number_input("📊 Score_Edu_HIV (Schooling / HIV/AIDS)", min_value=0.1, max_value=20.0, value=2.5)

# --- Prediction Input ---
X_input = pd.DataFrame({
    "Income composition of resources": [income_comp],
    "Score_Edu_HIV": [score_edu_hiv]
})

# --- Load Data & Train Model ---
df = pd.read_csv("/Users/kamisama/Desktop/intro/Life Expectancy Data.csv")
df = df[df["Year"] == 2014].copy()
df = df.rename(columns={" HIV/AIDS": "HIV/AIDS"})
df = df.dropna(subset=["Schooling", "Income composition of resources", "HIV/AIDS", "Life expectancy "])
df["Score_Edu_HIV"] = df["Schooling"] / df["HIV/AIDS"]

X = df[["Income composition of resources", "Score_Edu_HIV"]]
y = df["Life expectancy "]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=4))
])
pipeline.fit(X, y)

# --- Predict ---
predicted = pipeline.predict(X_input)[0]

# --- Display Prediction ---
st.subheader("🧮 Predicted Life Expectancy:")
st.success(f"{predicted:.2f} years")
