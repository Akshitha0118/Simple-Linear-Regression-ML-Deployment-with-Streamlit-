
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Load Model & Dataset
# -----------------------------
with open("Linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

dataset = pd.read_csv("Salary_Data.csv")

X = dataset[['YearsExperience']]
y = dataset['Salary']

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Salary Predictor Pro", page_icon="📊", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 class='title'>📊 Salary Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>ML + Statistics + Visualization</p>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("🔢 Input Features")
experience = st.sidebar.slider("Years of Experience", 0.0, 20.0, 5.0, 0.5)

# -----------------------------
# Prediction
# -----------------------------
exp_array = np.array([[experience]])
prediction = model.predict(exp_array)[0]

# -----------------------------
# Confidence Interval (95%)
# -----------------------------
y_pred_all = model.predict(X)
residuals = y - y_pred_all
std_error = np.std(residuals)

confidence_interval = 1.96 * std_error
lower = prediction - confidence_interval
upper = prediction + confidence_interval

# -----------------------------
# Metric Cards
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(
    f"<div class='card'>💰<br><span>Predicted Salary</span><br>₹ {prediction:,.0f}</div>",
    unsafe_allow_html=True
)

col2.markdown(
    f"<div class='card'>📉<br><span>Lower Bound (95%)</span><br>₹ {lower:,.0f}</div>",
    unsafe_allow_html=True
)

col3.markdown(
    f"<div class='card'>📈<br><span>Upper Bound (95%)</span><br>₹ {upper:,.0f}</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Charts Section
# -----------------------------
st.markdown("## 📈 Salary vs Experience")

fig, ax = plt.subplots()
ax.scatter(X, y, label="Actual Data")
ax.plot(X, y_pred_all, color="red", label="Regression Line")
ax.scatter(experience, prediction, color="green", s=100, label="Your Prediction")
ax.legend()
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")

st.pyplot(fig)

# -----------------------------
# Statistical Insights
# -----------------------------
st.markdown("## 🧠 Statistical Insights")

r2 = model.score(X, y)
mean_salary = y.mean()
std_salary = y.std()
z_score = (prediction - mean_salary) / std_salary

col4, col5, col6, col7 = st.columns(4)

col4.metric("R² Score", f"{r2:.3f}")
col5.metric("Mean Salary", f"₹ {mean_salary:,.0f}")
col6.metric("Std Deviation", f"₹ {std_salary:,.0f}")
col7.metric("Z-Score", f"{z_score:.2f}")

# -----------------------------
# Interpretation
# -----------------------------
st.markdown("### 📌 Interpretation")
st.write(
    f"""
- **Predicted salary** for **{experience} years** experience is **₹ {prediction:,.0f}**
- With **95% confidence**, salary lies between **₹ {lower:,.0f} and ₹ {upper:,.0f}**
- Z-Score of **{z_score:.2f}** shows how far this salary is from average
"""
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)


