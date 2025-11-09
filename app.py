import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="🌾 Sensor-Free Agri Digital Twin", layout="wide")
st.title("🌾 Sensor-Free Agricultural Digital Twin Dashboard")

DATA_PATH = "data/smart_agri.csv"

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load dataset: {e}")
    st.stop()

# Clean numeric columns
df = df.applymap(lambda x: str(x).replace('-', '') if isinstance(x, str) else x)
df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
df['soilmiosture'] = pd.to_numeric(df['soilmiosture'], errors='coerce')
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')

df = df.dropna(subset=['temperature', 'pressure', 'altitude', 'soilmiosture'])

st.write("### Sample of Data")
st.dataframe(df.head())

# ----------------------------
# VISUALIZATION SECTION
# ----------------------------
st.subheader("📊 Environmental Parameter Trends")

col1, col2 = st.columns(2)
with col1:
    st.line_chart(df[['temperature', 'pressure']])
with col2:
    st.line_chart(df[['altitude', 'soilmiosture']])

# Correlation heatmap
st.subheader("📈 Feature Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df[['temperature', 'pressure', 'altitude', 'soilmiosture']].corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# ----------------------------
# MODEL SECTION
# ----------------------------
st.subheader("🤖 Digital Twin Model Training")

# Input and target
X = df[['temperature', 'pressure', 'altitude']].values
y = df['soilmiosture'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define neural network
class AgriNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = AgriNN(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Evaluate
with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

y_pred = scaler_y.inverse_transform(preds)
y_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.metric("📏 Mean Absolute Error", f"{mae:.2f}")
st.metric("🎯 R² Score", f"{r2:.2f}")

# ----------------------------
# VISUALIZE MODEL RESULTS
# ----------------------------
st.subheader("📉 True vs Predicted Soil Moisture")

fig2, ax2 = plt.subplots()
ax2.plot(y_true, label="True", marker='o')
ax2.plot(y_pred, label="Predicted", marker='x')
ax2.set_xlabel("Sample")
ax2.set_ylabel("Soil Moisture")
ax2.legend()
st.pyplot(fig2)

# ----------------------------
# FEATURE IMPORTANCE (Explainability)
# ----------------------------
st.subheader("🔍 Feature Importance (Weight Magnitude)")
weights = model.layers[0].weight.detach().numpy().mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': ['Temperature', 'Pressure', 'Altitude'],
    'Importance': np.abs(weights)
}).sort_values(by='Importance', ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax3, palette='viridis')
ax3.set_title("Feature Importance based on Model Weights")
st.pyplot(fig3)

st.success("✅ Digital Twin model built and analyzed successfully!")
