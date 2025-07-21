import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Email Emotion Reports", layout="centered")
st.title("📊 Email Emotion Report")

DATA_FILE = "data.json"

# Load email data
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
else:
    st.warning("No email data found.")
    st.stop()

# Normalize into DataFrame
df = pd.json_normalize(data)

# Display table
st.subheader("📄 All Email Entries")
st.dataframe(df, use_container_width=True)

# Emotion Pie Chart
if "emotion" in df.columns:
    emotion_counts = df["emotion"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.subheader("🎯 Emotion Distribution")
    st.pyplot(fig)
else:
    st.info("No emotion data found in entries.")

# Download CSV
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv_data, "email_reports.csv", "text/csv")

