import streamlit as st
import json
import pandas as pd
from transformers import pipeline
import os
from datetime import datetime

# -------------- File Config --------------
DATA_FILE = "data.json"

# -------------- Helper Functions --------------
def load_data():
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    else:
        return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -------------- App Config --------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
emoji_map = {
    "anger": "😠", "joy": "😊", "sadness": "😢",
    "fear": "😨", "disgust": "🤢", "surprise": "😲", "neutral": "😐"
}

# -------------- Sidebar --------------
st.sidebar.title("📊 Dashboard")
st.sidebar.markdown("Welcome, Izummi!")
st.sidebar.markdown("---")
st.sidebar.write("📬 Inbox")
st.sidebar.write("📈 Reports")
st.sidebar.write("⚙️ Settings")

# -------------- Title --------------
st.title("Customer Sentiment Dashboard")
st.caption("Monitor customer support sentiment in real-time")

# -------------- Email Submission --------------
st.markdown("### 📨 Submit Your Email for Emotion Analysis")
with st.form("user_email_form"):
    user_email = st.text_input("Your Email Address")
    subject = st.text_input("Subject")
    message = st.text_area("Email Message", height=120)
    submitted = st.form_submit_button("🔍 Analyze Now")

if submitted:
    if message.strip():
        detected_emotion = classifier(message)[0][0]['label'].lower()
        emoji = emoji_map.get(detected_emotion, "💬")

        new_email = {
            "sender": user_email or "anonymous",
            "subject": subject or "No Subject",
            "body": message,
            "emotion": detected_emotion,
            "timestamp": str(datetime.now())
        }

        data = load_data()
        data.append(new_email)
        save_data(data)

        st.success(f"Emotion Detected: {emoji} {detected_emotion.capitalize()}")
        with st.expander("📬 Submitted Email"):
            st.markdown(f"**From:** {new_email['sender']}  \n**Subject:** {new_email['subject']}  \n**Emotion:** {emoji} {detected_emotion.capitalize()}")
            st.markdown(f"**Message:**\n{new_email['body']}")
    else:
        st.warning("Please enter an email body to analyze.")

# -------------- Load All Emails (User + Display) --------------
emails = load_data()
processed = []
negative_emotions = ["anger", "sadness", "disgust", "fear"]

for email in emails:
    urgency = "High" if email["emotion"].lower() in negative_emotions else "Low"
    processed.append({
        "from": email["sender"],
        "subject": email["subject"],
        "body": email["body"],
        "emotion": email["emotion"].capitalize(),
        "urgency": urgency
    })

# -------------- Display Saved Emails --------------
st.markdown("### 📁 Saved Emails")
for email in emails[-5:]:  # Show latest 5
    emoji = emoji_map.get(email['emotion'].lower(), "💬")
    st.write(f"**From:** {email['sender']} | **Subject:** {email['subject']} | **Emotion:** {emoji} {email['emotion'].capitalize()} | {email['timestamp']}")
    st.markdown("---")

# -------------- Metrics --------------
total_emails = len(processed)
critical_alerts = sum(1 for e in processed if e["urgency"] == "High")
positive_feedback = sum(1 for e in processed if e["emotion"].lower() == "joy")
avg_response = "2.4h"  # Placeholder

col1, col2, col3, col4 = st.columns(4)
col1.metric("Emails Analyzed", str(total_emails))
col2.metric("Critical Alerts", str(critical_alerts))
col3.metric("Avg. Response Time", avg_response)
col4.metric("Positive Feedback", str(positive_feedback))

# -------------- Search + Table UI --------------
st.markdown("#### Recent Email Insights")
colf1, colf2 = st.columns([4, 1])
colf1.text_input("🔍 Search emails, subjects, or senders")
colf2.selectbox("Sort By", ["Newest First", "Oldest First"])

# -------------- Show Table --------------
df = pd.DataFrame(processed)
if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("No email data to display.")
