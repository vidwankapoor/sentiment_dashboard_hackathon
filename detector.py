import json
from transformers import pipeline

# Load the emotion classifier model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Load email data
with open("emails.json", "r") as f:
    emails = json.load(f)

# Function to classify emotion
def detect_emotion(text):
    result = classifier(text)[0][0]['label'].lower()
    return result

# Process each email
for email in emails:
    emotion = detect_emotion(email["body"])
    print(f"From: {email['from']}")
    print(f"Subject: {email['subject']}")
    print(f"Emotion: {emotion}")
    print("-" * 40)
negative_emotions = ["anger", "sadness", "disgust", "fear"]
emotion_log = []

# Process each email
for email in emails:
    emotion = detect_emotion(email["body"])
    emotion_log.append(emotion)

    print(f"From: {email['from']}")
    print(f"Subject: {email['subject']}")
    print(f"Emotion: {emotion}")
    print("-" * 40)

# Alert check
negative_count = sum(1 for e in emotion_log if e in negative_emotions)

if negative_count >= 3:
    print("🚨 ALERT: High number of negative emails detected!")
else:
    print("✅ Sentiment under control.")