import streamlit as st
import pandas as pd
from transformers import pipeline
import re
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Multilingual Sentiment Analyzer", layout="centered")
st.title("🌍 Multilingual Sentiment Analysis (English + Arabic)")

# Load sentiment pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="bert-base-multilingual-uncased", tokenizer="bert-base-multilingual-uncased")

classifier = load_model()

# -------------------------
# 🔹 Text Input Section
# -------------------------
st.header("📥 Analyze a Single Review")

text = st.text_area("Enter review text (English or Arabic):", height=100)

if st.button("Analyze"):
    if text.strip():
        result = classifier(text)[0]
        label_map = {"LABEL_0": "Positive", "LABEL_1": "Negative"}
        pred_label = label_map.get(result["label"], result["label"])
        st.success(f"Sentiment: **{pred_label}** (Confidence: {round(result['score'], 2)})")
    else:
        st.warning("Please enter some text.")

# -------------------------
# 🔹 File Upload Section
# -------------------------
st.header("📤 Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a column named 'text'.")
    else:
        st.info("Analyzing sentiments... please wait ⏳")

        label_map = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
        df['prediction'] = df['text'].apply(lambda x: label_map.get(classifier(str(x))[0]['label']))

        st.subheader("📊 Sentiment Counts")
        sentiment_counts = df['prediction'].value_counts()
        st.bar_chart(sentiment_counts)

        # Word Clouds per sentiment
        st.subheader("☁️ Word Clouds by Sentiment")

        for sentiment in sentiment_counts.index:
            text_blob = " ".join(df[df['prediction'] == sentiment]['text'].astype(str))

            # Handle Arabic reshaping
            is_arabic = any(re.search(r'[\u0600-\u06FF]', t) for t in text_blob)
            if is_arabic:
                text_blob = get_display(arabic_reshaper.reshape(text_blob))

            wc = WordCloud(width=800, height=300, background_color="white", font_path="arial").generate(text_blob)
            st.image(wc.to_array(), caption=f"Word Cloud: {sentiment}")

        st.subheader("📄 Full Output")
        st.dataframe(df[['text', 'prediction']])
