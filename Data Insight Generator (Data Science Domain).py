# Project: Data Insight Generator (Data Science Domain)

# === MAIN APP ===
# File: app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
from sklearn.ensemble import IsolationForest

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_insights(df):
    prompt = f"""
    You are a senior data analyst. Analyze this dataset and:
    - Summarize key insights
    - Identify patterns
    - Point out any unusual or interesting trends
    - Give suggestions on how the dataset could be used in real-world business applications

    Dataset preview:
    {df.head(10).to_string(index=False)}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def detect_anomalies(df):
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        return pd.DataFrame(), "Not enough numeric data to detect anomalies."
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(num_df)
    df['Anomaly'] = clf.predict(num_df)
    anomalies = df[df['Anomaly'] == -1]
    return anomalies, f"Detected {len(anomalies)} anomalies."

def download_insights(insights_text):
    with open("insights.txt", "w") as f:
        f.write(insights_text)
    return "insights.txt"

st.set_page_config(page_title="📊 Data Insight Generator", layout="wide")
st.title("📊 AI-Powered Data Insight Generator")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Generate Plots")
    num_cols = df.select_dtypes(include='number').columns

    if len(num_cols) >= 2:
        chart_type = st.selectbox("Choose Chart Type", ["Scatter", "Histogram", "Box", "Bar"])
        x_axis = st.selectbox("X-axis", num_cols)
        y_axis = st.selectbox("Y-axis", num_cols, index=1)

        if chart_type == "Scatter":
            chart = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Histogram":
            chart = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}")
        elif chart_type == "Box":
            chart = px.box(df, y=y_axis, x=x_axis, title=f"Box plot of {y_axis} by {x_axis}")
        elif chart_type == "Bar":
            chart = px.bar(df, x=x_axis, y=y_axis, title=f"Bar chart of {y_axis} by {x_axis}")

        st.plotly_chart(chart)
    else:
        st.warning("Not enough numeric columns for plotting.")

    st.subheader("📊 AI-Generated Insights")
    if st.button("Generate Insights with GPT"):
        with st.spinner("Generating insights using GPT-4..."):
            insights = generate_insights(df)
            st.success("Insights generated successfully!")
            st.markdown(insights)
            file_path = download_insights(insights)
            st.download_button("📥 Download Insights as TXT", data=open(file_path).read(), file_name="insights.txt")

    st.subheader("🚨 Anomaly Detection")
    if st.button("Detect Anomalies"):
        with st.spinner("Running anomaly detection..."):
            anomalies, message = detect_anomalies(df)
            st.success(message)
            if not anomalies.empty:
                st.dataframe(anomalies)
else:
    st.info("Please upload a CSV file to get started.")
