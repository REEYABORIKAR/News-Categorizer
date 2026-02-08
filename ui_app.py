import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.append(os.getcwd())

try:
    from src.inference.predictor import Predictor
except ImportError:
    st.error("❌ Could not find 'src.inference.predictor'. Ensure your folder structure is correct!")

st.set_page_config(
    page_title="NewsLine AI | News Classifier",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stTextArea textarea { border-radius: 12px; border: 1px solid #e5e7eb; padding: 15px; }
    .predict-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-top: 5px solid #6366f1;
    }
    .category-label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
    .category-value { color: #111827; font-size: 1.8rem; font-weight: 800; margin-bottom: 8px; }
    .confidence-bar { background-color: #ecfdf5; color: #065f46; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return Predictor()

with st.spinner("🚀 Loading AI Engine..."):
    predictor = load_model()

with st.sidebar:
    st.title("Settings & Info")
    st.markdown("---")
    st.markdown("### 📊 Capabilities")
    st.write("This engine is trained to identify news patterns across 8+ global categories.")
    
    st.info("**Tech Stack:**\n- Python / Streamlit\n- Scikit-Learn\n- Deployment: Production Ready")
    
    st.markdown("---")
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

st.markdown("# 📰 News Intelligence Dashboard")
st.markdown("##### Transform unstructured news text into actionable categorical data.")

st.markdown("### 📝 Article Input")
examples = [
        "",
        "The government passed a new education reform bill in parliament today.",
        "The prime minister announced major policy changes ahead of the elections.",
        "World leaders met at the climate summit to discuss global carbon emission targets.",
        "The United Nations held an emergency meeting over the international crisis.",
        "The stock market rallied after the company reported record quarterly profits.",
        "The central bank announced new interest rate policies to control inflation.",
        "Google unveiled a new AI model for real-time language translation.",
        "Apple launched a new AI-powered smartphone.",
        "The football team won the championship in a thrilling final.",
        "The cricket captain scored a century to lead his team to victory.",
        "Scientists discovered a new exoplanet in a distant galaxy.",
        "Researchers developed a breakthrough vaccine using novel gene-editing techniques.",
        "The company reported record profits this quarter.",
        "The new superhero movie broke box office records in its opening weekend.",
        "The popular web series was renewed for another season after fan demand.",
        "Experts shared tips on healthy eating and daily exercise routines.",
        "Travel bloggers recommended the top destinations to visit this summer.",
        "The government announced funding for new AI research programs.",
        "A famous athlete invested in a new tech startup.",
        "The movie explores climate change and its global impact."
    ]
selected_example = st.selectbox("Quick-test examples:", examples)
default_text = "" if selected_example == examples[0] else selected_example

input_text = st.text_area(
    label="Paste headline or full article text here:",
    value=default_text,
    height=180,
    placeholder="Waiting for input..."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("Analyze Content", use_container_width=True, type="primary")

if predict_btn:
    if input_text.strip():
        with st.spinner("Classifying..."):
            label, confidence, top3 = predictor.predict_with_topk(input_text, k=3)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(f"""
            <div class="predict-card">
                <div class="category-label">Primary Classification</div>
                <div class="category-value">{label}</div>
                <span class="confidence-bar">Confidence: {confidence*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            df_plot = pd.DataFrame(top3, columns=['Category', 'Score'])
            fig = px.bar(
                df_plot, 
                x='Score', 
                y='Category', 
                orientation='h',
                title="Top-3 Confidence Scores",
                color='Score',
                color_continuous_scale='GnBu',
                text_auto='.2f'
            )
            fig.update_layout(height=230, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please provide news text to analyze.")

st.markdown("---")
