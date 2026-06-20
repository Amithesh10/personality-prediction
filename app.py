import re
import joblib
import nltk
import streamlit as st
import plotly.graph_objects as go

from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

model = joblib.load("personality_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("target_scaler.pkl")

tk = TweetTokenizer()
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text)
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    tokens = tk.tokenize(text)
    tokens = [word for word in tokens if len(word) >= 3]
    tokens = [stemmer.stem(word.lower()) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def clamp_score(score):
    return max(0, min(100, round(float(score), 2)))

st.set_page_config(
    page_title="Personality Prediction",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e3a8a 100%);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #93c5fd;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #bfdbfe;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: rgba(15, 23, 42, 0.95);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #1d4ed8;
}

.info-box {
    background: rgba(30, 64, 175, 0.25);
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #3b82f6;
    color: #dbeafe;
}

.dominant-card {
    background: linear-gradient(135deg, #1e3a8a, #2563eb, #38bdf8);
    padding: 32px;
    border-radius: 25px;
    text-align: center;
    margin-top: 20px;
}

.insight-card {
    background: rgba(15, 23, 42, 0.95);
    padding: 25px;
    border-radius: 20px;
    border-left: 6px solid #3b82f6;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #93c5fd;
    margin-top: 35px;
    font-size: 14px;
}

.stTextArea textarea {
    background-color: #020617;
    color: white;
    border-radius: 14px;
    border: 1px solid #2563eb;
}

.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 14px;
    font-weight: 700;
    background: linear-gradient(90deg, #1d4ed8, #2563eb, #38bdf8);
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background-color: rgba(15, 23, 42, 0.95);
    border: 1px solid #2563eb;
    padding: 18px;
    border-radius: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #0f172a;
    border-radius: 12px;
    padding: 10px 18px;
    color: #bfdbfe;
    border: 1px solid #1d4ed8;
}

.stTabs [aria-selected="true"] {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Personality Prediction from Writing Style</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered Big Five personality analysis using NLP and Machine Learning</div>', unsafe_allow_html=True)

left, right = st.columns([1.3, 1])

with left:
    st.markdown("""
    <div class="card">
        <h3>✍️ Enter Your Writing Sample</h3>
        <p style="color:#bfdbfe;">
        Write a paragraph about your thoughts, goals, interests, or daily life.
        The system analyzes your writing pattern and predicts your OCEAN personality traits.
        </p>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "Writing Sample",
        height=240,
        placeholder="Example: I enjoy learning new technologies, solving complex problems, and exploring creative ideas..."
    )

    predict = st.button("🔍 Analyze Personality")

with right:
    st.markdown("""
    <div class="card">
        <h3>📌 About This Project</h3>
        <p style="color:#bfdbfe;">
        This project predicts personality traits from text using Natural Language Processing.
        It uses TF-IDF vectorization and a Machine Learning regression model.
        </p>
        <div class="info-box">
        <b>Framework:</b> OCEAN Personality Model<br>
        <b>Input:</b> User-written text<br>
        <b>Output:</b> Five trait scores
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🛠 Technologies Used</h3>
        <p style="color:#bfdbfe;">
        Python • Streamlit • Scikit-Learn • Pandas • NumPy • NLTK • TF-IDF • Joblib • Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)

if predict:
    if text.strip() == "":
        st.warning("Please enter some text before prediction.")
    else:
        cleaned_text = preprocess_text(text)
        vector = vectorizer.transform([cleaned_text])

        pred_scaled = model.predict(vector)
        pred = scaler.inverse_transform(pred_scaled)

        scores = {
            "Openness": clamp_score(pred[0][0]),
            "Conscientiousness": clamp_score(pred[0][1]),
            "Extraversion": clamp_score(pred[0][2]),
            "Agreeableness": clamp_score(pred[0][3]),
            "Neuroticism": clamp_score(pred[0][4]),
        }

        descriptions = {
            "Openness": "Creativity, imagination, curiosity, and willingness to explore new ideas.",
            "Conscientiousness": "Discipline, organization, responsibility, and goal-oriented behavior.",
            "Extraversion": "Sociability, confidence, energy, and comfort in social situations.",
            "Agreeableness": "Kindness, cooperation, empathy, and compassion toward others.",
            "Neuroticism": "Emotional sensitivity, stress response, and mood variation."
        }

        trait_summary = {
            "Openness": "You are curious, creative, and open to exploring new ideas, technologies, and experiences.",
            "Conscientiousness": "You tend to be organized, disciplined, responsible, and focused on achieving your goals.",
            "Extraversion": "You appear energetic, expressive, confident, and comfortable interacting with people.",
            "Agreeableness": "You show empathy, cooperation, kindness, and a strong concern for others.",
            "Neuroticism": "You may be emotionally sensitive and responsive to stress, pressure, or uncertainty."
        }

        icons = {
            "Openness": "🌟",
            "Conscientiousness": "📋",
            "Extraversion": "🗣",
            "Agreeableness": "🤝",
            "Neuroticism": "💭"
        }

        dominant_trait = max(scores, key=scores.get)
        dominant_score = scores[dominant_trait]

        st.markdown("---")
        st.markdown("## 📊 Personality Results Dashboard")

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.metric("🌟 Openness", f"{scores['Openness']}/100")
        with k2:
            st.metric("📋 Conscientiousness", f"{scores['Conscientiousness']}/100")
        with k3:
            st.metric("🗣 Extraversion", f"{scores['Extraversion']}/100")
        with k4:
            st.metric("🤝 Agreeableness", f"{scores['Agreeableness']}/100")
        with k5:
            st.metric("💭 Neuroticism", f"{scores['Neuroticism']}/100")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "🌟 Openness",
            "📋 Conscientiousness",
            "🗣 Extraversion",
            "🤝 Agreeableness",
            "💭 Neuroticism"
        ])

        with tab1:
            st.markdown("### Overall Personality Profile")

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=list(scores.values()),
                theta=list(scores.keys()),
                fill="toself",
                name="Personality Score"
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(15,23,42,0.8)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                font=dict(color="white"),
                showlegend=False
            )

            st.plotly_chart(fig, width="stretch")

            st.markdown(f"""
            <div class="dominant-card">
                <h1 style="color:white;font-size:42px;">🏆 Dominant Personality Trait</h1>
                <h2 style="color:#f8fafc;font-size:38px;">{icons[dominant_trait]} {dominant_trait}</h2>
                <h3 style="color:#dbeafe;font-size:28px;">Score: {dominant_score}/100</h3>
                <p style="font-size:18px;color:white;max-width:760px;margin:auto;">
                    This trait appears most strongly in your writing style and has the greatest influence on your predicted personality profile.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-card">
                <h3 style="color:#93c5fd;">🧠 AI Personality Insight</h3>
                <p style="font-size:18px;color:white;">
                    {trait_summary[dominant_trait]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.info("This prediction is based on writing patterns and is intended for learning and demonstration purposes.")

        trait_tabs = {
            tab2: "Openness",
            tab3: "Conscientiousness",
            tab4: "Extraversion",
            tab5: "Agreeableness",
            tab6: "Neuroticism"
        }

        for tab, trait in trait_tabs.items():
            with tab:
                st.markdown(f"## {icons[trait]} {trait}")
                st.metric("Score", f"{scores[trait]}/100")
                st.progress(int(scores[trait]))

                st.markdown(f"""
                <div class="card">
                    <h3>What this trait means</h3>
                    <p style="color:#bfdbfe;">{descriptions[trait]}</p>
                </div>
                """, unsafe_allow_html=True)

                if scores[trait] >= 70:
                    level = "High"
                    message = f"You show a strong level of {trait.lower()} based on your writing."
                elif scores[trait] >= 40:
                    level = "Moderate"
                    message = f"You show a balanced level of {trait.lower()} based on your writing."
                else:
                    level = "Low"
                    message = f"You show a lower level of {trait.lower()} based on your writing."

                st.info(f"Level: {level}")
                st.write(message)

st.markdown("""
<div class="footer">
    Developed by Amithesh T S | AI & Machine Learning Portfolio Project
</div>
""", unsafe_allow_html=True)