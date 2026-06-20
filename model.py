import re
import joblib
import nltk
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

nltk.download("stopwords")
nltk.download("punkt")

df = pd.read_csv("personality.csv")

df = df.drop(["ptype", "__index_level_0__", "random"], axis=1, errors="ignore")

df = df.sample(n=min(12000, len(df)), random_state=42)

ocean = ["O", "C", "E", "A", "N"]
df = df[~(df[ocean] == 0).all(axis=1)]
df = df.dropna(subset=["text"] + ocean)
df.reset_index(drop=True, inplace=True)

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

X_text = df["text"].apply(preprocess_text)
y = df[ocean]

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y_scaled,
    test_size=0.3,
    random_state=42
)

model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
)

print("Training model...")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, "personality_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "target_scaler.pkl")

print("Model, vectorizer, and scaler saved successfully.")