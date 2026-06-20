# Personality Prediction from Writing Style

## Overview

This project predicts a person's personality traits based on their writing style using Machine Learning and Natural Language Processing (NLP). The system analyzes textual input and predicts personality scores based on the OCEAN (Big Five) personality model.

The OCEAN model consists of:

* **Openness** – Creativity and willingness to try new experiences
* **Conscientiousness** – Organization and responsibility
* **Extraversion** – Sociability and outgoing nature
* **Agreeableness** – Cooperation and compassion
* **Neuroticism** – Emotional stability and stress sensitivity

---

## Features

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Machine Learning-based personality prediction
* OCEAN personality score generation
* Interactive user interface using Streamlit
* Real-time prediction from user text input

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* TF-IDF Vectorization
* Streamlit
* Joblib

---

## Dataset

The project uses the Pandora Personality Dataset, which contains user-generated text along with personality trait information based on the Big Five personality framework.

---

## Workflow

1. Data Collection
2. Text Preprocessing

   * Lowercasing
   * Removing punctuation
   * Removing stopwords
   * Tokenization
3. Feature Extraction using TF-IDF
4. Model Training
5. Personality Prediction
6. Result Visualization

---

## Model Development

The model was trained using Machine Learning techniques on processed textual data.

Steps included:

* Data Cleaning
* Feature Engineering
* TF-IDF Vectorization
* Train-Test Split
* Model Training
* Performance Evaluation

---

## Project Structure

```text
Personality-Prediction/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── dataset.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/personality-prediction.git
cd personality-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Sample Input

```text
I enjoy learning new technologies and exploring innovative ideas. I like solving complex problems and working on challenging projects.
```

---

## Sample Output

```text
Openness: 78
Conscientiousness: 71
Extraversion: 65
Agreeableness: 69
Neuroticism: 32
```

---

## Applications

* Personality Assessment
* Human Resource Analytics
* Career Guidance
* Psychological Research
* User Behavior Analysis

---

## Future Enhancements

* Deep Learning-based prediction models
* Support for multiple languages
* Advanced sentiment analysis integration
* Improved personality visualization dashboards
* Real-time personality tracking

---

## Author

**Amithesh T S**

AI & Machine Learning Engineer | Data Science Enthusiast

LinkedIn: [www.linkedin.com/in/amithesh-ts](http://www.linkedin.com/in/amithesh-ts)

GitHub: https://github.com/Amithesh10
