from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import nltk
import numpy as np
from textblob import TextBlob
import re
from scipy.sparse import hstack

app = Flask(__name__)

try:
    model = joblib.load('your_model.pkl')
    vectorizer = joblib.load('your_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
    print("Vectorizer Vocabulary:", vectorizer.vocabulary_)

except FileNotFoundError:
    print("Model or vectorizer files not found.")
    exit()
except Exception as e:
    print(f"ERROR loading model/vectorizer: {e}")
    exit()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    return text.lower()  # Or your actual cleaning function

def generate_reasons(text, prediction):
    reasons = []
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if prediction == 1:  # Fake
        if sentiment > 0.5:
            reasons.append("Review sentiment is very positive, which can sometimes indicate inauthenticity.")
        elif sentiment < -0.5:
            reasons.append("Review sentiment is very negative, which can sometimes indicate inauthenticity.")
        # ... other reasons for fake reviews ...
    elif prediction == 0:  # Real
        reasons.append("Review appears genuine.")  # Provide a default message for real reviews
    return reasons

def extract_features(text): # <--- This is the crucial missing function
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Stylistic Features (Example)
    exclamation_points = text.count("!")
    capitalized_words = len(re.findall(r"[A-Z]", text))  # Count capitalized words

    return [sentiment, subjectivity, exclamation_points, capitalized_words]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    try:
        review_cleaned = clean_text(review)
        review_vectorized = vectorizer.transform([review_cleaned])

        features = extract_features(review_cleaned)  # Now this function is defined
        review_features = hstack([review_vectorized, np.array([features])])
        review_features = review_features.toarray()  # Keep as float64

        print("review_features shape (prediction):", review_features.shape)
        print("review_features type (prediction):", type(review_features))
        print("review_features dtype (prediction):", review_features.dtype)
        print("review_features:\n", review_features)

        prediction = model.predict(review_features)[0]
        reasons = generate_reasons(review_cleaned, prediction)
        return jsonify({'result': {'prediction': "Fake" if prediction == 1 else "Real", 'reasons': reasons}})

    except FileNotFoundError:
        return jsonify({'error': 'Model or vectorizer files not found.'}), 404
    except ValueError:
        return jsonify({'error': 'Invalid input data.'}), 400
    except Exception as e:
        print(f"ERROR in /analyze route: {e}")
        return jsonify({'error': 'An error occurred during analysis.'}), 500

if __name__ == '__main__':
    app.run(debug=True)