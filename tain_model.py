import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from textblob import TextBlob
import re
import numpy as np

# Download NLTK resources (do this ONCE)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# 1. Load/Create Training Data
training_data = [
    "This is a real review.",
    "This is another real review.",
    "This is a fake review.",
    "This is another fake review."
]
labels = [0, 0, 1, 1]

# 2. Create and Fit Vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(training_data)

# 3. Vectorize Training Data
training_vectors = vectorizer.transform(training_data)

# *** Crucial print statements ***
X = training_vectors.toarray()  # Convert to dense NumPy array

print("X shape (training):", X.shape)
print("X type (training):", type(X))
print("X dtype (training):", X.dtype)

# Define extract_features function here (WITH CORRECT INDENTATION)
def extract_features(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Stylistic Features (Example)
    exclamation_points = text.count("!")
    capitalized_words = len(re.findall(r"[A-Z]", text))  # Count capitalized words

    return [sentiment, subjectivity, exclamation_points, capitalized_words]

# Extract features for training data
training_features = [extract_features(text) for text in training_data]
training_features_array = np.array(training_features)  # Convert to NumPy array

# Combine vectorized text data with extracted features
X_combined = np.hstack([X, training_features_array])

print("X_combined shape (training):", X_combined.shape)
print("X_combined type (training):", type(X_combined))
print("X_combined dtype (training):", X_combined.dtype)

# 4. Train Model
model = LogisticRegression()
model.fit(X_combined, labels)  # Use X_combined for training

# 5. Save Model and Vectorizer
joblib.dump(model, 'your_model.pkl')
joblib.dump(vectorizer, 'your_vectorizer.pkl')

print("Model and vectorizer saved successfully.")