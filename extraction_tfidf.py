import re
import os
import nltk
import json
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import fitz  # PyMuPDF for reading PDFs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to read text from a PDF file
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

# Example of how to load legal documents from PDF files
pdf_files = [
    "akt_prawny_2020_1.pdf", 
    "akt_prawny_2020_2.pdf", 
    "akt_prawny_2020_3.pdf", 
    "akt_prawny_2020_4.pdf", 
    "akt_prawny_2020_5.pdf", 
    "akt_prawny_2020_6.pdf", 
    "akt_prawny_2020_7.pdf", 
    "akt_prawny_2020_8.pdf", 
    "akt_prawny_2020_9.pdf", 
    "akt_prawny_2020_10.pdf"
]

legal_docs = [read_pdf(pdf_file) for pdf_file in pdf_files]

# Preprocessing Function
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in legal_docs]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit the TF-IDF model and transform the documents
X = vectorizer.fit_transform(processed_docs)

# Labels - In a real case, you should have labels for classification. This is just an example.
# In this case, I'm assuming a binary classification problem, but you can adjust this as needed.
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Dummy binary labels for example (replace with real labels)

# Initialize the classifier
clf = SVC(kernel='linear')

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation evaluation
cross_val_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')

# Log the cross-validation results
logging.info(f"Cross-Validation Accuracy Scores: {cross_val_scores}")
logging.info(f"Mean Accuracy: {cross_val_scores.mean()}")

# Evaluate the model using F1-Score
cross_val_f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')

# Log the F1-Score results
logging.info(f"Cross-Validation F1-Score: {cross_val_f1_scores}")
logging.info(f"Mean F1-Score: {cross_val_f1_scores.mean()}")

# Fit the model to the entire dataset for final evaluation
clf.fit(X, y)

# You can now make predictions or evaluate on a separate test set if available
# Example: Make predictions on the training data itself (you can replace this with test data)
y_pred = clf.predict(X)

# Calculate F1-Score, Accuracy, etc.
final_f1_score = f1_score(y, y_pred, average='macro')
final_accuracy = accuracy_score(y, y_pred)

logging.info(f"Final F1-Score: {final_f1_score}")
logging.info(f"Final Accuracy: {final_accuracy}")

# Function to extract top keywords for each document based on TF-IDF scores
def extract_keywords_for_documents(tfidf_matrix, feature_names, top_n=5):
    keywords_per_document = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = row.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        keywords_per_document.append(top_keywords)
    return keywords_per_document

# Extract and log the top keywords for each document
top_keywords = extract_keywords_for_documents(X, vectorizer.get_feature_names_out())

# Log the keywords for each document
logging.info("Top Keywords for Each Document:")
for i, keywords in enumerate(top_keywords):
    logging.info(f"Document {i + 1}: {', '.join(keywords)}")
