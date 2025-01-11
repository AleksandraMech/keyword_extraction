import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import fitz  # PyMuPDF for reading PDFs
import logging
from sklearn.preprocessing import StandardScaler

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

# Example PDF documents
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
    "akt_prawny_2020_10.pdf",
    "akt_prawny_2020_11.pdf", 
    "akt_prawny_2020_12.pdf", 
    "akt_prawny_2020_13.pdf", 
    "akt_prawny_2020_14.pdf", 
    "akt_prawny_2020_15.pdf", 
    "akt_prawny_2020_16.pdf", 
    "akt_prawny_2020_17.pdf", 
    "akt_prawny_2020_18.pdf", 
    "akt_prawny_2020_19.pdf", 
    "akt_prawny_2020_20.pdf"
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

# Standardize the feature vectors (important for DBSCAN)
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=1.0, min_samples=2, metric='euclidean')
y_pred = db.fit_predict(X_scaled)

# Count the number of clusters found
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
logging.info(f"Number of clusters: {n_clusters}")

# Logging the clusters assigned to each document
logging.info("Cluster assignments:")
for i, label in enumerate(y_pred):
    logging.info(f"Document {i + 1} is in cluster {label}")

# Function to extract top keywords for each cluster based on TF-IDF scores
def extract_keywords_for_clusters(tfidf_matrix, feature_names, labels, top_n=5):
    cluster_keywords = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_tfidf = tfidf_matrix[cluster_indices]
        mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        cluster_keywords[cluster_id] = top_keywords
    return cluster_keywords

# Extract top keywords for each cluster
top_keywords = extract_keywords_for_clusters(X, vectorizer.get_feature_names_out(), y_pred)

# Log the keywords for each cluster
logging.info("Top Keywords for Each Cluster:")
for cluster_id, keywords in top_keywords.items():
    logging.info(f"Cluster {cluster_id}: {', '.join(keywords)}")

# Evaluate the clustering quality
# Since this jest unsupervised learning, you can't directly calculate accuracy.
# You might want to use metrics like silhouette score, or compare clusters to known labels if available.

# Optional: If you have true labels for validation, you can compute classification metrics
# If you had labels for documents (y_true), you could calculate F1-Score, Precision, etc.
# For now, this step is omitted because the task is unsupervised.
