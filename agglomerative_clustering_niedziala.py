import re
import os
import nltk
import json
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
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

# Standardize the features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.toarray())

# Apply Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3)
y_pred = agg_clust.fit_predict(X_scaled)

# Logging the number of clusters found and the assignments
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
logging.info(f"Number of clusters: {n_clusters}")

logging.info("Cluster assignments:")
for i, label in enumerate(y_pred):
    logging.info(f"Document {i + 1} is in cluster {label}")

# Function to extract top keywords for each cluster
def extract_keywords_for_clusters(tfidf_matrix, feature_names, labels, top_n=5):
    keywords_per_cluster = {}
    for cluster in set(labels):
        # Get the indices of the documents in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
        # Sum the TF-IDF values for all documents in the cluster
        cluster_tfidf_values = np.sum(tfidf_matrix[cluster_indices], axis=0).A1
        # Get the top N keywords for this cluster
        top_indices = cluster_tfidf_values.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        keywords_per_cluster[cluster] = top_keywords
    return keywords_per_cluster

# Extract and log the top keywords for each cluster
top_keywords = extract_keywords_for_clusters(X, vectorizer.get_feature_names_out(), y_pred)

# Log the keywords for each cluster
logging.info("Top Keywords for Each Cluster:")
for cluster, keywords in top_keywords.items():
    logging.info(f"Cluster {cluster}: {', '.join(keywords)}")

# Metrics calculation
# Since clustering is unsupervised, we can't compute typical classification metrics like accuracy or F1-score
# But we can use clustering evaluation metrics like purity or silhouette score
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(X_scaled, y_pred)
logging.info(f"Silhouette Score: {silhouette}")
