import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import logging
import fitz  # PyMuPDF for reading PDFs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

nltk.download('stopwords')
from nltk.corpus import stopwords
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

# Read all PDFs into text
legal_docs = [read_pdf(pdf_file) for pdf_file in pdf_files]

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in legal_docs]

# Convert text to TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)

# Apply K-means clustering
num_clusters = 3  # Number of clusters, adjust based on the data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Log the clustering results
logging.info(f"K-means Clustering Labels (Total Clusters: {num_clusters}):")
for idx, label in enumerate(labels):
    logging.info(f"Document {idx + 1} is in cluster {label}")

# Function to extract top keywords for each cluster
def extract_top_keywords_for_clusters(tfidf_matrix, feature_names, top_n=5):
    cluster_keywords = {}
    for i in range(num_clusters):
        # Get the indices of documents in this cluster
        cluster_indices = np.where(labels == i)[0]
        # Get the TF-IDF values for these documents
        cluster_tfidf_matrix = tfidf_matrix[cluster_indices]
        # Calculate the mean TF-IDF score for each word in this cluster
        mean_tfidf = np.mean(cluster_tfidf_matrix.toarray(), axis=0)
        # Get the top N keywords based on the mean TF-IDF scores
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        cluster_keywords[i] = top_keywords
    return cluster_keywords

# Extract top keywords for each cluster
top_keywords = extract_top_keywords_for_clusters(X, vectorizer.get_feature_names_out())

# Log the top keywords for each cluster
logging.info("Top Keywords for Each Cluster:")
for cluster, keywords in top_keywords.items():
    logging.info(f"Cluster {cluster}: {', '.join(keywords)}")

# Metrics to evaluate clustering quality

# 1. Silhouette Score
sil_score = silhouette_score(X, labels)
logging.info(f"Silhouette Score: {sil_score}")

# 2. Inertia
inertia = kmeans.inertia_
logging.info(f"Inertia: {inertia}")

# 3. Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(X.toarray(), labels)
logging.info(f"Davies-Bouldin Index: {db_index}")

# 4. Dunn Index (Requires Custom Implementation, example only)
# We would need to implement Dunn Index as a custom function, as it is not available in sklearn

# Summary of Clustering Results
logging.info("Clustering Summary:")
logging.info(f"Total number of clusters: {num_clusters}")
logging.info(f"Cluster sizes: {[np.sum(labels == i) for i in range(num_clusters)]}")
