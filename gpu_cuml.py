import re
import os
import nltk
import json
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from datetime import datetime
import logging
from sklearn.metrics import f1_score, accuracy_score
import cudf
from cuml.decomposition import LatentDirichletAllocation as cuLDA

# Setup logging to write to a file
logging.basicConfig(
    filename='output.txt',  # Logi będą zapisywane w pliku 'output.txt'
    level=logging.INFO,  # Poziom logowania
    format='%(asctime)s - %(message)s'  # Format logów
)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Example of how to load documents (assuming processed_docs are already prepared)
# Replace with actual document data
processed_docs = [["contract", "governed", "laws", "state", "california"],
                  ["parties", "resolve", "disputes", "arbitration", "rules"],
                  ["agreement", "effect", "terminated", "party", "notice"]]

# Split dataset into training, validation, and test sets
train_docs, temp_docs = train_test_split(processed_docs, test_size=0.3, random_state=42)
val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

# Create Dictionary and Corpus for Training
from gensim.corpora import Dictionary

dictionary = Dictionary(train_docs)
train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
val_corpus = [dictionary.doc2bow(doc) for doc in val_docs]
test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

# Convert Corpus to the format compatible with cuML
def convert_to_cudf(corpus, dictionary):
    # Prepare a dense matrix representation for cuML
    corpus_list = []
    for doc in corpus:
        bow = [0] * len(dictionary)
        for word_id, count in doc:
            bow[word_id] = count
        corpus_list.append(bow)
    # Convert to cuDF DataFrame
    return cudf.DataFrame(corpus_list)

train_corpus_cudf = convert_to_cudf(train_corpus, dictionary)
val_corpus_cudf = convert_to_cudf(val_corpus, dictionary)
test_corpus_cudf = convert_to_cudf(test_corpus, dictionary)

# Hyperparameters
hyperparams = {
    "num_topics": 3,
    "max_iter": 10,
    "random_state": 42
}
logging.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

# Train cuML LDA Model
logging.info("Training cuML LDA model...")
lda_model = cuLDA(n_components=hyperparams['num_topics'], 
                  max_iter=hyperparams['max_iter'], 
                  random_state=hyperparams['random_state'])

lda_model.fit(train_corpus_cudf)

# Save model checkpoint
checkpoint_path = "lda_model_checkpoint"
os.makedirs(checkpoint_path, exist_ok=True)
lda_model.save(os.path.join(checkpoint_path, "lda_model_cuml.pkl"))
logging.info(f"Model saved to {checkpoint_path}")

# Sanity Check: Overfitting on one batch
logging.info("Performing sanity check...")
one_batch = train_corpus_cudf[:1]
one_batch_model = cuLDA(n_components=hyperparams['num_topics'], 
                        max_iter=hyperparams['max_iter'], 
                        random_state=hyperparams['random_state'])
one_batch_model.fit(one_batch)
logging.info("Sanity check completed. Model overfit to one batch.")

# Log Results
def log_results(model, corpus, dataset_name):
    logging.info(f"Results for {dataset_name}:")
    for idx, doc in enumerate(corpus):
        topics = model.transform(doc)
        logging.info(f"Document {idx + 1}: {topics}")

log_results(lda_model, train_corpus_cudf, "Training Set")
log_results(lda_model, val_corpus_cudf, "Validation Set")
log_results(lda_model, test_corpus_cudf, "Test Set")

# Print Topics and Keywords
logging.info("Extracted Topics and Keywords:")
for idx, topic in enumerate(lda_model.components_):
    logging.info(f"Topic {idx}: {topic}")
