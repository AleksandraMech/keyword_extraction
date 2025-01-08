import re
import os
import nltk
import json
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample Legal Documents (replace with your dataset)
legal_docs = [
    "The contract shall be governed by the laws of the State of California.",
    "Parties agree to resolve disputes through arbitration in accordance with applicable rules.",
    "This agreement shall remain in effect until terminated by either party with proper notice.",
    "The law provides specific guidelines for intellectual property disputes.",
    "Privacy laws govern the collection and use of personal data."
]

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
    return tokens

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in legal_docs]

# Split dataset into training, validation, and test sets
train_docs, temp_docs = train_test_split(processed_docs, test_size=0.3, random_state=42)
val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

# Create Dictionary and Corpus for Training
dictionary = Dictionary(train_docs)
train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
val_corpus = [dictionary.doc2bow(doc) for doc in val_docs]
test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

# Hyperparameters
hyperparams = {
    "num_topics": 3,
    "passes": 10,
    "random_state": 42
}
logging.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

# Train LDA Model
logging.info("Training LDA model...")
lda_model = LdaModel(corpus=train_corpus, id2word=dictionary, 
                     num_topics=hyperparams['num_topics'], 
                     random_state=hyperparams['random_state'], 
                     passes=hyperparams['passes'])

# Save model checkpoint
checkpoint_path = "lda_model_checkpoint"
os.makedirs(checkpoint_path, exist_ok=True)
lda_model.save(os.path.join(checkpoint_path, "lda_model.gensim"))
logging.info(f"Model saved to {checkpoint_path}")

# Sanity Check: Overfitting on one batch
logging.info("Performing sanity check...")
one_batch = train_corpus[:1]
one_batch_model = LdaModel(corpus=one_batch, id2word=dictionary, 
                          num_topics=hyperparams['num_topics'], 
                          random_state=hyperparams['random_state'], 
                          passes=hyperparams['passes'])
logging.info("Sanity check completed. Model overfit to one batch.")

# Log Results
def log_results(model, corpus, dataset_name):
    logging.info(f"Results for {dataset_name}:")
    for idx, doc in enumerate(corpus):
        topics = model.get_document_topics(doc)
        logging.info(f"Document {idx + 1}: {topics}")

log_results(lda_model, train_corpus, "Training Set")
log_results(lda_model, val_corpus, "Validation Set")
log_results(lda_model, test_corpus, "Test Set")

# Print Topics and Keywords
logging.info("Extracted Topics and Keywords:")
for idx, topic in lda_model.print_topics(-1):
    logging.info(f"Topic {idx}: {topic}")

# Extract Keywords for Each Document
def extract_keywords(doc, model, dictionary):
    bow = dictionary.doc2bow(doc)
    topics = model.get_document_topics(bow)
    topic_keywords = []
    for topic_id, _ in topics:
        keywords = model.show_topic(topic_id, topn=5)
        topic_keywords.extend([word for word, _ in keywords])
    return set(topic_keywords)

# Example of Keyword Extraction
logging.info("Keywords per Document:")
for i, doc in enumerate(processed_docs):
    keywords = extract_keywords(doc, lda_model, dictionary)
    logging.info(f"Document {i + 1}: {keywords}")
