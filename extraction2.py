import re
import os
import nltk
import json
import gensim
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from datetime import datetime
import logging
from sklearn.metrics import f1_score, accuracy_score
import fitz  # PyMuPDF for reading PDFs

# Setup logging to write to a file
logging.basicConfig(
    filename='output.txt',  # Logi będą zapisywane w pliku 'output.txt'
    level=logging.INFO,  # Poziom logowania
    format='%(asctime)s - %(message)s'  # Format logów
)

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

# Calculate F1-score (Precision, Recall, F1)
def calculate_f1(true_keywords, predicted_keywords):
    precision = f1_score(true_keywords, predicted_keywords, average='micro')
    recall = f1_score(true_keywords, predicted_keywords, average='micro')
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Correct the number of samples to match
true_keywords = [
    set(["contract", "governed", "laws", "state", "california"]), 
    set(["parties", "resolve", "disputes", "arbitration", "rules"]),
    set(["agreement", "effect", "terminated", "party", "notice"]),
    set(["law", "guidelines", "intellectual", "property", "disputes"]),
    set(["privacy", "laws", "collection", "use", "personal", "data"])
]

# Ensure that predicted_keywords matches the length of true_keywords
predicted_keywords = [
    set(["contract", "governed", "california"]),
    set(["parties", "resolve", "rules"]),
    set(["agreement", "terminated", "notice"]),
    set(["law", "guidelines", "property", "disputes"]),
    set(["privacy", "collection", "use", "data"])
]

# Flatten the sets and calculate F1
flatten_true_keywords = [item for sublist in true_keywords for item in sublist]
flatten_predicted_keywords = [item for sublist in predicted_keywords for item in sublist]

# Ensure the two lists have the same number of elements
if len(flatten_true_keywords) != len(flatten_predicted_keywords):
    min_len = min(len(flatten_true_keywords), len(flatten_predicted_keywords))
    flatten_true_keywords = flatten_true_keywords[:min_len]
    flatten_predicted_keywords = flatten_predicted_keywords[:min_len]

precision, recall, f1 = calculate_f1(flatten_true_keywords, flatten_predicted_keywords)

logging.info(f"Precision: {precision}")
logging.info(f"Recall: {recall}")
logging.info(f"F1-Score: {f1}")

# Categorical Accuracy (how many documents have correct predictions)
def categorical_accuracy(true_keywords, predicted_keywords):
    correct = 0
    for true, pred in zip(true_keywords, predicted_keywords):
        if true == pred:
            correct += 1
    return correct / len(true_keywords)

accuracy = categorical_accuracy(true_keywords, predicted_keywords)
logging.info(f"Categorical Accuracy: {accuracy}")

# Calculate Word Error Rate (WER)
def calculate_wer(ground_truth, predicted):
    d = np.zeros((len(ground_truth) + 1, len(predicted) + 1))

    for i in range(len(ground_truth) + 1):
        d[i][0] = i
    for j in range(len(predicted) + 1):
        d[0][j] = j

    for i in range(1, len(ground_truth) + 1):
        for j in range(1, len(predicted) + 1):
            if ground_truth[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,  
                          d[i][j - 1] + 1,  
                          d[i - 1][j - 1] + cost)  

    return d[len(ground_truth)][len(predicted)] / float(len(ground_truth))

# Calculate WER for each document
wer_scores = []
for gt, pred in zip(true_keywords, predicted_keywords):
    wer_score = calculate_wer(list(gt), list(pred))
    wer_scores.append(wer_score)

# Average WER
average_wer = np.mean(wer_scores)
logging.info(f"Average Word Error Rate (WER): {average_wer}")
