import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import os

# Load the dataset
df = pd.read_csv('dataset/fake_news_final.csv')
x = df['Complete News']
y = df['Fake News(Yes/No)']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Text preprocessing and vectorization
vectorization = TfidfVectorizer()
x_train_vectorized = vectorization.fit_transform(x_train)
x_test_vectorized = vectorization.transform(x_test)

# Tokenization and padding for LSTM
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_padded = pad_sequences(x_train_seq, maxlen=100, padding='post')
x_test_padded = pad_sequences(x_test_seq, maxlen=100, padding='post')

# Filepath for the saved ensemble model
ensemble_model_file = 'dataset/new_model_folder/model.pkl'

# Check if an existing ensemble model is available
if os.path.exists(ensemble_model_file):
    with open(ensemble_model_file, 'rb') as file:
        saved_data = pickle.load(file)
        print("Loaded pre-trained ensemble model.")
        vectorization = saved_data['vectorizer']
        tokenizer = saved_data['tokenizer']
else:
    print("No pre-trained ensemble model found. Training a new ensemble model.")

# Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(x_train_vectorized, y_train)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train_vectorized, y_train)

# LSTM model
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(x_train_padded, y_train, epochs=5, batch_size=64, verbose=1)

# Predictions for the Ensemble
log_reg_preds = log_reg.predict_proba(x_test_vectorized)[:, 1]
dt_preds = dt_model.predict_proba(x_test_vectorized)[:, 1]
lstm_preds = lstm_model.predict(x_test_padded).flatten()

# Ensemble: Soft Voting
import numpy as np
ensemble_preds = (log_reg_preds + dt_preds + lstm_preds) / 3
ensemble_final_preds = (ensemble_preds >= 0.5).astype(int)

# Classification Reports
print("Logistic Regression Report:")
print(classification_report(y_test, (log_reg_preds >= 0.5).astype(int)))

print("Decision Tree Report:")
print(classification_report(y_test, (dt_preds >= 0.5).astype(int)))

print("LSTM Report:")
print(classification_report(y_test, (lstm_preds >= 0.5).astype(int)))

print("Ensemble Model Report:")
print(classification_report(y_test, ensemble_final_preds))

# Save only the Ensemble Model
new_model_folder = 'dataset/new_model_folder'
os.makedirs(new_model_folder, exist_ok=True)
'''
with open(ensemble_model_file, 'wb') as file:
    pickle.dump({'vectorizer': vectorization, 
                 'tokenizer': tokenizer, 
                 'ensemble_weights': [log_reg_preds, dt_preds, lstm_preds]}, file)
    print(f"Ensemble model saved to {ensemble_model_file}.")
'''
# Save trained models along with vectorizer and tokenizer
with open(ensemble_model_file, 'wb') as file:
    pickle.dump({
        'log_reg': log_reg,
        'dt_model': dt_model,
        'lstm_model': lstm_model,
        'vectorizer': vectorization,
        'tokenizer': tokenizer
    }, file)
print("Trained models and preprocessing tools saved to model.pkl.")
