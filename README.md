# Fake-News-Detection-NLP
Classify news articles as genuine or fake using Natural Language Processing
# Fake News Detection
## Project Dependecies:
pandas
numpy
scikit-learn
tensorflow
keras
transformers
nltk
matplotlib
joblib

## Overview
Fake news has become a significant issue in the digital age. This project aims to classify news articles as genuine or fake using Natural Language Processing (NLP) techniques. The system is designed to help identify misinformation and ensure the credibility of news sources.

## Features
- Preprocessing of text data, including cleaning and tokenization.
- Feature extraction using techniques like TF-IDF.
- Model training using Logistic Regression, LSTM, and BERT.
- Evaluation of model performance using accuracy, precision, recall, and F1-score.

## Tech Stack
- **Python**: Core programming language.
- **Libraries**: 
  - TF-IDF: `scikit-learn`
  - LSTM: `tensorflow`, `keras`
  - BERT: `transformers`
- **Tools**: Jupyter Notebook for data exploration and visualization.

## Dataset
We used publicly available datasets for fake news detection:
- [Fake News Dataset from Kaggle](https://www.kaggle.com/c/fake-news/data)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

Download the datasets and place them in the `data/` directory.

## Installation
To set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fake-News-Detection.git
   cd Fake-News-Detection

## Folder Structure
Fake-News-Detection/
│
├── data/                  # For datasets
│   ├── train.csv
│   └── test.csv
│
├── notebooks/             # Jupyter Notebooks
│   └── EDA.ipynb
│
├── src/                   # Python scripts for models and preprocessing
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/                # Saved models
│   └── fake_news_model.pkl
│
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── .gitignore             # Ignored files
