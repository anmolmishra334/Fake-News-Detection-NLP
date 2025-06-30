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

## Fake News Detection Project (Basic Plan)
# Step 1: Technical Documentation Creation (SRS)

Understand the problem statement thoroughly.

Write down requirements needed to be addressed.

Mention goals of the project. Highlight PC requirements.

Create a detailed Project Scope.

# Step 2: Data Collection

Gather all needed and best datasets. Highlight no. of columns and rows of datasets.

Find CSV files released for research.

Mention source from where it is obtained.

# Step 3: Data Cleaning and Pre-processing

Clean the CSV files, remove any fields with null values.

Add features for each row, remove old features.

Check and remove any missing or duplicate data.

Make features consistent in one fixed format.

Use a Python script for data automation to process input and output files.

Create a pipeline of scripts for automation.

Don't use hardcoded file names for training and test datasets.

# Step 4: ML Model Creation and Training

Create a Python script. Import all required libraries (mentioned already in the YouTube video) and include .pdf file exported as archive.

Train data and training data separation.

ML model creation and testing on test workers within Jupyter/Colab.

If the model is successfully giving correct predictions, make changes in CSV files and re-separate two CSV files. Save in a folder.

Matplotlib and Seaborn utilization for creating diagrams/charts showing:

Count of fake and real news.

Number of fake news increasing/decreasing with time.

Differences caused by news reproduction.

Any other plot/chart creation as needed.

# Step 5: Web-based Application

Simple web-app connected in the background with ML model.

Takes input as new CSV file.

Test incoming CSV file.

Test incoming CSV file has same no. of columns as in the training set. Data cleaning script runs on CSV file.

ML model makes predictions, highlights results, and creates two CSV files and one .ipynb file representing all bar/graph charts.

# Step 6: MLOps Implementation and Automation

MLOps pipeline creation with GitHub Actions:

Auto pre-processing, testing, scanning ML model.

Training, model packaging, and scanning.


Deploying on repo and web app, working on server, ready to use.

# Step 7 (Optional): Power BI Dashboard

Creation to showcase different analytics.

Further Enhancement (Optional):

Every new CSV file to check fake news is pushed back in GitHub. Pipeline triggers, re-trains model on new data, makes ML model give better predictions.
