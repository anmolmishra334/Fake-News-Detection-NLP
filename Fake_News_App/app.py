import pickle
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load ML models and vectorizer
# Load saved model data including vectorizer, tokenizer, and individual models
with open('model.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    log_reg = saved_data['log_reg']  # Logistic Regression model
    dt_model = saved_data['dt_model']  # Decision Tree model
    lstm_model = saved_data['lstm_model']  # LSTM model
    vectorization = saved_data['vectorizer']  # TfidfVectorizer for text preprocessing
    tokenizer = saved_data['tokenizer']  # Tokenizer for LSTM preprocessing

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the main page and handles prediction requests.
    Supports predictions for both text input and file uploads.
    """
    result = ''
    if request.method == 'POST':
        # Check if text is submitted
        if 'submit_text' in request.form:
            news_text = request.form.get('news_text')  # Get the input text
            if news_text.strip():  # Ensure input is not empty
                prediction = predict_news(news_text)  # Predict using the ensemble
                result = f"Prediction: {prediction}"
            else:
                result = "Please enter some news text."

        # Check if a file is uploaded
        elif 'submit_file' in request.form:
            uploaded_file = request.files.get('news_file')  # Get uploaded file
            if uploaded_file and uploaded_file.filename.endswith('.txt'):
                file_content = uploaded_file.read().decode('utf-8')  # Read file content
                prediction = predict_news(file_content)  # Predict using the ensemble
                result = f"Prediction (from file): {prediction}"
            else:
                result = "Please upload a valid .txt file."

    return render_template('index.html', result=result)  # Render the page with result

def predict_news(text):
    """
    Preprocesses the input text and predicts whether the news is fake or real.
    Uses an ensemble of Logistic Regression, Decision Tree, and LSTM models.
    """
    # Preprocess text for Tfidf-based models
    vectorized_text = vectorization.transform([text])
    
    # Preprocess text for LSTM
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=100, padding='post')

    # Get predictions from individual models
    log_reg_pred = log_reg.predict_proba(vectorized_text)[:, 1][0]
    dt_pred = dt_model.predict_proba(vectorized_text)[:, 1][0]
    lstm_pred = lstm_model.predict(text_padded).flatten()[0]

    # Combine predictions using ensemble logic (soft voting)
    ensemble_pred = (log_reg_pred + dt_pred + lstm_pred) / 3

    # Determine final prediction
    return "FAKE" if ensemble_pred >= 0.5 else "REAL"

if __name__ == '__main__':
    # Start the Flask application in debug mode
    app.run(debug=True)
