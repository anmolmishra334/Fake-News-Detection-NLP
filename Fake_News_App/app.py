import os
import pickle
import pandas as pd
import numpy._core as np
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load ML models and necessary utilities
with open('model.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    log_reg = saved_data['log_reg']
    dt_model = saved_data['dt_model']
    lstm_model = saved_data['lstm_model']
    vectorization = saved_data['vectorizer']
    tokenizer = saved_data['tokenizer']

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    csv_file_path = None

    if request.method == 'POST':
        # Handle single text submission
        if 'submit_text' in request.form:
            news_text = request.form.get('news_text')
            if news_text.strip():
                prediction = predict_news(news_text)
                result = f"Prediction: {prediction}"
            else:
                result = "Please enter some news text."

        # Handle CSV upload and processing
        elif 'submit_csv' in request.form:
            uploaded_file = request.files.get('news_csv')
            if uploaded_file and uploaded_file.filename.endswith('.csv'):
                # Save the uploaded CSV
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(file_path)

                # Process the CSV and get the output file
                csv_file_path = process_csv(file_path)
            else:
                result = "Please upload a valid .csv file."

    return render_template('index.html', result=result, csv_file=csv_file_path)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return f"File not found: {file_path}", 404

def predict_news(text):
    # Vectorize and tokenize the text for predictions
    vectorized_text = vectorization.transform([text])
    lstm_seq = tokenizer.texts_to_sequences([text])
    lstm_padded = pad_sequences(lstm_seq, maxlen=100, padding='post')

    # Ensemble prediction
    log_reg_pred = log_reg.predict_proba(vectorized_text)[:, 1]
    dt_pred = dt_model.predict_proba(vectorized_text)[:, 1]
    lstm_pred = lstm_model.predict(lstm_padded).flatten()

    final_pred = (log_reg_pred + dt_pred + lstm_pred) / 3
    return "FAKE" if final_pred >= 0.5 else "REAL"

def process_csv(file_path):
    # Load the uploaded CSV
    df = pd.read_csv(file_path)

    # Check for required column
    if 'Complete News' not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'Complete News' column.")

    # Predict for each row in the 'Complete News' column
    df['Fake News'] = df['Complete News'].apply(lambda x: 1 if predict_news(x) == "FAKE" else 0)

    # Save the processed CSV
    output_file = f"processed_{os.path.basename(file_path)}"
    output_path = os.path.join(UPLOAD_FOLDER, output_file)
    df.to_csv(output_path, index=False)
    return output_file  # Return only the filename for download

if __name__ == '__main__':
    app.run(debug=True)
