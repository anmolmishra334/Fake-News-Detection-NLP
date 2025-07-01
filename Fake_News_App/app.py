import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

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


@app.route('/generate_notebook/<filename>')
def generate_graphs_notebook(filename):
    # Load the test CSV file
    test_file_path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(test_file_path)

    # Add word and character counts if not present
    if 'word_count' not in df.columns:
        df['word_count'] = df['Complete News'].apply(lambda x: len(str(x).split()))
    if 'character_count' not in df.columns:
        df['character_count'] = df['Complete News'].apply(lambda x: len(str(x)))

    # Create notebook
    notebook_path = os.path.join(UPLOAD_FOLDER, 'graph_details.ipynb')
    nb = new_notebook()
    nb.cells.append(new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom wordcloud import WordCloud"))
    
    # Fake vs. Real News Count
    nb.cells.append(new_code_cell("""
plt.figure(figsize=(6,4))
sns.countplot(x='Fake News(Yes/No)', data=df, palette='Set1')
plt.title("Fake vs Real News Count")
plt.xlabel("Fake News (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()
"""))

    # Word Count Distribution
    nb.cells.append(new_code_cell("""
plt.figure(figsize=(8,5))
sns.histplot(df['word_count'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of News Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()
"""))

    # Character Count Distribution
    nb.cells.append(new_code_cell("""
plt.figure(figsize=(8,5))
sns.histplot(df['character_count'], bins=50, kde=True, color='orange')
plt.title("Distribution of Character Counts")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.show()
"""))

    # Word Cloud
    nb.cells.append(new_code_cell("""
text_combined = ' '.join(df['Complete News'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of News Content")
plt.show()
"""))

    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    # Execute the notebook
    execute_notebook(notebook_path)

    return send_file(notebook_path, as_attachment=True)


def execute_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})

    with open(notebook_path, "w", encoding="utf-8") as nb_file:
        nbformat.write(notebook, nb_file)


def predict_news(text):
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
    df = pd.read_csv(file_path)

    if 'Complete News' not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'Complete News' column.")

    df['Model Predicted Fake News'] = df['Complete News'].apply(lambda x: 1 if predict_news(x) == "FAKE" else 0)

    output_file = f"processed_{os.path.basename(file_path)}"
    output_path = os.path.join(UPLOAD_FOLDER, output_file)
    df.to_csv(output_path, index=False)
    return output_file


if __name__ == '__main__':
    app.run(debug=True)
