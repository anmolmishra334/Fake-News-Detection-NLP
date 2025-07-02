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
from flask import Flask, request, render_template, redirect, url_for

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
        
        elif 'submit_csv' in request.form:
            uploaded_file = request.files.get('news_csv')
            if uploaded_file and uploaded_file.filename.endswith('.csv'):
                filename = uploaded_file.filename
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(file_path)

                # Process the CSV
                csv_file_path = process_csv(file_path)

                # Check if checkbox was selected
                if 'allow_training' in request.form:
                    branch_name = os.path.splitext(filename)[0]
                    processed_csv_path = os.path.join(UPLOAD_FOLDER, csv_file_path)
                    push_to_git_branch(processed_csv_path, branch_name)
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

    # Create notebook
    notebook_path = os.path.join(UPLOAD_FOLDER, 'graph_details.ipynb')
    nb = new_notebook()
    nb.cells.append(new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom wordcloud import WordCloud"))

    nb.cells.append(new_code_cell(f'''
df = pd.read_csv(r"{test_file_path}")
df
'''))

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
# News Articles per Year
    nb.cells.append(new_code_cell("""
if 'Publish Dates' in df.columns:
    df['Publish Dates'] = pd.to_datetime(df['Publish Dates'], errors='coerce')
    df['Year'] = df['Publish Dates'].dt.year
    plt.figure(figsize=(10,6))
    sns.countplot(x='Year', data=df, palette='coolwarm')
    plt.title("News Articles per Year")
    plt.xticks(rotation=45)
    plt.show()
    """))

   # Fake News Count by Month
    nb.cells.append(new_code_cell("""
df['Publish Dates'] = pd.to_datetime(df['Publish Dates'], errors='coerce')
# Extract month name
df['Month'] = df['Publish Dates'].dt.strftime('%B')  # Full month name
df['Month_Num'] = df['Publish Dates'].dt.month       # For sorting
# Filter only fake news
df_fake = df[df['Fake News(Yes/No)'] == 1]
# Group by month
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
fake_by_month = df_fake.groupby('Month').size().reindex(month_order).fillna(0)
# Plot
plt.figure(figsize=(12,6))
sns.barplot(x=fake_by_month.index, y=fake_by_month.values, palette='Reds_r')
plt.title("Fake News Count by Month")
plt.xlabel("Month")
plt.ylabel("Number of Fake News Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
    """))
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    allow_training = request.form.get("allow_training")  # Will be 'on' if checked

    if file:
        filename = file.filename
        # Save the file somewhere or process it
        print(f"Received file: {filename}")
        print(f"Allow training: {allow_training}")

        # Example response
        return f"File {filename} received. Allow training: {'Yes' if allow_training else 'No'}"

    return "No file uploaded", 400
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


def push_to_git_branch(file_path, branch_name):
    GITHUB_USERNAME = "anmolmishra334"
    GITHUB_PASSWORD = "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"
    REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PASSWORD}@github.com/{GITHUB_USERNAME}/your_repo_name.git"

    # Git setup
    repo_dir = os.getcwd()  # assuming your repo is the project root

    subprocess.run(['git', 'config', '--global', 'user.name', GITHUB_USERNAME], cwd=repo_dir)
    subprocess.run(['git', 'config', '--global', 'user.email', f'{GITHUB_USERNAME}@users.noreply.github.com'], cwd=repo_dir)

    subprocess.run(['git', 'checkout', '-b', branch_name], cwd=repo_dir)
    subprocess.run(['git', 'add', file_path], cwd=repo_dir)
    subprocess.run(['git', 'commit', '-m', f'Add training data: {os.path.basename(file_path)}'], cwd=repo_dir)
    subprocess.run(['git', 'push', '-u', REPO_URL, branch_name], cwd=repo_dir)


if __name__ == '__main__':
    app.run(debug=True)
