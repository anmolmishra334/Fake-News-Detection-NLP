import pickle
from flask import Flask, render_template, request

# Load ML model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorization.pkl', 'rb') as vec_file:
    vectorization = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        if 'submit_text' in request.form:
            news_text = request.form.get('news_text')
            if news_text.strip():
                prediction = predict_news(news_text)
                result = f"Prediction: {prediction}"
            else:
                result = "Please enter some news text."

        elif 'submit_file' in request.form:
            uploaded_file = request.files.get('news_file')
            if uploaded_file and uploaded_file.filename.endswith('.txt'):
                file_content = uploaded_file.read().decode('utf-8')
                prediction = predict_news(file_content)
                result = f"Prediction (from file): {prediction}"
            else:
                result = "Please upload a valid .txt file."

    return render_template('index.html', result=result)

def predict_news(text):
    vectorized_text = vectorization.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return "FAKE" if prediction == 1 else "REAL"

if __name__ == '__main__':
    app.run(debug=True)
