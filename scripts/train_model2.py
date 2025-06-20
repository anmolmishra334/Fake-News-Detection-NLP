import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle
import os

# Load the dataset
df = pd.read_csv('dataset/fake_news_final.csv')
x = df['Complete News']
y = df['Fake News(Yes/No)']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize the text data
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

# Filepath for the saved model in the cloned repository
repo_model_file = 'dataset/git_repo_FND/Fake_News_App/model.pkl'

# Load the existing model if available, otherwise initialize a new Logistic Regression model
if os.path.exists(repo_model_file):
    with open(repo_model_file, 'rb') as file:
        model = pickle.load(file)
    print(f"Loaded pre-trained model from {repo_model_file}.")
else:
    model = LogisticRegression()
    print("No pre-trained model found in the repository. Initializing a new model.")

# Train the model on the new data
model.fit(x_train, y_train)

# Create a new folder for the updated model
new_model_folder = 'dataset/new_model_folder'
os.makedirs(new_model_folder, exist_ok=True)

# Save the updated model with the fixed filename `model.pkl`
new_model_file = os.path.join(new_model_folder, 'model.pkl')
with open(new_model_file, 'wb') as file:
    pickle.dump(model, file)
    print(f"Updated model saved to {new_model_file}.")

# Predict and evaluate the Logistic Regression model
pred_lr = model.predict(x_test)
print("Logistic Regression Report:")
print(classification_report(y_test, pred_lr))

# Decision Tree Classifier for comparison (optional)
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
print("Decision Tree Classifier Report:")
print(classification_report(y_test, pred_dt))
