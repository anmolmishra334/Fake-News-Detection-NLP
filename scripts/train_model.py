import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('dataset/fake_news_final.csv')
x = df['Complete News']
y = df['Fake News(Yes/No)']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
model = LogisticRegression()
model.fit(x_train, y_train)
pred_lr = model.predict(x_test)
print(classification_report(y_test, pred_lr))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = model.predict(x_test)
print(classification_report(y_test, pred_dt))
