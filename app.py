import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Specify the encoding when reading the CSV file
datas = pd.read_csv("dataset.csv", encoding='latin1')

# Convert the 'COMMENT' column to string
comments = datas['COMMENT'].astype(str)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(comments)

# Train the DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, datas['CATEGORY'])

# Streamlit app
st.title("Sentiment Analysis")

user_input = st.text_input("Type your comment: ", "")

if user_input:
    user_input_vector = vectorizer.transform([user_input])
    response = clf.predict(user_input_vector)
    st.text(f"Category: {response[0]}")
