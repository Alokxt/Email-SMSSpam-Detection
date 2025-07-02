
import nltk
import streamlit as st
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer
ps = PorterStemmer()

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))  # Cache for speed


def transform(text):
    text = text.lower()

    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|@\S+|[\w\.-]+@[\w\.-]+', '', text)

    # Remove digits and phone-like patterns
    text = re.sub(r'\b\d{10,}\b', '', text)  # phone numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    words = nltk.word_tokenize(text)

    # Remove stopwords, punctuation, and non-alphanumerics
    y = []
    for i in words:
        if i not in stopwords_set and i not in string.punctuation and i.isalnum():
            y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer4.pkl','rb'))
model = pickle.load(open('model4.pkl','rb'))

st.title("Email/sms Spam Classifier")
input_text = st.text_area("Enter Your Message")
if st.button("Predict"):

    transformed_text = transform(input_text)
    if transformed_text:

        vector_input = tfidf.transform([transformed_text]).toarray()

        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")




