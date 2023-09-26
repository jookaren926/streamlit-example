#### Import packages

import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("customer_reviews.csv", index_col=0) ## Load Data, Create Dataset
df['pos_rev'] = np.where(df['rating']>=4,1,0)
X = df['reviewContent'] 
y = df['pos_rev']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=987)

#Vectorize using TfidVectorizer
tfidfvec = TfidfVectorizer(stop_words = 'english')
X_train_vectorized_tfidf = tfidfvec.fit_transform(X_train.values)
X_test_vectorized_tfidf = tfidfvec.transform(X_test.values)

#Fit and train a classification models
    # Logistic regression
lr_class_tfidf = LogisticRegression(max_iter=5000) #Increase max_iter to 5000
lr_class_tfidf.fit(X_train_vectorized_tfidf, y_train)
lr_predictions_tfidf = lr_class_tfidf.predict(X_test_vectorized_tfidf)

def textclass_pred(text):
        # Create a new DataFrmae with the provided text
    newdf = pd.DataFrame({'text':[text]})
    
    # Transform the input text using the same CountVectorizer instance
    newdf_tfidfvec = tfidfvec.transform(newdf['text'])
    
    # Create dataframe with text and predicted label
    prediction = lr_class_tfidf.predict(newdf_tfidfvec)
    
    # Define sentiment labels
    sentiment_labels = {0: 'Negative', 1: 'Positive'}

    # Return a statement indicating whether the text is positive or negative
    sentiment = sentiment_labels[prediction[0]]
    return sentiment



st.markdown("### Enter your review, we'll guess how you feel!")

user_input = st.text_input("Enter review")

# Add an "Enter" button
if st.button("Enter"):
    if user_input:
        st.markdown("You Feel...")
        st.write(textclass_pred(user_input),"!!!")
