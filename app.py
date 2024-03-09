# Importing the neccessary libraries
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

# Code to load the vectorizer, tokenizer, Naive Bayes and LSTM models
load_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
load_nb_model = pickle.load(open('initial_nb_model.pkl', 'rb'))
load_improved_nb_model = pickle.load(open('improved_nb_model.pkl', 'rb'))
load_tokenizer=pickle.load(open('tokenizer.pkl', 'rb'))
load_lstm_model=pickle.load(open('trained_lstm_model.pkl', 'rb'))

# Function to remove punctuation etc
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

# Function to remove stopwords and make the words lowercase etc
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to clean the data overall
def clean_data(text):
    cleaned_text = text.lower()
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)
    return cleaned_text

# function to make predictions of the news using the input values
def output_prediction_label(model_name, n, confidence_score):
    # If a score of 0 is returned, it is Fake News
    if n == 0:
        st.warning(f"{model_name}: Fake News. \nModel Confidence Score: {confidence_score}%")
    # If a score of 1 is returned, it is a Real News (not a Fake News)
    if n == 1:
        st.success(f"{model_name}: Real News(Not a Fake News). \nModel Confidence Score: {confidence_score}%")

# function to calculate the confidence score of the model when making predictions of a particular news article
def calculate_confidence_value(value):
    # if the article has a score of 0-0.5, it is a Fake news. The score is mapped to a confidence score of 100%.
    if 0 <= value < 0.5:
        calculated_percentage = ((0.5 - value) / (0.5)) * 100
    # if the article has a score of 0.5-1, it is a Real news. The score is mapped to a confidence score of 100%.
    elif 0.5 <= value <= 1:
        calculated_percentage = ((value - 0.5)/ 0.5) * 100 
    # if no score is returned, no score is printed. (To handle exceptions)
    else:
        calculated_percentage = None
    return calculated_percentage

# function to make predictions of a news article using a news text, initialized vectorizer and tokenizer.
def news_test(news, vectorizer, tokenizer):
    # storing the news article in the test_news["text"] as an array
    test_news = {"text": [news]}
    # storing the test_news in the new_def_test inside a dataframe
    new_def_test = pd.DataFrame(test_news)
    # cleaning the new_def_test["text"] data
    new_def_test["text"] = new_def_test["text"].apply(clean_data)
    # storing the new_def_test["text"] data as a new variable
    new_x_test = new_def_test["text"]
    # using a vectorizer to transform the new_def_test["text"] data
    new_xv_test = vectorizer.transform(new_x_test)
    # making a prediction of the news article using the Naive Bayes model
    pred_NB = load_improved_nb_model.predict(new_xv_test)

    # converting the news into a string and using the tokenizer to convert the news text to a sequence and storing it
    lstm_new_xv_test = tokenizer.texts_to_sequences([str(news)])
    # padding the tokenized news sequences to be of max length 10786 and storing it
    lstm_new_xv_test = pad_sequences(lstm_new_xv_test, maxlen=10786)

    # making a prediction of the news article using the LSTM model
    pred_LSTM = load_lstm_model.predict(lstm_new_xv_test)
    # converting the predicted values to an integer(0 or 1), to be classified as Fake or Real News
    pred_transformed_LSTM = (pred_LSTM > 0.499).astype(int)
    
    # printing the predicted values of the Naive Bayes model and LSTM model
    print(pred_LSTM[0][0])
    print(pred_NB[0])

    # returning 4 items(2 predictions and 2 confidence scores of the 2 models)
    # (set 1) the predicted LSTM value(0 or 1) to classify as Fake/Real News and confidence score of the LSTM model
    # (set 2) the predicted Naive Bayes value(0 or 1) to classify as Fake/Real News and confidence score of the Naive Bayes model
    return output_prediction_label("LSTM model", pred_transformed_LSTM[0], calculate_confidence_value(pred_LSTM[0][0])), output_prediction_label("Naive Bayes model", pred_NB[0], calculate_confidence_value(pred_NB[0]))

# Code to display the contents of the Fake News Web Application
if __name__ == '__main__':
    st.title('Fake News Detection Web Application')
    st.subheader("Enter the news content below and the various fake news detection models will predict the reliability of the news")
    sentence = st.text_area("Enter a news string: ", "",height=200)

    # predict the authenticity of the news when the predict button is called.
    predict_btt = st.button("predict")
    if predict_btt:
        lstm_prediction, nb_prediction = news_test(sentence, load_vectorizer, load_tokenizer)
