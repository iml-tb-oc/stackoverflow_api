from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn import linear_model
#from sklearn.externals import joblib
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
from nltk.corpus import stopwords
import spacy

import flask

app = Flask(__name__)

model_path = "models/"
vectorizer = joblib.load(model_path + "tfidf_vectorizer_2.pkl", 'r')
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer_saved.pkl", 'r')
model = joblib.load(model_path + "regression_model_saved.pkl", 'r')
    
###################################################

def garder_nom(x):
    text = []
    for token in x:
        if token.pos_ in ["NOUN","PROPN"]:
            text.append(token.text)
    text=" ".join(text)
    text = text.lower().replace("c #", "c#")
    return text


def text_cleaner(x):
    # Remove POS not in "NOUN", "PROPN"
    nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
    x=nlp(x)
    x=garder_nom(x)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words("english")
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x
###################
###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #on clean notre question et on la vectorize 
    question= request.form.ToString()()
    cleaned_question=text_cleaner(question)
    X_tfidf = vectorizer.transform([cleaned_question])
    #prediction
    predict = model.predict(X_tfidf)    
    predict_probas = model.predict_proba(X_tfidf)
    tags_predict = multilabel_binarizer.inverse_transform(predict)
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    df_predict_probas['Tags'] = multilabel_binarizer.classes_
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33]\
            .sort_values('Probas', ascending=False)
    #resultat à retourner 
    results = {}
    results['Predicted_Tags'] = tags_predict
    results['Predicted_Tags_Probabilities'] = df_predict_probas\
            .set_index('Tags')['Probas'].to_dict()
    return results 
        
   
    
    return flask.render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
    
