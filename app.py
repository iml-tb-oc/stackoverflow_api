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
vectorizer = joblib.load(model_path + "tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer.pkl", 'r')
model = joblib.load(model_path + "logit_nlp_model.pkl", 'r')
    
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
    to_predict_list = request.form.to_dict()
    review_text = pre_processing(to_predict_list['review_text'])
    
    pred = clf.predict(count_vect.transform([review_text]))
    prob = clf.predict_proba(count_vect.transform([review_text]))
    #pr =  1
    if prob[0][0]>=0.5:
        prediction = "Positive"
        #pr = prob[0][0]
    else:
        prediction = "Negative"
        #pr = prob[0][0]

    # sanity check to filter out non questions. 
    if not re.search("(?i)(what|which|who|where|why|when|how|whose|\?)",to_predict_list['review_text']):
        prediction = "Negative"
        #prob = prob*0
        
   
    
    return flask.render_template('predict.html', prediction = prediction, prob =np.round(prob[0][0],3)*100)


if __name__ == '__main__':
    app.run(debug=True)
    
