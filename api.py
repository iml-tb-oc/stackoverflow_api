#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask_restful import Api, Resource
import pandas as pd
import joblib
import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


# In[2]:


app = Flask(__name__)
api = Api(app)


# In[3]:


def remove_pos(nlp, x, pos_list):
    # Test of language detection
    lang = detect(x)
    if(lang != "en"):
        # Deep translate if not in English
        x = GoogleTranslator(source='auto', target='en').translate(x)
    
    doc = nlp(x)
    list_text_row = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row


# In[4]:


def text_cleaner(x, nlp, pos_list, lang="english"):
    # Remove POS not in "NOUN", "PROPN"
    x = remove_pos(nlp, x, pos_list)
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
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x


# In[5]:


vectorizer = TfidfVectorizer(analyzer="word",
                             max_df=.6,
                             min_df=0.005,
                             tokenizer=None,
                             preprocessor=' '.join,
                             stop_words=None,
                             lowercase=False,
                            max_features=20000)


# In[6]:


multilabel_binarizer = MultiLabelBinarizer()


# In[7]:


class Autotag(Resource):
    def get(self, question):
        # Clean the question sent
        nlp = en_core_web_sm.load(exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
        #nlp = spacy.load('en_core_web_md', exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
        pos_list = ["NOUN","PROPN"]
        rawtext = question
        cleaned_question = text_cleaner(rawtext, nlp, pos_list, "english")
        
        # Apply saved trained TfidfVectorizer
        X_tfidf = vectorizer.transform([cleaned_question])
        model=LogisticRegression()
        # Perform prediction
        predict = model.predict(X_tfidf)
        predict_probas = model.predict_proba(X_tfidf)
        # Inverse multilabel binarizer
        tags_predict = multilabel_binarizer.inverse_transform(predict)
        
        # DataFrame of probas
        df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
        df_predict_probas['Tags'] = multilabel_binarizer.classes_
        df_predict_probas['Probas'] = predict_probas.reshape(-1)
        # Select probas > 33%
        df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33]            .sort_values('Probas', ascending=False)
            
        # Results
        results = {}
        results['Predicted_Tags'] = tags_predict
        results['Predicted_Tags_Probabilities'] = df_predict_probas            .set_index('Tags')['Probas'].to_dict()
        
        return results, 200


# In[8]:


api.add_resource(Autotag, '/autotag/<question>')


# In[9]:


if __name__ == "__main__":
    app.run()
