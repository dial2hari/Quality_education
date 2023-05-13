from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

import pickle
import numpy as np
import re
from flask import Flask, request, render_template
import pandas as pd


import emoji
import string
import nltk
# from PIL import Image
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
nltk.download('stopwords')
#from pypmml import Model
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
#parser.add_argument('data')

#stop_words = set(stopwords.words('english'))

def strip_emoji(text):
    return emoji.replace_emoji(text,replace="")

def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub(r'(.)1+', r'1', text)
    text = re.sub('[0-9]+', '', text)
    stopchars= string.punctuation
    table = str.maketrans('', '', stopchars)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text

def decontract(text):
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text

def clean_hashtags(tweet):
        new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
        new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
        return new_tweet2

def filter_chars(a):
        sent = []
        for word in a.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)

def remove_mult_spaces(text):
        return re.sub("\s\s+" , " ", text)

def stemmer(text):
        tokenized = nltk.word_tokenize(text)
        ps = PorterStemmer()
        return ' '.join([ps.stem(words) for words in tokenized])

def lemmatize(text):
        tokenized = nltk.word_tokenize(text)
        lm = WordNetLemmatizer()
        return ' '.join([lm.lemmatize(words) for words in tokenized])



def preprocess(text): #(initial one)
        text = strip_emoji(text)
        text = decontract(text)
        text = strip_all_entities(text)
        text = clean_hashtags(text)
        text = filter_chars(text)
        text = remove_mult_spaces(text)
        text = stemmer(text)
        text = lemmatize(text)
        return text

# Define how the api will respond to the post requests
class Classifier(Resource):
    def post(self):
        args = parser.parse_args()
        X = preprocess(args)
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

api.add_resource(Classifier, '/emotionaL.html')

if __name__ == '__main__':
    # Load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl','rb') as v:
        vec = pickle.load(v)
    app.run(debug=True)