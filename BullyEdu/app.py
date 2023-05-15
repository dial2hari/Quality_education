import pickle
import numpy as np
import re
import multiprocessing as mp
from multiprocessing import Process
from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import emoji
import string
import nltk
from PIL import Image
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
nltk.download('stopwords')
from pypmml import Model
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn2pmml.feature_extraction.text import Splitter
from nyoka import skl_to_pmml


#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained models. (Pickle files)
regressor_agg = pickle.load(open('models/regressor_agg.pkl', 'rb'))
regressor_att = pickle.load(open('models/regressor_att.pkl', 'rb'))
regressor_tox = pickle.load(open('models/regressor_tox.pkl', 'rb'))

vectorizer_agg = pickle.load(open('models/vectorizer_agg.pkl',"rb"))
vectorizer_att = pickle.load(open('models/vectorizer_att.pkl',"rb"))
vectorizer_tox = pickle.load(open('models/vectorizer_tox.pkl',"rb"))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.

@app.route('/')
def home():
    return render_template('emotional.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output

@app.route('/search',methods=['POST'])
def search():
    str_features=[]
    
    #Convert inputs to string
    str_features = [str(request.form.values())]
    data1 = pd.DataFrame(str_features,columns=['text'])
    
    
    # Pre-processing Steps
    # Function to remove Emojis (if any) and replacing them with blank
    def rem_emoji(text):
        return emoji.replace_emoji(text,replace="")

    # # Fucntion to convert text to lowercase, remove (/r, /n  characters), URLs, non-utf characters, numbers, punctuations and/or stopword
    def rem_all_entities(text): 
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

    # # Function to remove contractions
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

    # # Function to Remove Hashtags
    def rem_hashtags(text):
        new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
        new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text))
        return new_text2

    # # Function to remove special characters like $, &
    def rem_chars(text):
        sent = []
        for word in text.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)

    # # Function to remove mutiple sequence spaces
    def rem_multi_spaces(text):
        return re.sub("\s\s+" , " ", text)

    # # Function to apply Porter stemming to words
    def stemmer(text):
        tokenized = nltk.word_tokenize(text)
        ps = PorterStemmer()
        return ' '.join([ps.stem(words) for words in tokenized]) # type: ignore

    # # Function to apply WordNet lemmatization to words
    def lemmatize(text):
        tokenized = nltk.word_tokenize(text)
        lm = WordNetLemmatizer()
        return ' '.join([lm.lemmatize(words) for words in tokenized]) # type: ignore

    # # Defining the stopwords
    stop_words = set(stopwords.words('english')) # type: ignore

    # Combining all the pre-processing functionxss to a single function
    def preproces(text):
        text = rem_emoji(text)
        text = decontract(text)
        text = rem_all_entities(text)
        text = rem_hashtags(text)
        text = rem_chars(text)
        text = rem_multi_spaces(text)
        text = stemmer(text)
        text = lemmatize(text)
        return text


    # Applying the Pre-processing function
    features = data1['text'].apply(preproces)
    # features = list(features)
    
    
    # Applying the TfidfVectorizer  
    features_tf_agg = vectorizer_agg.transform(features)
    features_tf_att = vectorizer_att.transform(features)
    features_tf_tox = vectorizer_tox.transform(features)
    
    
    # Prediction of regressor_agg
    prediction_agg = regressor_agg.predict(features_tf_agg)
    
    # Prediction of regressor_att
    prediction_att = regressor_att.predict(features_tf_att)
    
    # Prediction of regressor_tox
    prediction_tox = regressor_tox.predict(features_tf_tox)
    
    # print(float(np.asarray(prediction_tox)))
    
    prediction_agg = float(np.asarray(prediction_agg))
    prediction_att = float(np.asarray(prediction_att))
    prediction_tox = float(np.asarray(prediction_tox))
    
    # Converting to percent and rounding off with no decimal points
    output_agg = round(100*abs(prediction_agg), 1)
    output_att = round(100*abs(prediction_att), 1)
    output_tox = round(100*abs(prediction_tox), 1)
    
    # print(output_agg)
    # print(output_att)
    # print(output_tox)
    # print(prediction_agg)
    # print(prediction_att)
    # print(prediction_tox)
    # print(features_tf_agg )
    # print(features_tf_att )
    # print(features_tf_tox )
    # print(features)
    # print(data1)
    # print(str_features)
    
    
    

    return render_template('emotional.html', prediction_text=f'Aggressive stance % is {format(output_agg)}%, Attacking Stance % is {format(output_att)}% and {format(output_tox)}%.')


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()