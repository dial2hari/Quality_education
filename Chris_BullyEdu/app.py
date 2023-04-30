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
#from sklearn2pmml.feature_extraction.text import Splitter
#from nyoka import skl_to_pmml


#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained models. (Pickle files)
model = pickle.load(open('models/model.pkl',"rb"))
vectorizer = pickle.load(open('models/vectorizer.pkl',"rb"))

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

@app.route('/predict',methods=['POST'])
def predict():
    str_features=[]
    print(request.form.values())
    
    #Convert inputs to string.
    str_features = [str(request.form.values())]
    data = pd.DataFrame(str_features,columns=['text'])
    
    
    # Pre-processing Steps
    stop_words = set(stopwords.words('english'))

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

    data['cleaned_text'] = data['text'].apply(preprocess)
    # Applying the TfidfVectorizer  
    X_test_tf = vectorizer.transform(data['cleaned_text'])
    
    from sklearn.svm import LinearSVC

    y_pred = model.predict(X_test_tf)
    
    
    print(int(np.asarray(y_pred)))
    output = int(np.asarray(y_pred))
    
    if output == 1:
        result = 'Religion bullying'
    elif output == 2:
        result = 'Ageism'
    elif output == 3:
        result = 'Racial Discrimination / Bullying'
    elif output == 4:
        result = 'Gender Bullying / Sexism'
    elif output == 5:
        result = 'Other Bully Type'
    elif output == 6:
        result = 'Not Bully'
    else:
        result = 'Processing'

    return render_template('emotional.html', prediction_text=f'{format(result)}')


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()