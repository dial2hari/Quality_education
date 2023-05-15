!pip install "ray[tune]"

# Importing Libraries
import multiprocessing as mp
from multiprocessing import Process
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import emoji
import string
import nltk
from PIL import Image
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import FunctionTransformer
import sys

# Reading the data and storing it in the dataframes
agg_df = pd.read_csv("data/aggression_parsed_dataset.csv")
att_df = pd.read_csv("data/attack_parsed_dataset.csv")
tox_df = pd.read_csv("data/toxicity_parsed_dataset.csv")"

# Converting oh_label from int to str
agg_df['oh_label'] = agg_df['oh_label'].astype(str)
att_df['oh_label'] = att_df['oh_label'].astype(str)
tox_df['oh_label'] = tox_df['oh_label'].astype(str)

# Renaming the columns
agg_df = agg_df.rename({'ed_label_0':'Prob_not_aggression','ed_label_1':'Prob_aggression','oh_label':'aggression'},axis='columns')
att_df = att_df.rename({'ed_label_0':'Prob_not_attack','ed_label_1':'Prob_attack','oh_label':'attack'},axis='columns')
tox_df = tox_df.rename({'ed_label_0':'Prob_not_toxicity','ed_label_1':'Prob_toxicity','oh_label':'toxicity'},axis='columns')

# Dropping columns
agg_df.drop(['index','Prob_not_aggression','aggression'],axis='columns',inplace=True)
att_df.drop(['index','Prob_not_attack','attack'],axis='columns',inplace=True)
tox_df.drop(['index','Prob_not_toxicity','toxicity'],axis='columns',inplace=True)



print(tox_df['Prob_toxicity'].max())



### PreProcessing

# Function to remove Emojis (if any) and replacing them with blank
def rem_emoji(text):
    return emoji.replace_emoji(text,replace="")

# Fucntion to convert text to lowercase, remove (/r, /n  characters), URLs, non-utf characters, numbers, punctuations and/or stopword
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

# Function to remove contractions
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

# Function to Remove Hashtags
def rem_hashtags(text):
    new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
    new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text))
    return new_text2

# Function to remove special characters like $, &
def rem_chars(text):
    sent = []
    for word in text.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

# Function to remove mutiple sequence spaces
def rem_multi_spaces(text):
    return re.sub("\s\s+" , " ", text)

# Function to apply Porter stemming to words
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized]) # type: ignore

# Function to apply WordNet lemmatization to words
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized]) # type: ignore

# Defining the stopwords
stop_words = set(stopwords.words('english')) # type: ignore

# Combining all the pre-processing functions to a single function
def preproces(text):
#     text = rem_emoji(text)
    text = decontract(text)
    text = rem_all_entities(text)
    text = rem_hashtags(text)
    text = rem_chars(text)
    text = rem_multi_spaces(text)
#    text = stemmer(text)
#    text = lemmatize(text)
    return text

import sys

# Defining the stack overflow for recursive functions
sys.setrecursionlimit(5000)


# Applying Pre-processing in the text of the dataframes
agg_df['tokenized_text'] = agg_df['Text'].apply(preproces)
att_df['tokenized_text'] = att_df['Text'].apply(preproces)
tox_df['tokenized_text'] = tox_df['Text'].apply(preproces)

# Dealing with Duplicates
print(agg_df["Text"].duplicated().sum())
print(att_df["Text"].duplicated().sum())
print(tox_df["Text"].duplicated().sum())

# Removing the duplicates
agg_df.drop_duplicates("Text", inplace=True)
att_df.drop_duplicates("Text", inplace=True)
tox_df.drop_duplicates("Text", inplace=True)

# Assigning data types for numbers for all the datasets
agg_df['Prob_aggression'] = agg_df['Prob_aggression'].astype(float)
att_df['Prob_attack'] = att_df['Prob_attack'].astype(float)
tox_df['Prob_toxicity'] = tox_df['Prob_toxicity'].astype(float)

# Dropping the text column from the datasets
agg_df.drop('Text',axis='columns',inplace=True)
att_df.drop('Text',axis='columns',inplace=True)
tox_df.drop('Text',axis='columns',inplace=True)

# Rearranging the columns of the datasets
# agg_df = agg_df['tokenized_text','Prob_aggression']
# att_df = att_df['tokenized_text','Prob_attack']
# tox_df = tox_df['tokenized_text','Prob_toxicity']

print(tox_df.head())

# Writing the dataframe to csv files
agg_df.to_csv("data/agg_dataset.csv", header=True, index=True,mode='w')
att_df.to_csv("data/att_dataset.csv", header=True, index=True,mode='w')
tox_df.to_csv("data/tox_dataset.csv", header=True, index=True,mode='w')

# Splitting the datasets into predictor and target datasets for all dataframes
X_agg,Y_agg = agg_df['tokenized_text'],agg_df['Prob_aggression']
X_att,Y_att = att_df['tokenized_text'],att_df['Prob_attack']
X_tox,Y_tox = tox_df['tokenized_text'],tox_df['Prob_toxicity']

# Splitting the datasets into training and testing datasets for all dataframes
X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(X_agg, Y_agg, test_size = 0.2, random_state = 1234)
X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_att, Y_att, test_size = 0.2, random_state = 1234)
X_train_tox, X_test_tox, y_train_tox, y_test_tox = train_test_split(X_tox, Y_tox, test_size = 0.2, random_state = 1234)

# Function for TF-IDF Vectoraization
def vectorization(X_train,X_test):
    
    # Instantiating the vectorizer
    vectorizer = TfidfVectorizer()
    
    # Applying tf-idf vecotization to the training and testing datasets
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)
    
    return X_train_tf, X_test_tf, vectorizer

# Applying TF-IDF Vectoraization to the training and testing datasets
X_train_agg_tf, X_test_agg_tf, vectorizer_agg = vectorization(X_train_agg,X_test_agg)
X_train_att_tf, X_test_att_tf, vectorizer_att = vectorization(X_train_att,X_test_att)
X_train_tox_tf, X_test_tox_tf, vectorizer_tox = vectorization(X_train_tox,X_test_tox)

# Grid Search CV to arrive at the best parameters for training the SVR model
# def hypertuning(x,y):
    
#     # Defining the parameters
#     parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[0.1, 1, 10, 100, 1000],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
    
#     # Instantiating the SVR
#     svm = RandomForestRegressor()
    
#     # Instantiating the Grid Search CV
#     clf = GridSearchCV(svm, parameters,n_jobs=-1)
    
#     # Fitting the grid search with predictors and target variables of the training dataset
#     clf.fit(x,y)
    
#     return clf.best_params_

# best_param = hypertuning(X_train_agg_tf,y_train_agg)
# print(best_param)


def train_model(predictor,target):
    
    # Instantiating the Support Vector Regressor
    regressor = RandomForestRegressor()
    
    # Fitting the model with the train and test data
    regressor.fit(predictor, target)
    return regressor

# Training the model
regressor_agg = train_model(X_train_agg_tf,y_train_agg)
regressor_att = train_model(X_train_att_tf,y_train_att)
regressor_tox = train_model(X_train_tox_tf,y_train_tox)


print(type(regressor_agg))

# Defining a function to predict
def predictor(model,test):
    y_pred = model.predict(test)
    return y_pred

# Predicting the Percentages
y_pred_agg = predictor(regressor_agg,X_test_agg_tf)
y_pred_att = predictor(regressor_att,X_test_att_tf)
y_pred_tox = predictor(regressor_tox,X_test_tox_tf)

# print(y_pred_agg)
# print(y_pred_att)
# print(y_pred_tox)

y_pred_agg_1 = pd.DataFrame(y_pred_agg)
y_pred_att_1 = pd.DataFrame(y_pred_att)
y_pred_tox_1 = pd.DataFrame(y_pred_tox)

score_agg = np.sqrt(mean_squared_error(y_test_agg, y_pred_agg_1))
print(score_agg)

score_att = np.sqrt(mean_squared_error(y_test_att, y_pred_att_1))
print(score_att)

score_tox = np.sqrt(mean_squared_error(y_test_tox, y_pred_tox_1))
print(score_tox)

# R2 Score of the model
print(f'The R2 Score of agg is: {r2_score(y_test_agg, y_pred_agg_1)}')
print(f'The R2 Score of att is: {r2_score(y_test_att, y_pred_att_1)}')
print(f'The R2 Score of tox is: {r2_score(y_test_tox, y_pred_tox_1)}')









import pickle

# Dumping the models in the form of pickles (.pkl)
pickle.dump(regressor_agg,open('regressor_agg.pkl',"wb"))
pickle.dump(regressor_att,open('regressor_att.pkl',"wb"))
pickle.dump(regressor_tox,open('regressor_tox.pkl',"wb"))

# Dumping the fitted TFIDFVectorizer
pickle.dump(vectorizer_agg,open('vectorizer_agg.pkl',"wb"))
pickle.dump(vectorizer_att,open('vectorizer_att.pkl',"wb"))
pickle.dump(vectorizer_tox,open('vectorizer_tox.pkl',"wb"))


regressor_agg = pickle.load(open('regressor_agg.pkl',"rb"))
regressor_att = pickle.load(open('regressor_att.pkl',"rb"))
regressor_tox = pickle.load(open('regressor_tox.pkl',"rb"))

vectorizer_agg = pickle.load(open('vectorizer_agg.pkl',"rb"))
vectorizer_att = pickle.load(open('vectorizer_att.pkl',"rb"))
vectorizer_tox = pickle.load(open('vectorizer_tox.pkl',"rb"))






# Dumping the Preprocess function in form of pickle (.pkl)
# pickle.dump('preprocess',open('preprocess.pkl',"wb"))
