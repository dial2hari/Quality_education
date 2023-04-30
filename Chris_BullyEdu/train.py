import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
%matplotlib inline
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
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle
nltk.download('stopwords')

data = pd.read_csv('cyberbullying_tweets.csv')

data = data.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})

data["sentiment_encoded"] = data['sentiment'].replace({"religion": 1, "age": 2, "ethnicity": 3, "gender": 4, "other_cyberbullying": 5,"not_cyberbullying": 6})

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

# def preprocess(text):
#     text = strip_emoji(text)
#     text = decontract(text)
#     text = strip_all_entities(text)
#     text = clean_hashtags(text)
#     text = filter_chars(text)
#     text = remove_mult_spaces(text)
#     text = stemmer(text)
#     text = lemmatize(text)
#     return text

data['cleaned_text'] = data['text'].apply(strip_emoji)
data['cleaned_text'] = data['text'].apply(decontract)
data['cleaned_text'] = data['text'].apply(strip_all_entities)
data['cleaned_text'] = data['text'].apply(clean_hashtags)
data['cleaned_text'] = data['text'].apply(filter_chars)
data['cleaned_text'] = data['text'].apply(remove_mult_spaces)
data['cleaned_text'] = data['text'].apply(stemmer)
data['cleaned_text'] = data['text'].apply(lemmatize)

data.drop_duplicates("cleaned_text", inplace=True)

data['tweet_list'] = data['cleaned_text'].apply(word_tokenize)
data.head()

text_len = []
for text in data.tweet_list:
    tweet_len = len(text)
    text_len.append(tweet_len)
data['text_len'] = text_len

data = data[data['text_len']!=0]


sentiments = ["religion", "age", "ethnicity", "gender", "other_cyberbullying","not_cyberbullying"]

X,Y = data['cleaned_text'],data['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify =Y, random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn2pmml.feature_extraction.text import Splitter

tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(X_train)
X_test_tf = tf_idf.transform(X_test)


from sklearn.svm import LinearSVC

lin_svc = LinearSVC(C=1, loss='hinge')

lin_svc.fit(X_train_tf,y_train)

y_pred = lin_svc.predict(X_test_tf)






# Dumping the models in the form of pickles (.pkl)
pickle.dump(lin_svc,open('model.pkl',"wb"))
pickle.dump(tf_idf, open('vectorizer.pkl',"wb"))

model = pickle.load(open('model.pkl',"rb"))
vectorizer = pickle.load(open('vectorizer.pkl',"rb"))





