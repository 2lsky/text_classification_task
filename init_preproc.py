import pandas as pd
import os
import re
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.preprocessing import LabelEncoder
patts = [r"\d+'s", r"[.,?!/\()-;]+", r"<[\w\/\s]+>+", r'["]+']
stopwords = nltk.corpus.stopwords.words('english')

def preprocess(text, patterns=patts):
    for indx, pattern in enumerate(patterns):
        if indx == len(patterns) - 1:
            text = re.sub(pattern,'', text)
        else:
            text = re.sub(pattern,'', text.lower())
    return text

def simple_tokenizer(text):
    return re.sub(r' +', ' ', text).split(' ')
 
porter = PorterStemmer()
def porter_tokenizer(text):
    return [porter.stem(word) for word in re.sub(r' +', ' ', text).split(' ')]


script_dir = os.path.dirname(__file__)
rel_path = "IMDB Dataset.csv"
abs_file_path = os.path.join(script_dir,
                             rel_path)
                             
data = pd.read_csv(abs_file_path)
encoder = LabelEncoder()
X = data['review']
target = encoder.fit_transform(data['sentiment'])

