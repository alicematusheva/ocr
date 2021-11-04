from flask import Flask, render_template, request

from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer

import string
import re
import numpy as np
import pandas as pd
import pickle
import gensim

from nltk.corpus import stopwords
import sklearn.preprocessing

app= Flask(__name__)

model = pickle.load(open('app/modelLRw2v.pkl','rb'))
posts2vec = pickle.load(open('app/post2vec.pkl','rb'))
nltk.download('stopwords')
stops = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# for data persistence on the result page
title='Title'
body='Write your message'
prediction_title=''
prediction_text=''

@app.route('/')
@app.route('/index/')
def index():
    #return "<h1>Welcome to P5 Project Template 01:28</h1>"
    return render_template('index.html', title=title, body=body)

def process_text(txt):
	txt = txt.replace('\n',' ')
	txt = txt.replace('\r',' ')
	txt = txt.translate(str.maketrans('', '', string.punctuation))
	tokenizer = nltk.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(txt)
	post = [w.lower() for w in tokens if w.lower() not in stops]
	post = [w for w in post if not re.search(r'\d+',w)]
	post = [w for w in post if len(w)<16 and len(w)>1]
	post = [lemmatizer.lemmatize(w) for w in post]
	post = ' '.join(w for w in post)
	return post

def postVector(row):
	vector_sum = 0
	sentence = row.split()
	for word in sentence:
		vector_sum = vector_sum + posts2vec[word]
	vector_sum = vector_sum.reshape(1,-1)
	normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
	return normalised_vector_sum

def get_tags():
	with open("app/sorted_tags.txt") as file_in:
		lines = []
		for line in file_in:
			lines.append(line.rstrip())
	return lines

@app.route('/predict',methods=['POST'])
def predict():
	
	# feature extraction and transformation
	txt = [str(x) for x in request.form.values()]
	title = txt[0]
	body = txt[1]
	txt = [BeautifulSoup(x, 'html.parser').get_text() for x in txt]
	txt_features = [process_text(x) for x in txt]
	txt_features = ' '.join(t for t in txt_features)
	v_features = postVector(txt_features)
	x = np.array(v_features).reshape((1, 100))
	# prediction
	predictions = model.predict_proba(x)
	t = 0.15 # threshold value
	pred = (predictions >= t).astype(int)
	# pred is final vector where 1 correspond to predicted tags to retrieve from list (file)?
	tags = get_tags()
	pred_df = pd.DataFrame(pred, columns=tags)
	df = pred_df.dot(pred_df.columns + '\n').str.rstrip('\n')
	# the result is df[0]
	resultat = df[0]
	return render_template('index.html', prediction_title="Tags we suggest:", prediction_text=resultat, title=title, body=body)