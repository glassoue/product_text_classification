import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn import metrics

from flask import Flask,render_template,url_for,request
import pickle
# from sklearn.externals import joblib
import joblib
app = Flask(__name__)

# constats need to load transformer, model
DATA_ROOT= '/home/ghassen/Desktop/Data Science/projects/NLP/text_classification'

count_vectorizer_path= os.path.join(DATA_ROOT, 'count_vectorizer.pkl')
tfidf_transformer_path= os.path.join(DATA_ROOT, 'tfidf_transformer.pkl')
model_path= os.path.join(DATA_ROOT, 'products_classifier.pkl')

@app.route('/')
def home():
	return render_template('home.html')
	# return render_template(os.path.join(DATA_ROOT, 'deploy/templates/home.html'))

@app.route('/predict',methods=['POST'])
def predict():

	# load transforer
	count_vectorizer = joblib.load(open(count_vectorizer_path, 'rb'))
	tfidf_transformer = joblib.load(open(tfidf_transformer_path, 'rb'))

	classifier_pkl = open(os.path.join(DATA_ROOT, 'products_classifier.pkl'), 'rb')
	classifier = joblib.load(classifier_pkl)

	if request.method == 'POST':
		element = request.form['message']

		element = [element]
		element_counts = count_vectorizer.transform(element)
		element_tfidf = tfidf_transformer.transform(element_counts)
		pred = classifier.predict(element_tfidf)

	return render_template('result.html', prediction=pred)
	# return render_template( os.path.join(DATA_ROOT,'deploy/templates/result.html') ,prediction = pred)


if __name__ == '__main__':
	app.run(debug=True)