#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:32:11 2020

@author: ghassen97
"""

#check if i get these changes on local repo
# check on 28/01

#for organization matters, some remarks about the code are written before the code, but understood after running the code.
'''Import'''
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn import metrics
from tqdm import tqdm
import joblib

####################
# Data exploration #
####################

DATA_ROOT= '/home/ghassen/Desktop/Data Science/projects/NLP/text_classification'
df=pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
df.shape # (20000, 4)
df.head()
df.isnull().values.any() # no Null values in df
# number of categories is 20
nbr_categories=len(df.category_id.unique())
print(f'There are {nbr_categories} \n')
df.info()

#Merge description and title into description to get more representative data of the category
df["description"] = df["title"] + ' ' + df["description"] # space for the last word + ' ' + first word of df['descritption']

# df["description"] = df["title"] + df["description"]
# df.description[6]
# we only need id, description in X matrix (train and test)
X= df['description']
y= df['category_id']

df.shape
df.head()

# get unique category_id and category
category_id_category= df[['category_id', 'category']]
category_id_category= category_id_category.drop_duplicates(keep='first', inplace= False)

#Imbalanced Classes#

#the categories are balanced,we have the same amount of each category, 1001 product from each category
#We are lucky here, if there is imbalance one should be careful with algorith choice because they man treat a class as outlier
fig = plt.figure(figsize=(9,6))
df['category_id'].value_counts().plot(kind='bar')
plt.show()

################################################
# data processing -- TF-IDF text representation #
################################################
count_vectorizer = CountVectorizer( min_df=6, ngram_range=(1, 3), stop_words='english') #transformer
X_counts= count_vectorizer.fit_transform(X)
tfidf_transformer= TfidfTransformer(sublinear_tf=True ,norm='l2')
X_tfidf= tfidf_transformer.fit_transform(X_counts)

# save transformers
count_vectorizer_path= os.path.join(DATA_ROOT, 'count_vectorizer.pkl')
tfidf_transformer_path= os.path.join(DATA_ROOT, 'tfidf_transformer.pkl')
# save tfidf transformer
joblib.dump(count_vectorizer, open(count_vectorizer_path,'wb') )
joblib.dump(tfidf_transformer, open(tfidf_transformer_path,'wb') )



########################################
# Model selection: Multiclassification #
########################################

#Random Forest
#Linear Support Vector Machine
#Multinomial Naive Bayes
#Logistic Regression

models = [
    LinearSVC(),
    MultinomialNB(),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=7),
    LogisticRegression(random_state=7)
]
models_names= [type(model).__name__ for model in models]
##CV role have validation set and training set, accuracy is calculated on validation set
CV = 5
# kf = KFold(n_splits=CV)
kf= StratifiedKFold(n_splits=CV, shuffle=True, random_state=7)
rows, columns= 3, CV # 3 for model_name, cv_id, accuracy_id
results = [[None for c in range(columns)] for r in range(rows)]
results = np.asarray(results)
# results_models=[None for i in range(len(models))]
# results_models= np.asarray(results_models)
results_models= dict(zip(models_names, [None for i in range(len(models))] ))
# test for valid here
# for model in models:
for i, model in enumerate(tqdm(models)):
    print(f'i: {i} \n')
    # normalement mrigla ta3mel CV wa7dha
    for j, (train_index, test_index) in enumerate(kf.split(X_tfidf, y)):
        X_train = X_tfidf[train_index]
        y_train = y[train_index]
        X_test = X_tfidf[test_index]
        y_test = y[test_index]

        model_name = type(model).__name__
        accuracies_cv = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=CV)

        results[0,j]=model_name
        results[1,j]= range(CV)[j]
        results[2,j]= accuracies_cv[j]


    print(f" \n results :\n{results.shape} \n {results} ")
    accuracy_avg= np.mean(results[2])
    print(f'mean of accuracies within CV: {accuracy_avg}')

print("the selected model is: LinearSVC with a mean accuracy over CV folders= 0.98" )

####################
# Model evaluation #
####################
model= LinearSVC()
# np.arange(len(y)) here to get index of train and test
X_train, X_test, y_train, y_test, ids_train, ids_test  = train_test_split(X_tfidf, y, np.arange(len(y)), test_size=1/3, random_state=7, stratify= y)
model.fit(X_train, y_train)

y_pred= model.predict(X_test)
labels_unique= y.unique()
# metrics report
print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, y_pred)}\n")

cm= confusion_matrix(y_test, y_pred)

# plot_confusion_matrix(model, X_test, y_test,
#                       labels= LABELS_unique, # or category_id_category.category_id
#                       display_labels= LABELS_unique
#                       # cmap= 'd' )

cm_df= pd.DataFrame(cm, index= labels_unique, columns= labels_unique)

plt.figure()
sns.heatmap(cm_df, annot=True, fmt='g' , cmap=plt.cm.Blues, xticklabels=True, yticklabels=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion matrix without normalization")
plt.show()

# save model to reuse it later without re-training
model_path= os.path.join(DATA_ROOT, 'products_classifier.pkl')
joblib.dump(model, open(model_path,'wb') )
# load model
classifier= joblib.load(open(model_path,'rb'))

######################
# Misclassifications #
######################
# follow cm horizontally, i.e take each true_value and see the predicted_values
# index from y_test and y_pred , not from df
ids_misclassifications_y= np.where(y_test.array != y_pred )
ids_misclassifications_df= ids_test[ids_misclassifications_y]


# build MC dataframe
# all infos in MC are about y_test which is category_id
d= {'y_test':y_test.array[ids_misclassifications_y]  ,'y_pred':y_pred[ids_misclassifications_y]  ,'title':df.iloc[ids_misclassifications_df]['title'].array  ,'description' :df.iloc[ids_misclassifications_df]['description'].array, 'category_id':df.iloc[ids_misclassifications_df]['category_id'].array, 'category':df.iloc[ids_misclassifications_df]['category'].array}
MC= pd.DataFrame(data= d)
MC.to_csv(path_or_buf= os.path.join(DATA_ROOT,'MC.csv'), index=False)
#TODO reorder the csv


##########
# Deploy #
##########

# Example: predict one product based on its description:
element= "Fissure FissureDe Patricia Girod aux éditions IXCEA"
element= "Coque souple Blanche pour SAMSUNG GALAXY S3 / I… Coque souple Blanche pour SAMSUNG GALAXY S3 / I… Coque souple Blanche pour SAMSUNG GALAXY S3 / I9300 motif Orange mécanique + 3 Films - Coque souple Ultra Fine Blanche ORIGINALE de MUZZANO au motif Orange mécanique pour SAMS… Voir la présentation"
# element= "PIANINO 47 NOELS ALSACIENS PIANINO 47 NOELS ALSACIENS PIANINO 47 NOELS ALSACIENS - COLLECTIF"
element= [element]

element_counts = count_vectorizer.transform(element).toarray()
element_tfidf= tfidf_transformer.transform(element_counts).toarray()

element_tfidf_pred= classifier.predict(element_tfidf)






