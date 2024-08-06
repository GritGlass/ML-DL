import pandas as pd
import numpy as np
from tabulate import tabulate
import re
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_df=pd.read_csv('ratings_train.txt',sep='\t')
print(tabulate(train_df.head(),headers=train_df.columns),end='\n\n')
print(train_df['label'].value_counts())

train_df=train_df.fillna(' ')
train_df['document']=train_df['document'].apply(lambda x:re.sub(r"\d+"," ",x))

test_df=pd.read_csv('ratings_train.txt',sep='\t')
test_df=test_df.fillna(' ')
test_df['document']=test_df['document'].apply(lambda x : re.sub(r"\d"," ",x))

train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)

twitter=Twitter()
def tw_tokenizer(text):
    tokens_ko=twitter.morphs(text)
    return tokens_ko

tfidf_vect=TfidfVectorizer(tokenizer=tw_tokenizer,ngram_range=(1,2),min_df=3,max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train=tfidf_vect.transform(train_df['document'])

logistic=LogisticRegression(random_state=36)

params={'C':[1,3.5,4.5,5.5,10]}
grid_cv=GridSearchCV(logistic,param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(tfidf_matrix_train,train_df['label'])
print(grid_cv.best_params_, round(grid_cv.best_score_,4))

tfidf_matrix_test=tfidf_vect.transform(test_df['document'])

best_estimator=grid_cv.best_estimator_
preds=best_estimator.predict(tfidf_matrix_test)

print('accuracy : ', accuracy_score(test_df['label'],preds))