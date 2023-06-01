import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import nltk
from nltk.corpus import wordnet as wn
import warnings
warnings.filterwarnings('ignore')

nltk.download('all')
review_df=pd.read_csv('./labeledTrainData.tsv',header=0,sep='\t',quoting=3)

def check_data(df):
    print(tabulate(df.head(), headers=df.columns),end='\n\n')
    print(df['review'][0])

def data_split(df):
    target=df['sentiment']
    x_features=df.drop(['id','sentiment'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x_features,target,test_size=0.3,random_state=36)
    print('x_train :',x_train.shape,'   y_train:',y_train.shape,'   x_test:',x_test.shape,' y_test',y_test.shape,end='\n')
    return x_train,y_train,x_test,y_test

#지도학습
def Supervised_learning(x_train,y_train,x_test,y_test,vector_method):
    if vector_method=='cnt_vect':
        pipeline=Pipeline([
            ('CNT_vect',CountVectorizer(stop_words='english',ngram_range=(1,2))),
            ('lr',LogisticRegression(C=10))
        ])
    else:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('lr', LogisticRegression(C=10))
        ])

    pipeline.fit(x_train['review'],y_train)
    pred=pipeline.predict(x_test['review'])
    pred_probs=pipeline.predict_proba(x_test['review'])[:,1]
    accuracy=accuracy_score(y_test, pred)
    roc_auc=roc_auc_score(y_test, pred_probs)
    print('Vector:',pipeline[0].__class__.__name__, '   Model:',pipeline[1].__class__.__name__  )
    print('accuracy: ', accuracy, '  ROC-AUC:',roc_auc,end='\n\n')
    return accuracy,roc_auc


def UNS_learning(x_train,y_train,x_test,y_test,vector_method):





if __name__=='__main__':
    check_data(review_df)
    x_train,y_train,x_test,y_test=data_split(review_df)

    #지도학습,vector_method : 'cnt_vect'or Tfidf
    cnt_accuracy,cnt_roc_auc=Supervised_learning(x_train, y_train, x_test, y_test, 'cnt_vect')
    Tfidf_accuracy, Tfidf_roc_auc = Supervised_learning(x_train, y_train, x_test, y_test, 'Tfidf')
