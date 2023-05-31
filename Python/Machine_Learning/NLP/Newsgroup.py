from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

news_df=fetch_20newsgroups(subset='all',random_state=36)


def check_target(data):
    print('Data keys: ',data.keys(), end='\n\n')
    print('Target class value count \n',pd.Series(data.target).value_counts().sort_index(),end='\n\n')
    print('Target class names\n', data.target_names,end='\n\n')
    print('Data example \n',data.data[0])

#data_usage=['train','test']
def train_test_split(data_usage):
    data=fetch_20newsgroups(subset=data_usage,remove=('headers','footers','quotes'),random_state=36)
    x=data.data
    y=data.target
    return x,y

#-------------------------------------------------------------------------------------------------------------------------------------
def feature_to_vector(x_train,x_test,method):
    if method=='count':
        vect=CountVectorizer().fit(x_train)
    elif method=='tfidf':
        vect=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_df=300).fit(x_train)
    x_train_vect=vect.transform(x_train)
    x_test_vect=vect.transform(x_test)
    print('Vector method :',vect.__class__.__name__,', x_train Shape: ', x_train_vect.shape, ', x_test Shape: ', x_test_vect.shape)
    return x_train_vect, x_test_vect

def trian_model_accuracy(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    accu=accuracy_score(y_test,pred)
    print(model.__class__.__name__,' accuracy : ',np.round(accu,3))
    return model

def find_param(model,x_train,y_train):
    params={'C':[0.01,0.1,1,5,10]}
    grid_cv_model=GridSearchCV(model,param_grid=params,cv=3,scoring='accuracy',verbose=1)
    grid_cv_model.fit(x_train,y_train)
    print(model.__class__.__name__,'best C parameter :', grid_cv_model.best_params_)
    return grid_cv_model
#-------------------------------------------------------------------------------------------------------------------------------------



#pipeline 사용시 별도의 fit()과 transform(), predict()가 필요없음
def pipeline_(x_train,y_train,x_test,y_test):
    #pipeline 만들기
    pipeline=Pipeline([
        ('tfidf_vect',TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_df=300)),
        ('Lr_reg',LogisticRegression(C=10,n_jobs=-1))
    ])

    #gridsearch : 객체 변수에 언더바 2개 붙여서 하이퍼파라미터 이름과 값 설정
    param={'tfidf_vect__ngram_range':[(1,1),(1,2),(1,3)],
           'tfidf_vect__max_df':[100.300,700],
           'Lr_reg__C':[1,5,10]}

    grid_cv_pipe = GridSearchCV(pipeline, param_grid=param, cv=3, scoring='accuracy', verbose=1,n_jobs=-1)
    grid_cv_pipe.fit(x_train,y_train)
    print('Best param: ',grid_cv_pipe.best_params_,'Best score: ', grid_cv_pipe.best_score_)

    pred=grid_cv_pipe.predict(x_test)
    accu = accuracy_score(y_test, pred)
    print(pipeline['Lr_reg'].__class__.__name__,' accuracy : ', np.round(accu, 3))
    return grid_cv_pipe,accu



if __name__=='__main__':
    #data infomation check
    #check_target(news_df)

    #data load
    x_train,y_train = train_test_split('train')
    x_test, y_test = train_test_split('test')
    print('train data size: ', len(x_train), 'test data size: ', len(x_test))

    # pipeline = model: linearregresssion, vector : tfidf
    grid_cv_pipe, accu = pipeline_(x_train, y_train, x_test, y_test)

    #feature to vector, 11314개 문서에서 99238개 단어 추출
    # x_train_vect, x_test_vect=feature_to_vector(x_train,x_test,'tfidf')

    #train model and calculate accuracy score
    # model=LogisticRegression()
    # trained_model=trian_model_accuracy(model,x_train_vect,y_train, x_test_vect, y_test)

    #Gridsearch로 최적 param 찾기
    # best_model=find_param(model, x_train_vect, y_train)
    # pred = best_model.predict(x_test)
    # accu = accuracy_score(y_test, pred)
    # print(model.__class__.__name__, ' accuracy : ', np.round(accu, 3))

