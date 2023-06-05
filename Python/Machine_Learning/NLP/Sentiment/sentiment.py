import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

class wordnet:

    def wn_synset(self,term):
        synsets=wn.synsets(term)
        print('synsets() 반환 type: ', type(synsets))
        print('synsets() 반환 값 개수: ', len(synsets))
        print('synsets() 반환 값: ', synsets,end='\n\n')

        for synset in synsets:
            print('Synset name: ', synset.name())
            print('POS:',synset.lexname())
            print('Definition:', synset.definition())
            print('Lemmas:',synset.lemma_names(),end='\n\n')

    def similarity_wn(self):
        tree=wn.synset('tree.n.01')
        lion=wn.synset('lion.n.01')
        tiger=wn.synset('tiger.n.02')
        cat=wn.synset('cat.n.01')
        dog=wn.synset('dog.n.01')

        entities=[tree,lion,tiger,cat,dog]
        similarities=[]
        entity_names=[entity.name().split('.')[0] for entity in entities]

        #유사도 측정
        for entity in entities:
            similarity=[round(entity.path_similarity(compared_entity),2)
                        for compared_entity in entities]
            similarities.append(similarity)

        similarity_df=pd.DataFrame(similarities, columns=entity_names, index=entity_names)
        return similarity_df

class senti_wordnet:
    def __init__(self,term):
        self.term=term
    def swn_synsets(self):
        senti_synsets=list(swn.senti_synsets(self.term))
        print('senti_synsets() 반환 type : ', type(senti_synsets))
        print('senti_synsets() 반환 값 개수 : ', len(senti_synsets))
        print('senti_synsets() 반환 값 : ', senti_synsets,end='\n\n')

    def swn_synset_score(self,term_synset):

        synset=swn.senti_synset(term_synset)
        print('father 긍정감성 지수: ', synset.pos_score())
        print('father 부정감성 지수: ', synset.neg_score())
        print('father 객관성 지수: ', synset.obj_score(),end='\n\n')

class movie_senti:
    # def __init__(self):

    def penn_to_wn(self,tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB

    def swn_polarity(self,text):
        #초기화
        sentiment=0.0
        tokens_count=0
        final=0

        lemmatizer=WordNetLemmatizer()
        #문서->문장
        raw_sentences=sent_tokenize(text)

        for raw_sentence in raw_sentences:
            #문장->단어
            tagged_sentence=pos_tag(word_tokenize(raw_sentence))
            for word, tag in tagged_sentence:

                #단어 품사 태깅, 어근 추출
                wn_tag=self.penn_to_wn(tag)
                if wn_tag not in (wn.NOUN,wn.ADV,wn.ADV):
                    continue
                lemma=lemmatizer.lemmatize(word,pos=wn_tag)
                if not lemma:
                    continue

                #synset 파악
                synsets=wn.synsets(lemma,pos=wn_tag)
                if not synsets:
                    continue

                #긍정,부정 점수 계산
                synset=synsets[0]
                swn_synset=swn.senti_synset(synset.name())
                sentiment+=(swn_synset.pos_score()-swn_synset.neg_score())
                tokens_count+=1

            if not tokens_count:
                final=0

            #score>=0 이면 긍정, 아니면 부정
            if sentiment >=0:
                final=1

        return final

class Vader:
    def __init__(self):
        self.senti_analyzer=SentimentIntensityAnalyzer()
    def Senti_Score(self,text):
        s_score=self.senti_analyzer.polarity_scores(text)
        return s_score

    def vader_polarity(self,review,threshold=0.1):
        scores=self.Senti_Score(review)

        agg_score=scores['compound']
        final_sentiment= 1 if agg_score >= threshold else 0
        return final_sentiment



if __name__=='__main__':
    check_data(review_df)
    # x_train,y_train,x_test,y_test=data_split(review_df)

    # #지도학습,vector_method : 'cnt_vect'or Tfidf
    # cnt_accuracy,cnt_roc_auc=Supervised_learning(x_train, y_train, x_test, y_test, 'cnt_vect')
    # Tfidf_accuracy, Tfidf_roc_auc = Supervised_learning(x_train, y_train, x_test, y_test, 'Tfidf')

    # print('### wordnet ###')
    # word_ex=wordnet()
    # word_ex.wn_synset('present')
    # word_ex.similarity_wn()
    #
    # print('### senti wordnet ###')
    # senti_ex=senti_wordnet('slow')
    # senti_ex.swn_synsets()
    # senti_ex.swn_synset_score('slow.a.02')

    #WORDNET + SENTIWORDNET
    # train_df=review_df.copy()
    # movie_ex=movie_senti()
    # train_df['preds']=train_df['review'].apply(lambda x: movie_ex.swn_polarity(x))
    # y_target=train_df['sentiment'].values
    # preds=train_df['preds'].values
    #
    # print(confusion_matrix(y_target,preds))
    # print('정확도: ',np.round(accuracy_score(y_target,preds),4))
    # print('정밀도: ', np.round(precision_score(y_target, preds), 4))
    # print('재현율: ', np.round(recall_score(y_target, preds), 4))

    #VADER
    vader=Vader()
    review_df['vader_preds']=review_df['review'].apply(lambda x : vader.vader_polarity(x,0.1))
    y_target=review_df['sentiment'].values
    vader_preds=review_df['vader_preds'].values

    print(confusion_matrix(y_target, vader_preds))
    print('정확도: ', np.round(accuracy_score(y_target, vader_preds), 4))
    print('정밀도: ', np.round(precision_score(y_target, vader_preds), 4))
    print('재현율: ', np.round(recall_score(y_target, vader_preds), 4))