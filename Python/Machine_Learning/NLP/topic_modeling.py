from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tabulate import tabulate
topic=['rec.motorcycles','rec.sport.baseball','comp.graphics','comp.windows.x','talk.politics.mideast','soc.religion.christian','sci.electronics','sci.med']

news_df=fetch_20newsgroups(subset='all',remove=('headers','footers','quotes'),categories=topic, random_state=0)

count_vect=CountVectorizer(max_df=0.95,max_features=1000,min_df=2,stop_words='english',ngram_range=(1,2))
feat_vect=count_vect.fit_transform(news_df.data)
print('CountVectorizer Shape:', feat_vect.shape)

lda=LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(feat_vect)
print(lda.components_.shape)
print(tabulate(lda.components_))

def display_topics(model, feature_name,no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic - ', topic_index)

        #components에서 가장 값이 큰 순으로 정렬, 그 값의 인덱스 반환
        topic_word_indexes=topic.argsort()[::-1]
        top_indexes=topic_word_indexes[:no_top_words]

        #top_indexes대상인 인덱스별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
        feature_concat=''.join([feature_name[i] for i in top_indexes])
        print(feature_concat)

feature_names=count_vect.get_feature_names_out()

if __name__=='__main__':
    display_topics(lda,feature_names,15)