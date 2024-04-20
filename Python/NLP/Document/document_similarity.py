import numpy as np
import pandas as pd
import glob, os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def cos_simil(v1,v2):
    dot_product=np.dot(v1,v2)
    l2_norm=(np.sqrt(sum(v1*v1))*np.sqrt(sum(v2*v2)))
    similarity=dot_product/l2_norm
    return similarity

def cos_similarity_example():
    doc_list=['if you take the blue pill, the story ends',
              'if you take the red pill, you stay in wonderland',
              'if you take the yellow pill, i show you how deep the rabbit hole goes']

    #text-> vectorize
    tfidf_vect=TfidfVectorizer()
    feature_vect=tfidf_vect.fit_transform(doc_list)
    feature_vect_dense=feature_vect.todense()
    print('feature names')
    print(tfidf_vect.get_feature_names_out(),end='\n\n')
    #CRS형태의 희소행렬
    print('feature vector by sparse matrix (CRS)')
    print(feature_vect,end='\n\n')
    #밀집행렬
    print('feature vector by dense matrix')
    print(feature_vect_dense,end='\n\n')

    #cosine 게산을 위해 reshape
    vect1=np.array(feature_vect_dense[0]).reshape(-1)
    vect2=np.array(feature_vect_dense[1]).reshape(-1)
    vect3=np.array(feature_vect_dense[2]).reshape(-1)

    #직접 만든 함수와 sklearn 모듈 비교
    simil_vect1_2=cos_simil(vect1,vect2)
    similarity_vect1_2=cosine_similarity((vect1,vect2))
    print('My module cosine_simil:', round(simil_vect1_2,8), '\nsklearn cosine_similarity:\n', similarity_vect1_2,end='\n\n')

    #sklearn 모듈은 희소 행렬, 밀집 행렬 모두 가능
    print('sklearn - dense matrix')
    similarity_vect1_2_3=cosine_similarity((vect1,vect2,vect3))
    print(similarity_vect1_2_3,end='\n\n')

    print('sklearn - sparse matrix')
    similarity_vect_all=cosine_similarity(feature_vect,feature_vect)
    print(similarity_vect_all)


def read_data():
    # read data
    path = os.getcwd() + '/OpinosisDataset1.0/topics'
    all_files = glob.glob(os.path.join(path, '*.data'))
    filename_list = []
    opinion_text = []
    for file_ in all_files:
        df = pd.read_table(file_, header=None, index_col=False, encoding='latin1')
        text = ''.join([df[0][d] for d in range(len(df[0]))])
        filename_ = file_.split('/')[-1]
        filename = filename_.split('.')[0]
        filename_list.append(filename)
        opinion_text.append(text)

    document_df = pd.DataFrame({'filename': filename_list, 'opinion_text': opinion_text})
    return document_df


def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def similarity_plot(document_df,doc_similarity,hotel_id):
    sorted_id=doc_similarity.argsort()[:,::-1]
    sorted_id=sorted_id[:,1:]

    hotel_sorted_id=hotel_id[sorted_id.reshape(-1)]

    hotel_1_sim_value=np.sort(doc_similarity.reshape(-1))[::-1]
    hotel_1_sim_value=hotel_1_sim_value[1:]

    hotel_1_sim_df=pd.DataFrame()
    hotel_1_sim_df['filename']=document_df.iloc[hotel_sorted_id]['filename']
    hotel_1_sim_df['similarity']=hotel_1_sim_value

    sns.barplot(x='similarity',y='filename',data=hotel_1_sim_df)
    plt.title('comparison document')
    plt.show()


if __name__=='__main__':
    # cos_similarity_example()
    document_df=read_data()

    # text to vector
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    lemmar = WordNetLemmatizer()

    tfidf_vect=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english',ngram_range=(1,2),min_df=0.05,max_df=0.85)
    feature_vect=tfidf_vect.fit_transform(document_df['opinion_text'])

    #cluster model
    km_cluster=KMeans(n_clusters=3,max_iter=10000,random_state=36)
    km_cluster.fit(feature_vect)
    cluster_label=km_cluster.labels_
    cluster_centers=km_cluster.cluster_centers_
    document_df['cluster_label']=cluster_label

    #cosine similarity
    hotel_id=document_df[document_df['cluster_label']==1].index
    print('hotel로 분류된 문서 번호', hotel_id)
    compar_doc=document_df.iloc[hotel_id[0]]['filename']
    print('기준 문서명', compar_doc)
    doc_similarity=cosine_similarity(feature_vect[hotel_id[0]],feature_vect[hotel_id])
    print(doc_similarity)

    #show plot
    similarity_plot(document_df, doc_similarity, hotel_id)