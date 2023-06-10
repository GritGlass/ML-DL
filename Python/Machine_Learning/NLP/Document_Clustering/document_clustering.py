import pandas as pd
import glob, os
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
#---------------------------------------------------------------------------------------------------------------------

#read data
path=os.getcwd()+'/OpinosisDataset1.0/topics'
all_files=glob.glob(os.path.join(path,'*.data'))
filename_list=[]
opinion_text=[]
for file_ in all_files:
    df=pd.read_table(file_,header=None,index_col=False,encoding='latin1')
    text=''.join([df[0][d] for d in range(len(df[0]))])
    filename_=file_.split('/')[-1]
    filename=filename_.split('.')[0]
    filename_list.append(filename)
    opinion_text.append(text)

document_df=pd.DataFrame({'filename':filename_list,'opinion_text':opinion_text})


#---------------------------------------------------------------------------------------------------------------------
#cluster model
def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#---------------------------------------------------------------------------------------------------------------------

def show_cluster(df,la):
    sort_df=df[df['cluster_label']==la].sort_values(by='filename')
    print(tabulate(sort_df.head(),headers=sort_df.columns),end='\n\n')

def get_cluster_details(cluster_model,cluster_data,feature_names,clusters_num,top_n_features=10):
    cluster_details={}

    #center로부터 거리가 먼 데이터 순으로 나열
    centroid_featrue_ordered_ind=cluster_model.cluster_centers_.argsort()[:,::-1]

    for cluster_num in range(clusters_num):

        cluster_details[cluster_num]={}
        cluster_details[cluster_num]['cluster']=cluster_num

        #top n 단어를 구함
        top_feature_indexes=centroid_featrue_ordered_ind[cluster_num:top_n_features]
        top_features=[feature_names[ind] for ind in top_feature_indexes]

        #feature 단어의 중심 위치 상대값
        top_feature_values=cluster_model.cluster_centers_[cluster_num,top_feature_indexes].tolist()

        #cluster_details 딕셔너리 객체에 개별 군집별 핵심단어와 중심위치 상대값, 파일명 입력
        cluster_details[cluster_num]['top_features']=top_features
        cluster_details[cluster_num]['top_features_value']=top_feature_values
        filenames=cluster_data[cluster_data['cluster_label']==cluster_num]['filename']
        filenames=filenames.values.tolist()

        cluster_details[cluster_num]['filenames']=filenames

    return cluster_details

def print_cluster_details(cluster_details):
    print('@@@ Cluster & Top words @@@')
    for cluster_num, cluster_detail in cluster_details.items():
        print('Cluster ',cluster_num)
        print('top words:',cluster_detail['top_features'])
        print('review 파일 명:',cluster_detail['filenames'][:7])
        print('-----------------------------------------------')





if __name__=='__main__':
    #data info
    # print(tabulate(document_df.head(), headers=document_df.columns))

    # lemmatizer set
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    lemmar = WordNetLemmatizer()

    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 2), min_df=0.05,
                                 max_df=0.85)
    feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])

    #cluster
    km_cluster = KMeans(n_clusters=3, max_iter=10000, random_state=0)
    km_cluster.fit(feature_vect)
    cluster_label = km_cluster.labels_
    cluster_centers = km_cluster.cluster_centers_
    document_df['cluster_label'] = cluster_label

    # print('cluster_centers shape:', cluster_centers.shape)
    # print(cluster_centers, end='\n\n')

    # print('category number & data')
    # show_cluster(document_df,0)

    #important words
    feature_names=tfidf_vect.get_feature_names_out()
    cluster_details=get_cluster_details(cluster_model=km_cluster, cluster_data=document_df, feature_names=feature_names, clusters_num=3, top_n_features=10)
    print_cluster_details(cluster_details)