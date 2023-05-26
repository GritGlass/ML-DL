import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from tk import *

retail_df=pd.read_excel('./Online Retail.xlsx')


def data_info(retail_df):
    print(tabulate(retail_df.head()))
    print(retail_df.info(),end='\n\n')
    print('Data Shape : ', retail_df.shape,end='\n\n')
    print(retail_df['Country'].value_counts()[:5])
'''
오류 ->  CustomerID IS NULL & Quantity=0 & UnitPrice=0 삭제
영국만 분석 -> 나머지 국가 삭제
'''
def data_preprocessing(retail_df):
    #오류 삭제
    retail_df = retail_df[retail_df['Quantity']>0]
    retail_df = retail_df[retail_df['UnitPrice'] > 0]
    retail_df = retail_df[retail_df['CustomerID'].notnull()]

    #영국외 나머지 국가 제외
    retail_df = retail_df[retail_df['Country']=='United Kingdom']

    # 변수 타입 변경
    retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)

    #새로운 변수 생성
    retail_df['sale_amount'] = retail_df['Quantity'] * retail_df['UnitPrice']

    print('Data Shape : ', retail_df.shape, end='\n\n')
    print('NUll Check : ', retail_df.isnull().sum())

    '''
    데이터의 마지막 일자가 2011년 12월 9일 이므로 오늘 날짜는 하루 더한 2011 12월 10일로 설정
    Recency   : 고객별 가장 최근 주문 일자 -> ID로 그룹 -> 가장 큰 주문 일자(가장 최근) -> 오늘 일자 - 가장 큰 주문 일자 (일별) 
    Frequency : 고객별 주문 건수 -> ID로 그룹 -> InvoiceNo count
    Monetary  : 고객별 주문 금액 -> ID로 그룹 -> sale_amount sum
    '''
    aggs={
        'InvoiceDate':'max',
        'InvoiceNo':'count',
        'sale_amount':'sum'
    }

    cust_df=retail_df.groupby('CustomerID').agg(aggs)
    cust_df=cust_df.rename(columns={
        'InvoiceDate':'Recency',
        'InvoiceNo':'Frequency',
        'sale_amount':'Monetary'
    }
    )

    cust_df=cust_df.reset_index()

    cust_df['Recency']=dt.datetime(2011,12,10)-cust_df['Recency']
    cust_df['Recency']=cust_df['Recency'].apply(lambda x : x.days+1)

    print('Preprocessed data Shape : ', cust_df.shape, end='\n\n')
    print(tabulate(cust_df.head()))

    return cust_df


def show_plot(cust_df):
    fig, (ax1,ax2,ax3)=plt.subplots(figsize=(12,4),nrows=1,ncols=3)
    ax1.set_title('Recency Histogram')
    ax1.hist(cust_df['Recency'])

    ax2.set_title('Frequency Histogram')
    ax2.hist(cust_df['Frequency'])

    ax3.set_title('Monetary Histogram')
    ax3.hist(cust_df['Monetary'])


def SCALER(cust_df,scaler=None):

    if scaler=='log':
        col_name=['Recency', 'Frequency', 'Monetary']
        for c in col_name:
            cust_df[c]=np.log1p(cust_df[c])

    x_features = cust_df[['Recency', 'Frequency', 'Monetary']].values
    x_features_scaled = StandardScaler().fit_transform(x_features)

    return x_features_scaled

def CLUSTER(model,x_features_scaled,n_clus):
    rs=36
    if model=='kmeans':
        kmeans=KMeans(n_clusters=n_clus, random_state=rs)
        labels=kmeans.fit_predict(x_features_scaled)

    elif model=='meanshift':
        best_bandwidth=estimate_bandwidth(x_features_scaled)
        meanshift=MeanShift(bandwidth=best_bandwidth)
        labels=meanshift.fit_predict(x_features_scaled)

    elif model=='gaussian':
        gmm=GaussianMixture(n_components=n_clus,random_state=rs)
        labels = gmm.fit_predict(x_features_scaled)

    elif model=='dbscan':
        db=DBSCAN(eps=0.6,min_samples=8,metric='euclidean')
        labels = db.fit_predict(x_features_scaled)

    return labels

def Silhouette_SCORE(x_features_scaled,labels):
    score=np.round(silhouette_score(x_features_scaled, labels), 3)
    print('실루엣 스코어 :', score)
    return score


def visualize_silhouette(cluster_lists, X_features):
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산.
        clusterer = KMeans(n_clusters=n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : ' + str(n_cluster) + '\n' \
                                                                     'Silhouette Score :' + str(round(sil_avg, 3)))
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                   facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")



def visualize_kmeans_plot_multi(cluster_lists, X_features):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np

    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1', 'PCA2'])

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장.
        clusterer = KMeans(n_clusters=n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(pca_transformed)
        dataframe['cluster'] = cluster_labels

        unique_labels = np.unique(clusterer.labels_)
        markers = ['o', 's', '^', 'x', '*']

        # 클러스터링 결과값 별로 scatter plot 으로 시각화
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster'] == label]
            if label == -1:
                cluster_legend = 'Noise'
            else:
                cluster_legend = 'Cluster ' + str(label)
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70, \
                             edgecolor='k', marker=markers[label], label=cluster_legend)

        axs[ind].set_title('Number of Cluster : ' + str(n_cluster))
        axs[ind].legend(loc='upper right')





if __name__=='__main__':

    data_info(retail_df)
    cust_df=data_preprocessing(retail_df)
    show_plot(cust_df)
    #데이터 분포확인
    cust_df[['Recency','Frequency','Monetary']].describe()

    #scaler, default=std, other=log+std : 'log'
    x_features_scaled=SCALER(cust_df,scaler='log')

    #n_clus: 군집 개수
    model=['kmeans','meanshift','gaussian','dbscan']
    n_clus=3
    scores=[]
    for m in model:
        print('Cluster Model : ',m)
        labels=CLUSTER(m,x_features_scaled, n_clus)
        cust_df['cluster_label']=labels
        score = Silhouette_SCORE(x_features_scaled, labels)
        scores.append(score)

    result=pd.DataFrame(model,columns=['Model'])
    result['Silhouette Score']=scores
    print(tabulate(result,headers=result.columns))

    cluster_lists=[2,3,4,5]
    # plt.figure(2)
    # visualize_silhouette(cluster_lists, x_features_scaled)
    # plt.figure(3)
    # visualize_kmeans_plot_multi(cluster_lists, x_features_scaled)
    plt.show()