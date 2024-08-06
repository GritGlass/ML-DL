import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', 10000000)
movies=pd.read_csv('./ml-latest-small/movies.csv')
ratings=pd.read_csv('./ml-latest-small/ratings.csv')
print("movies's shape", movies.shape)
print("ratings's shape", ratings.shape)

ratings=ratings[['userId','movieId','rating']]
ratings_matrix=ratings.pivot_table('rating',index='userId',columns='movieId')
print(tabulate(ratings_matrix.head(),headers=ratings_matrix.columns))

rating_movies=pd.merge(ratings,movies,on='movieId')
ratings_matrix=rating_movies.pivot_table('rating',index='userId',columns='title')

ratings_matrix=ratings_matrix.fillna(0)
print(tabulate(ratings_matrix.head(3)))
print(ratings_matrix.columns)

ratings_matrix_T=ratings_matrix.transpose()
print(tabulate(ratings_matrix_T.head(3),headers=ratings_matrix_T.columns))

item_sim=cosine_similarity(ratings_matrix_T,ratings_matrix_T)
item_sim_df=pd.DataFrame(data=item_sim, index=ratings_matrix.columns,columns=ratings_matrix.columns)
print("item_sim_df's shape", movies.shape)
print(tabulate(item_sim_df.head(3)))
print(ratings_matrix.columns)

#특정 아이템과 유사도가 높은 영화 추출
def top_sim(df,name):
    return df[name].sort_values(ascending=False)[1:6]

top_6=top_sim(item_sim_df,'Inception (2010)')
print(top_6)

#-----------------------------------------------------------------------------------------------------------------------
#아이템 기반 최근접 이웃 협업 필터링으로 개인 영화 추천
def predict_rating(ratings_arr,item_sim_arr):
    ratings_pred=ratings_arr.dot(item_sim_arr)/np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred


def get_mse(pred,actual):
    pred=pred[actual.nonzero()].flatten()
    actual=actual[actual.nonzero()].flatten()
    return mean_squared_error(pred,actual)


def predict_rating_topsim(ratings_arr,item_sim_arr,n=20):
    pred=np.zeros(rating_arr.shape)

    for col in range(ratings_arr.shape[1]):
        top_n_items=[np.argsort(item_sim_arr[:,col])[:-n-1:-1]]
        for row in range(rating_arr.shape[0]):
            pred[row,col]=item_sim_arr[col,:][top_n_items].dot(ratings_arr[row,:][top_n_items].T)
            pred[row,col] /=np.sum(np.abs(item_sim_arr[col,:][top_n_items]))
    return pred


def get_unseen_movies(ratings_matrix,userId):
    user_rating=ratings_matrix.loc[userId,:]
    already_seen=user_rating[user_rating>0].index.tolist()
    movie_list=ratings_matrix.columns.tolist()
    unseen_list=[movie for movie in movie_list if movie not in already_seen]
    return unseen_list

def recomm_movie_by_userid(pred_df,userId,unseen_list,top_n=10):
    recomm_movies=pred_df.loc[userId,unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies


#-------------------------------------------------------------------------------------------------------------------
#행렬 분해를 이용한 잠재 요인 협업 필터링
def get_rmse(R,P,Q,non_zeros):
    error=0
    full_pred_matrix=np.dot(P,Q.T)

    x_non_zero_ind=[non_zeros[0] for non_zero in non_zeros]
    y_non_zero_ind=[non_zeros[1] for non_zero in non_zeros]
    R_non_zeros=R[x_non_zero_ind,y_non_zero_ind]
    full_pred_matrix_non_zeros=full_pred_matrix[x_non_zero_ind,y_non_zero_ind]
    mse=mean_squared_error(R_non_zeros,full_pred_matrix_non_zeros)
    rmse=np.sqrt(mse)

    return rmse


def matrix_factorization(R,K,steps=200, learning_rate=0.01,r_lambda=0.01):
    num_users,num_items=R.shape
    np.random.seed(1)
    P=np.random.normal(scale=1./K,size=(num_users,K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse=10000
    break_count=0

    non_zeros=[(i,j,R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j]>0]

    for step in range(steps):
        for i,j,r in non_zeros:
            eij=r-np.dot(P[i,:],Q[j,:].T)
            P[i,:]=P[i,:]+learning_rate*(eij*Q[j,:]-r_lambda*P[i,:])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[j, :] - r_lambda * Q[i, :])

            rmse=get_rmse(R,P,Q,non_zeros)
            if (step % 10)==0:
                print("# iteration step: ",step, "rmse : ",rmse)

    return P,Q





if __name__=='__main__':

    # #personal recommendation by item based
    # ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
    # ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=ratings_matrix.index, columns=ratings_matrix.columns)
    # print(tabulate(ratings_pred_matrix.head(1)))
    # print(ratings_pred_matrix.columns)
    #
    # print('item based mse: ', get_mse(ratings_pred, ratings_matrix.values))
    #
    # ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
    # print('아이템 기반 최근점 top-20 이웃 MSE : ', get_mse(ratings_pred, ratings_matrix.values))
    #
    # ratings_pred_matrix = pd.DataFrame(ratings_pred, index=ratings_matrix.index, columns=ratings_matrix.columns)
    # user_rating_id = ratings_matrix.loc[9, :]
    # user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:10]
    #
    # unseen_list = get_unseen_movies(ratings_matrix, 9)
    #
    # recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
    # recomm_movies = pd.DataFrame(recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
    # print(recomm_movies)

    #협업필터링
    movies = pd.read_csv('./ml-latest-small/movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings.csv')
    ratings=ratings[['userId','movieId','rating']]
    ratings_matrix=ratings.pivot_table('rating',index='userId',columns='movieId')

    rating_movies=pd.merge(ratings,movies,on='movieId')
    ratings_matrix=rating_movies.pivot_table('rating',index='userId',columns='title')

    P, Q = matrix_factorization(ratings_matrix.values , 50)
    pred_matrix=np.dot(P,Q.T)

    ratings_pred_matrix=pd.DataFrame(pred_matrix,index=ratings_matrix.index,columns=ratings_matrix.columns)
    print(tabulate(ratings_pred_matrix.head(3),headers=ratings_pred_matrix.columns))

    unseen_list=get_unseen_movies(ratings_matrix,9)
    recomm_movies=recomm_movie_by_userid(ratings_pred_matrix,9,unseen_list,top_n=10)

    recomm_movies=pd.DataFrame(recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])
    print(recomm_movies)
