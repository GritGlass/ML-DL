from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from tabulate import tabulate
import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import cross_validate,GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# data=Dataset.load_builtin('ml-100k')
# train,test=train_test_split(data,test_size=0.25,random_state=0)
#
# algo=SVD()
# algo.fit(train)
# predictions=algo.test(test)
# print('prediction type : ', type(predictions), 'size: ', len(predictions))
# print('prediction 결과의 최초 5개 추출')
# print(predictions[:5])

#인덱스 , 헤더 제거
# ratings=pd.read_csv('./ml-latest-small/ratings.csv')
# ratings.to_csv('./ml-latest-small/ratings_noh.csv',index=False,header=False)

# reader=Reader(line_format='user item rating timestamp',sep=',',rating_scale=(0.5,5))
# data=Dataset.load_from_file('./ml-latest-small/ratings_noh.csv',reader=reader)
#
# train,test=train_test_split(data,test_size=0.25,random_state=0)
# algo=SVD(n_factors=50,random_state=0)
# algo.fit(train)
# predictions=algo.test(test)
# print(accuracy.rmse(predictions))
# #--------------------------------------------------------------------------------------
# ratings=pd.read_csv('./ml-latest-small/ratings.csv')
# reader=Reader(rating_scale=(0.5,5.0))
# data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
# train,test=train_test_split(data,test_size=0.25,random_state=0)
#
# algo=SVD(n_factors=50, random_state=0)
# algo.fit(train)
# predictions=algo.test(test)
# print(accuracy.rmse(predictions))
#
# #----------------------------------------------------------------------------------------
# ratings=pd.read_csv('./ml-latest-small/ratings.csv')
# reader=Reader(rating_scale=(0.5,5.0))
# data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
#
# algo=SVD(random_state=0)
# print(cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True))
#
# param_grid={'n_epochs' : [20,40,60], 'n_factors':[50,100,200]}
# gs=GridSearchCV(SVD,param_grid,measures=['rmse','mae'],cv=3)
# gs.fit(data)
# print(gs.bset_score['rmse'])
# print(gs.bset_params['rmse'])
#-------------------------------------------------------------------------------------------


def get_unseen_surprise(ratings,movies,userId):
    seen_movies=ratings[ratings['userId']==userId]['movieId'].tolist()
    total_movies=movies['movieId'].tolist()
    unseen_movies=[movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화 수 :',len(seen_movies), '추천 대상 영화 수 : ', len(unseen_movies), '전체 영화 수: ',len(total_movies))
    return unseen_movies



def recomm_movie_by_surprise(algo,userId,unseen_movies,top_n=10):
    predictions=[algo.predict(str(userId),str(movieId)) for movieId in unseen_movies]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)

    top_predictions=predictions[:top_n]

    top_movie_ids=[int(pred.iid) for pred in top_predictions]
    top_movie_rating=[pred.est for pred in top_predictions]
    top_movie_titles=movies[movies.movieId.isin(top_movie_ids)]['title']

    top_movie_preds=[(id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    return top_movie_preds




if __name__=='__main__':
    ratings = pd.read_csv('./ml-latest-small/ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)

    train = data_folds.build_full_trainset()

    algo = SVD(n_epochs=20, n_factors=50, random_state=0)
    algo.fit(train)

    movies = pd.read_csv('./ml-latest-small/movies.csv')

    moviesIds = ratings[ratings['userId'] == 9]['movieId']
    unseen_movies = get_unseen_surprise(ratings, movies, 9)

    top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)
    print('### top 10 ###')
    for top_movie in top_movie_preds:
        print(top_movie[1], ":", top_movie[2])