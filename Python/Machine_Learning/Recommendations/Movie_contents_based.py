import pandas as pd
import numpy as np
from tabulate import tabulate
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth',100)

movies=pd.read_csv('tmdb_5000_movies.csv')
print("movies's shape : ", movies.shape)
print(tabulate(movies.head(),headers=movies.columns))

movies_df=movies[['id','title','genres','vote_average','vote_count','popularity','keywords','overview']]
print(tabulate(movies_df[['genres','keywords']][:1]))

movies_df['genres']=movies_df['genres'].apply(literal_eval)
movies_df['keywords']=movies_df['keywords'].apply(literal_eval)


movies_df['genres']=movies_df['genres'].apply(lambda x: [y['name'] for y in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x: [y['name'] for y in x])
print(tabulate(movies_df[['genres','keywords']][:1]))

movies_df['genres_literal']=movies_df['genres'].apply(lambda x : (' ').join(x))
print(tabulate(movies_df['genres_literal'][:1][0]))
count_vect=CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat=count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)

genre_sim=cosine_similarity(genre_mat,genre_mat)
print('genre_sim shape: ',genre_sim.shape)
print('genre_sim : \n', tabulate(genre_sim[:1]))

genre_sim_sorted_ind=genre_sim.argsort()[:,::-1]
print('genre_sim_sorted_ind \n', genre_sim_sorted_ind[:1])
#-----------------------------------------------------------------------------------------------------------------------
def find_sim_movie(df,sorted_ind,title_name,top_n=10):
    title_movie=df[df['title'] == title_name]

    title_index=title_movie.index.values
    similar_indexes=sorted_ind[title_index, :(top_n)]
    print('similar_indexes', similar_indexes)
    similar_indexes=similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]

similar_movies=find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title','vote_average']]
movies_df[['title','vote_average','vote_count']].sort_values('vote_average',ascending=False)[:10]
C=movies_df['vote_average'].mean()
m=movies_df['vote_count'].quantile(0.6)
print('C:',round(C,3), 'm:',round(m,3))

percentile=0.6
m=movies_df['vote_count'].quantile(percentile)
C=movies_df['vote_average'].mean()

def weighted_vote_average(record):
    v=record['vote_count']
    R=record['vote_average']

    return ((v/(v+m))*R)+((m/(m+v))*C)

movies_df['weighted_vote']=movies.apply(weighted_vote_average,axis=1)
movies_df[['title','vote_average','weighted_vote','vote_count']].sort_values('weighted_vote',ascending=False)[:10]

def find_sim_movie(df,sorted_ind,title_name,top_n=10):
    title_movie=df[df['title']==title_name]
    title_index=title_movie.index.values

    similar_indexes=sorted_ind[title_index,:(top_n*2)]
    similar_indexes=similar_indexes.reshape(-1)

    similar_indexes=similar_indexes[similar_indexes != title_index]

    return df.iloc[similar_indexes].sort_values('weighted_vote',ascending=False)[:top_n]

similar_movies=find_sim_movie(movies_df,genre_sim_sorted_ind,'The Godfather',10)
print(tabulate(similar_movies[['title','vote_average','weighted_vote']],headers=similar_movies.columns))

