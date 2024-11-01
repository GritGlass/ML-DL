import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.figure_factory as ff
import os
from wordcloud import WordCloud
from PIL import Image
import spacy
import requests
from io import BytesIO


os.chdir='/Users/graceandrew/Documents/Git/Study/Visualization/movie_app'
st.set_page_config(page_title="Movie Chart", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)


@st.cache_data
def load_data():
    df_agg_sub=pd.read_csv('./data/processed_imdb_top_1000.csv')
    return  df_agg_sub


@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

# wordcloud
def wordcloud_genre(df_imdb,genre):
    texts=["".join(i) for i in df_imdb[df_imdb[genre]>0]['Overview']]
    genre_overview="".join(texts)

    # 텍스트 토큰화 (단어로 분리)
    nlp = spacy.load("en_core_web_sm")

    # NLTK의 영어 stopword 목록 불러오기
    doc = nlp(genre_overview)

    # Stopword를 제외한 단어들로 리스트 생성
    filtered_text = " ".join([token.text for token in doc if not token.is_stop])
    genre_overview="".join(filtered_text)

    # 워드클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genre_overview)

    # 워드클라우드를 배열로 변환 (Plotly로 시각화하기 위해)
    image_array = wordcloud.to_array()

    # Plotly를 사용하여 워드클라우드 시각화
    fig = px.imshow(image_array)
    fig.update_xaxes(showticklabels=False)  # x축 레이블 숨기기
    fig.update_yaxes(showticklabels=False)  # y축 레이블 숨기기
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),  # 여백 최소화
    )
    return fig

df_imdb=load_data()

movie_genre=df_imdb.columns.tolist() #영화 장르
before_genre_col=movie_genre.index('Gross')

#pie chart
genre_df=pd.DataFrame(df_imdb.iloc[:,before_genre_col+1:].sum())
genre_df.reset_index(inplace=True)
genre_df=pd.DataFrame(genre_df)
genre_df.columns=['genre','counts']
genre_df=genre_df[genre_df['counts']>0]
genre_pie = px.pie(genre_df, values='counts', names='genre', title='Population of Movie Genre')
pie_event = st.plotly_chart(genre_pie, key="genre", on_select="rerun")


#data
col_list=df_imdb.columns.tolist()
start_col=col_list.index('Gross')
genre_cols=sorted(col_list[start_col+1:])
genre_cols.insert(0,'Total')
add_slider=st.selectbox('Select movie genre',genre_cols)

for ge in genre_cols:
    if add_slider == ge:
        if ge=='Total':
            df_imdb.sort_values(['IMDB_Rating','Meta_score','No_of_Votes'],ascending=False)[['IMDB_Rating','Series_Title','Overview','Runtime(min)','Released_Year','Certificate','Genre']].iloc[:10]
        else:
            genre_sort_df=df_imdb[df_imdb[ge]>0].sort_values(['IMDB_Rating','Meta_score','No_of_Votes'],ascending=False)
            word_fig=wordcloud_genre(df_imdb,ge)
            runtime_event = st.plotly_chart(word_fig)
            st.dataframe(genre_sort_df[['IMDB_Rating','Series_Title','Overview','Runtime(min)','Released_Year','Certificate','Genre']].iloc[:10]) 
        


