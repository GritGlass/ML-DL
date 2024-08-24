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


df_imdb=load_data()

movie_genre=df_imdb.columns.tolist() #영화 장르
before_genre_col=movie_genre.index('Gross')
df_imdb['Genre']=["".join(i) for i in df_imdb['Genre']]
# st.dataframe(df_imdb.iloc[:,:before_genre_col]) #raw data


st.subheader("Total Data", divider=True)

options = st.multiselect(
    "Select genre",
    movie_genre[before_genre_col+1:]
)

a=[df_imdb[df_imdb[i]>0].index.tolist() for i in options]
b=sum(a,[])
rows_index=list(set(b))
genre_col=movie_genre[before_genre_col+1]
st.data_editor(
    df_imdb.loc[rows_index,:genre_col],
    column_config={
        "Poster_Link": st.column_config.ImageColumn(
            "Poster", help="Streamlit app preview screenshots"
        )
    },
    hide_index=True,
)