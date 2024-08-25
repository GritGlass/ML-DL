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
import matplotlib.pyplot as plt
import random 

random.seed(36)
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

#genre list
genre_list=sorted(movie_genre[before_genre_col+1:])
genre_list.insert(0,'Total')

#director list
director_list=sorted(df_imdb['Director'].unique().tolist())
director_list.insert(0,'Total')

#year list
year_min=df_imdb['Released_Year'].min()
year_max=df_imdb['Released_Year'].max()
year_range=np.arange(year_min,year_max+1)

#rating_range
rating_min=df_imdb['IMDB_Rating'].min()
rating_max=df_imdb['IMDB_Rating'].max()
rating_range=np.arange(rating_min,rating_max+1,0.1)
rating_range=list(map(lambda x: round(x,2),rating_range))
with st.sidebar:
    st.header('Seletct')
    seleted_genre=st.selectbox(
        "Genre",
        genre_list)
    
    seleted_director=st.selectbox(
        "Director",
        director_list)
    
    min_year, max_year = st.select_slider(
    "Year",
    options=year_range,
    value=(year_min,year_max))

    min_rating, max_rating = st.select_slider(
    "Rating",
    options=rating_range,
    value=(rating_min, rating_max))

st.header("Total Data", divider=True)

if seleted_genre=='Total':
    genre_const=True
else: 
    genre_const=(df_imdb[seleted_genre]>0)

if seleted_director=='Total':
    director_const=True
else: 
    director_const=(df_imdb['Director']==seleted_director)

year_const=((df_imdb['Released_Year']>=min_year)&(df_imdb['Released_Year']<=max_year))
rating_const=((df_imdb['IMDB_Rating']>=min_rating)&(df_imdb['IMDB_Rating']<=max_rating))

selected_movies=df_imdb[genre_const&director_const&year_const&rating_const]
df_event=st.dataframe(
    selected_movies,
    column_config={
        "Poster_Link": st.column_config.ImageColumn(
            "Poster", help="Streamlit app preview screenshots"
        )
    },
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
)

st.divider()
st.subheader('Detail information')

def make_total_graph(df_imdb,feature,color=None):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df_imdb['Released_Year'],
        y=df_imdb[feature],
        mode='markers',
        marker=dict(color=color),
        name='ALL'
    ))
    return fig


def add_graph(fig,row_movie,feature):

    fig.add_trace(go.Scatter(
        x=[row_movie['Released_Year']],
        y=[row_movie[feature]],
        mode='markers',
        marker=dict(color='red', symbol='star',line=dict(color='black',width=1),size=12),
        name=str(row_movie['Series_Title'])
    ))
    # 레이아웃 설정
    fig.update_layout(
        title=str(feature),
        xaxis_title="Year",
        yaxis_title=str(feature),
        showlegend=True,
        yaxis=dict(showticklabels=False)  # y축 라벨을 숨김
    )
    return fig
  

try:
    detailed_movies=df_event.get('selection').get('rows')
    IMDB_Rating_fig=make_total_graph(df_imdb,'IMDB_Rating')
    Runtime_fig=make_total_graph(df_imdb,'Runtime(min)')
    Gross_fig=make_total_graph(df_imdb,'Gross')
 
    row_index=detailed_movies[0]
    row_movie=selected_movies.loc[row_index]

    row_rating=row_movie['IMDB_Rating']
    row_rating=row_movie['Runtime(min)']
    row_rating=row_movie['Gross']
    row_rating=row_movie['Certificate']

    updated_IMDB_fig=add_graph(IMDB_Rating_fig,row_movie,'IMDB_Rating')
    runtime_fig=add_graph(Runtime_fig,row_movie,'Runtime(min)')
    gross_fig=add_graph(Gross_fig,row_movie,'Gross')

    # Streamlit에서 Plotly 차트 표시
    st.plotly_chart(updated_IMDB_fig)
    st.plotly_chart(runtime_fig)
    st.plotly_chart(gross_fig)
    
except:
    st.write('Click the dataframe row')
    

st.divider() 

def get_image(link):
    # 인터넷에서 이미지 가져오기
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    return img

poster,content=st.columns(2)
with poster:
    #shortest
    st.markdown(f"##### The selected movie - ({row_movie['Runtime(min)'].item()} min) <br><br>",unsafe_allow_html=True)
    with st.container():
        po_col1,po_col2,po_col3=st.columns([1,5,1])
        # Streamlit에서 이미지 표시
        short_poster=get_image(row_movie['Poster_Link'])
        with po_col2:
            st.image(short_poster,width=350,caption=row_movie['Series_Title'], use_column_width=False)


with content:
    with st.container():
        st.markdown('<br><br>',unsafe_allow_html=True)
        st.markdown(f"""##### Title    
                    {row_movie['Series_Title']} """)
        st.markdown(f"""##### Director    
                    {row_movie['Director']}""")
        st.markdown(f"""##### Year    
                    {row_movie['Released_Year']}""")        
        st.markdown(f"""##### Genre     
                    {row_movie['Genre']} """)
        st.markdown(f"""##### Overview    
                    {row_movie['Overview']} """)
        st.markdown('<br><br><br><br><br>',unsafe_allow_html=True)

                           
st.divider() 
