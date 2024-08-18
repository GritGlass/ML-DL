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

@st.cache_data
def load_data():
    df_agg_sub=pd.read_csv('./data/processed_imdb_top_1000.csv')
    return  df_agg_sub

st.title("Movie Chart")
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
st.divider() 
st.subheader("Genre", divider=False)


#GENRE 
#pie chart

genre_df=pd.DataFrame(df_imdb.iloc[:,before_genre_col+1:].sum())
genre_df.reset_index(inplace=True)
genre_df=pd.DataFrame(genre_df)
genre_df.columns=['genre','counts']
genre_df=genre_df[genre_df['counts']>0]
genre_pie = px.pie(genre_df, values='counts', names='genre', title='Population of Movie Genre')
pie_event = st.plotly_chart(genre_pie, key="genre", on_select="rerun")



#Tree chart
genre_tree = px.treemap(genre_df, 
                 path=['genre'], 
                 values='counts', 
                 title="Movie Genre Distribution")
tree_event = st.plotly_chart(genre_tree)
st.divider()

#RUNTIME
#data
st.subheader("Movie Runtime", divider=False)
genre=df_imdb.columns.tolist()[before_genre_col+1:]
avg_run_time=[]
for g in genre:
    avg_run_time.append(round(df_imdb[df_imdb[g]>0]['Runtime(min)'].mean(),2))
runtime=pd.DataFrame(genre,columns=['genre'])
runtime['avg_runtime']=avg_run_time
runtime['total_avg']=runtime['avg_runtime'].mean()
runtime.dropna(inplace=True)

# Bar Chart 생성
runtime_fig = go.Figure()
# Bar Chart 추가
runtime_fig.add_trace(go.Bar(
    x=runtime['genre'],
    y=runtime['avg_runtime'],
    name='Bar Chart'
))


# Line Chart 추가
runtime_fig.add_trace(go.Scatter(
    x=runtime['genre'],
    y=runtime['total_avg'],
    line = dict(color='blueviolet', width=4, dash='dash'),
    mode='lines',
    name='total avg runtime'
))


# 레이아웃 설정
runtime_fig.update_layout(
    xaxis_title="Genre",
    yaxis_title="Average Runtime",
    xaxis_tickangle=0,
    width=2000,
    margin=dict(autoexpand=True)
)

# 그래프 표시
runtime_event = st.plotly_chart(runtime_fig)

#get poster image
def get_image(link):
    # 인터넷에서 이미지 가져오기
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    return img

#the shortest runtime movie
st.write('The most shoooortest runtime movie')
shortest_runtime=df_imdb[df_imdb['Runtime(min)']==df_imdb['Runtime(min)'].min()]
st.dataframe(shortest_runtime)
# Streamlit에서 이미지 표시
short_poster=get_image(shortest_runtime['Poster_Link'].item())
st.image(short_poster,width=100,caption=shortest_runtime['Series_Title'].item(), use_column_width=False)


#the loggest runtime movie
st.write('The most loooongest runtime movie')
loggest_runtime=df_imdb[df_imdb['Runtime(min)']==df_imdb['Runtime(min)'].max()]
st.dataframe(loggest_runtime)
# Streamlit에서 이미지 표시
long_poster=get_image(loggest_runtime['Poster_Link'].item())



# Streamlit에서 컨테이너 사용
st.subheader("The most loooongest runtime movie", divider=False)
with st.container():
    # 2개의 컬럼 생성
    col1, col2 = st.columns(2)
    
    # 첫 번째 컬럼에 이미지 추가
    with col1:
        st.markdown("#### Poster")
        st.image(long_poster,width=100,caption=loggest_runtime['Series_Title'].item(), use_column_width=False)

    # 두 번째 컬럼에 텍스트 추가
    with col2:
        st.markdown("#### Title")
        st.write(loggest_runtime['Series_Title'].item())
        st.markdown("#### Director")
        st.write(loggest_runtime['Director'].item())
        st.markdown("#### Year")
        st.write(loggest_runtime['Released_Year'].item())
        st.markdown("#### Genre")
        st.write(loggest_runtime['Genre'].item())
        st.markdown("#### Overview")
        st.write(loggest_runtime['Overview'].item())
        


#top movies
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

st.divider()
st.subheader("Top-10", divider=False)
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
        




#stylessss
# def style_negative(v, props=''):
#     """ style negative values in dataframe """
#     try:
#         return props if v<0 else None
#     except:
#         pass

# def style_positive(v,props=''):
#     """ style positive values in dataframe"""
#     try:
#         return props if v>0 else None
#     except:
#         pass


    
