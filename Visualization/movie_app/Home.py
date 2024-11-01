import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.figure_factory as ff
import os
from PIL import Image
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

#---------------------------------------------------------------------------------------
# movie raw data set
st.title("Movies")
st.divider()
df_imdb=load_data() #data

col1, col2, col3 = st.columns([3,3,1])

with col1:
    st.metric(label="Total Movies", value=len(df_imdb))

with col2:
    st.container()

with col3:
    csv = convert_df(df_imdb)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="movie_df.csv",
        mime="text/csv",
    )

movie_genre=df_imdb.columns.tolist() #영화 장르
before_genre_col=movie_genre.index('Gross')
genre_col=movie_genre[before_genre_col+1]
st.data_editor(
    df_imdb.loc[:,:genre_col].sort_values('IMDB_Rating',ascending=False),
    column_config={
        "Poster_Link": st.column_config.ImageColumn(
            "Poster", help="Streamlit app preview screenshots"
        )
    },
    hide_index=True,
)

st.divider() 
#---------------------------------------------------------------------------------------
#pie chart
st.subheader("Movie Genre Ratio", divider=False)
genre_df=pd.DataFrame(df_imdb.iloc[:,before_genre_col+1:].sum())
genre_df.reset_index(inplace=True)
genre_df=pd.DataFrame(genre_df)
genre_df.columns=['genre','counts']
genre_df=genre_df[genre_df['counts']>0]
genre_pie = px.pie(genre_df, values='counts', names='genre')
pie_event = st.plotly_chart(genre_pie, key="genre", on_select="rerun")
st.divider() 
#---------------------------------------------------------------------------------------
# released year

movie_genre=df_imdb.columns.tolist()
before_genre_col=movie_genre.index('Gross') #gross index+1 = genre columns

#RUNTIME
#data
st.subheader("Number of movies by year", divider=False)
year_df=df_imdb.groupby('Released_Year').count().reset_index()
avg_released_movie_n=year_df['Series_Title'].sum()/len(year_df['Released_Year'].unique())

# Bar Chart 생성
runtime_fig = go.Figure()
# Bar Chart 추가
runtime_fig.add_trace(go.Bar(
    x=year_df['Released_Year'],
    y=year_df['Series_Title'],
    name='Bar Chart',
    marker_color='indianred'
))


# Line Chart 추가
runtime_fig.add_trace(go.Scatter(
    x=year_df['Released_Year'],
    y=[avg_released_movie_n]*len(year_df['Released_Year']),
    line = dict(color='white', width=2, dash='dash'),
    mode='lines',
    name='avg'
))


# 레이아웃 설정
runtime_fig.update_layout(
    xaxis_title="Genre",
    yaxis_title="Number of Released Movie",
    xaxis_tickangle=0,
    width=2000,
    margin=dict(autoexpand=True)
)

# 그래프 표시
runtime_event = st.plotly_chart(runtime_fig)

st.divider()
#---------------------------------------------------------------------------------------

# runtime chart

movie_genre=df_imdb.columns.tolist()
before_genre_col=movie_genre.index('Gross') #gross index+1 = genre columns

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
    name='Bar Chart',
))


# Line Chart 추가
runtime_fig.add_trace(go.Scatter(
    x=runtime['genre'],
    y=runtime['total_avg'],
    line = dict(color='blue', width=2, dash='dash'),
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
#---------------------------------------------------------------------------------------
#gross 그래프
st.subheader("Gross", divider=False)
gross_fig=go.Figure()
gross_fig.add_trace(go.Scatter(
        x=df_imdb['Released_Year'],
        y=df_imdb['Gross'],
        mode='markers',
        marker=dict(color='orange',line=dict(color='black',width=1)),
        name=str(df_imdb['Series_Title'])))

gross_df=pd.DataFrame(df_imdb[['Gross','Released_Year']])
gross_df['avg_gross']=gross_df['Gross'].mean()

#Line Chart 추가
gross_fig.add_trace(go.Scatter(
    x=gross_df['Released_Year'],
    y=gross_df['avg_gross'],
    line = dict(color='red', width=2, dash='dash'),
    mode='lines',
    name='avg_gross'
))

# 레이아웃 설정
gross_fig.update_layout(
    title=str('Gross'),
    xaxis_title="Year",
    yaxis_title=str('Gross'),
    xaxis=dict(range=[gross_df['Released_Year'].min(), gross_df['Released_Year'].max()]),
    showlegend=False,
    yaxis=dict(showticklabels=False)  # y축 라벨을 숨김
)
gross_event = st.plotly_chart(gross_fig)
#---------------------------------------------------------------------------------------
#longest runtime & shortest runtime movies
#get poster image
def get_image(link):
    # 인터넷에서 이미지 가져오기
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    return img

shortest_runtime=df_imdb[df_imdb['Runtime(min)']==df_imdb['Runtime(min)'].min()]
loggest_runtime=df_imdb[df_imdb['Runtime(min)']==df_imdb['Runtime(min)'].max()]



poster,content=st.columns(2)
with poster:
    #shortest
    st.markdown(f"##### The shortest movie - ({shortest_runtime['Runtime(min)'].item()} min) <br><br>",unsafe_allow_html=True)
    with st.container():
        po_col1,po_col2,po_col3=st.columns([1,5,1])
        # Streamlit에서 이미지 표시
        short_poster=get_image(shortest_runtime['Poster_Link'].item())
        with po_col2:
            st.image(short_poster,width=350,caption=shortest_runtime['Series_Title'].item(), use_column_width=False)
    #longest
    st.markdown('<br><br>',unsafe_allow_html=True)
    st.markdown(f"##### The longset movie - ({loggest_runtime['Runtime(min)'].item()} min) <br><br>",unsafe_allow_html=True)
    with st.container():
        po_col1,po_col2,po_col3=st.columns([1,5,1])
        # Streamlit에서 이미지 표시
        long_poster=get_image(loggest_runtime['Poster_Link'].item())
        with po_col2:
            st.image(long_poster,width=350,caption=loggest_runtime['Series_Title'].item(), use_column_width=False)

with content:
    with st.container():
        st.markdown('<br><br>',unsafe_allow_html=True)
        st.markdown(f"""##### Title    
                    {shortest_runtime['Series_Title'].item()} """)
        st.markdown(f"""##### Director    
                    {shortest_runtime['Director'].item()}""")
        st.markdown(f"""##### Year    
                    {shortest_runtime['Released_Year'].item()}""")        
        st.markdown(f"""##### Genre     
                    {shortest_runtime['Genre'].item()} """)
        st.markdown(f"""##### Overview    
                    {shortest_runtime['Overview'].item()} """)
        st.markdown('<br><br><br><br><br>',unsafe_allow_html=True)

    with st.container():
        st.markdown(f"""##### Title    
                    {loggest_runtime['Series_Title'].item()} """)
        st.markdown(f"""##### Director    
                    {loggest_runtime['Director'].item()}""")
        st.markdown(f"""##### Year    
                    {loggest_runtime['Released_Year'].item()}""")        
        st.markdown(f"""##### Genre     
                    {loggest_runtime['Genre'].item()} """)
        st.markdown(f"""##### Overview    
                    {loggest_runtime['Overview'].item()} """)      
                           
st.divider() 
#---------------------------------------------------------------------------------------
st.subheader("Director", divider=False)


director=df_imdb.groupby(['Director','Genre']).count().reset_index()
st.metric(label="Total Directors", value=len(director['Director'].unique()))

director_fig = px.treemap(director, path=[px.Constant("Director"), 'Director', 'Genre'], values='Series_Title',
                  color='Series_Title', hover_data=['Series_Title'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(director['Series_Title'], weights=director['Series_Title']))
director_fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
director_event = st.plotly_chart(director_fig, theme="streamlit", use_container_width=True)
