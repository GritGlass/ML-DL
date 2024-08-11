import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

@st.cache_resource
def load_data():
    df_agg_sub=pd.read_csv('/Users/graceandrew/Documents/Git/Study/Visualization/data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    return  df_agg_sub
df_agg_sub=load_data()
st.dataframe(df_agg_sub)

#style
def style_negative(v, props=''):
    """ style negative values in dataframe """
    try:
        return props if v<0 else None
    except:
        pass

def style_positive(v,props=''):
    """ style positive values in dataframe"""
    try:
        return props if v>0 else None
    except:
        pass

add_slider=st.selectbox('Aggregate or Individual Video?',('option1','option2','option3'))

if add_slider == 'option1':
    st.write('option1')
    st.dataframe(df_agg_sub[:10])
    st.metric('length',len(df_agg_sub[:10]), 10)
    fig=px.bar(df_agg_sub[:10], x='Views',y='Video Likes Added',color='Country Code',orientation='h')
    st.plotly_chart(fig)
    
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=df_agg_sub[:10].index,y=df_agg_sub[:10]['Average View Percentage'],mode='lines',
                   name='AVERGAE VIEWS',line=dict(color='purple',dash='dash')))
    fig2.add_trace(go.Scatter(x=df_agg_sub[:10].index,y=df_agg_sub[:10]['Average Watch Time'],mode='lines',
                   name='AVERGAE VIEWS',line=dict(color='green',dash='dot')))
    st.plotly_chart(fig2)

if add_slider == 'option2':
    st.write('option2') 
    st.dataframe(df_agg_sub[10:20])
    
if add_slider == 'option3':
    st.write('option3')
    df_agg_sub[20:30]['Average View Percentage']=df_agg_sub[20:30]['Average View Percentage'].apply(lambda x: round(x,1))
    st.dataframe(df_agg_sub[20:30].style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;'))
    
