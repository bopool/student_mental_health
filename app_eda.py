import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_app_eda() : 
    st.subheader('데이터 분석') # st. 쓰려고 보니까 streamlit이 안 불러졌네. 그래서 맨 위에 import 시켜줌. 
    df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')
    print(df)

    if st.checkbox('데이터 프레임 보기') : 
        st.dataframe(df)

    if st.subheader('기본 통계 데이터') : 
        st.dataframe(df.describe())
    
    if st.subheader('최대 / 최소 데이터 확인하기') :
        column = st.selectbox('컬럼을 선택하세요.', df.columns[4 :]) 
        st.text('최대 데이터')
        st.dataframe(df.loc[df[column] == df[column].max(), ])  # 변수로 처리하는 것. 일반화 
        st.text('최소 데이터')
        st.dataframe(df.loc[df[column] == df[column].min(), ]) 

    st.subheader('컬럼 별 히스토그램') 
    column = st.selectbox('히스토그램 확인할 컬럼을 선택하세요.', df.columns[3 :]) # 실행을 무조건 많이 시켜봐야 한다. 
    bins_number = st.number_input('빈의 갯수를 입력하세요.', 10, 30, 20, 1)

    # 그래프 시각화 구간. 시작은 fig = plt.figure(), 끝은 st.pyplot(fig)
    fig = plt.figure()
    df[column].hist(bins =bins_number)
    plt.title(column + 'Histogram')
    plt.xlabel(column)
    plt.ylabel('count')
    st.pyplot(fig)
    
    # change_text = """
    # <style>
    # div.st-cs.st-c5.st-bc.st-ct.st-cu {visibility: hidden;}
    # div.st-cs.st-c5.st-bc.st-ct.st-cu:before {content: "두 개 이상의 컬럼을 선택하세요."; visibility: visible;}
    # </style>
    # """
    # st.markdown(change_text, unsafe_allow_html=True)
    st.subheader('상관분석 하고 싶은 컬럼을 선택하세요.')
    column_list = st.multiselect('상관분석 하고 싶은 컬럼', df.columns)
    fig2 = plt.figure()
    # 1. 데이터를 가져와서 2. 표든 그래프든 그리는 거다. 
    if len(column_list) >= 2 : 
        sns.heatmap(data = df[column_list].corr(),
                    annot=True, vmin=-1, vmax=1, cmap='coolwarm',
                    fmt='.2f', linewidths= 0.5)
        st.pyplot(fig2)
        # 문제점은 테스트를 하면서 발견되는 것이다. # 인터페이스에서 의미없는 것은 보여주지 말도록 해야 한다.  
    elif len(column_list) == 1 : 
        st.text('2개 이상의 컬럼을 선택하세요.')

