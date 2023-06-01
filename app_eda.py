import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import date


def run_app_eda() : 
    line1 = '<div style="border-top:1px solid #006D64; width:100%; height:10px;"></div>'    
    st.markdown(line1, unsafe_allow_html=True)
     
    st.header('데이터 분석') 
    # st.subheader('데이터 확인 및 전처리')
    df = pd.read_csv('data/Student_Mental_health.csv', encoding = 'ISO-8859-1')

    if st.checkbox('데이터 프레임 보기') : 
        st.dataframe(df)
        st.write("데이터 정보를 확인하니 'Age' column만 실수형 데이터타입(float64)이고 나머지는 모두 object였다. 데이터 간의 상관관계를 확인하기 위해 데이터 전처리 작업을 진행한다.")
        # st.((df.info()))
        
        line2 = '<div style="border-top:1px solid #ddd; width:100%; height:30px; margin-top:18px;"></div>'    
        st.markdown(line2, unsafe_allow_html=True)

        if st.checkbox('데이터 전처리') : 
            # 데이터 다듬기 
            # 날짜 별로 정렬해주기 
            # object type이었던 Timestamp를 datetime64[ns] type으로 변경, 시간으로 인식하도록 처리함. 
            # 기록 시간별로 다시 정렬해 줌. 
            pd.to_datetime(df['Timestamp'], )
            df.sort_values('Timestamp', inplace= True) # 시간 값 통일하는 방법 확인하자. 
            st.write('• Object 타입인 Timestamp column값들을 datetime64[ns] 타입으로 변경, 데이터를 Timestamp 시간순으로 재정렬')
            st.dataframe(df)

            # 칼럼명을 보기 쉽게 다듬기 
            df.rename(columns={'Choose your gender':'Gender', 
                            'What is your course?':'Course', 
                            'Your current year of Study':'S-Year', 
                            'What is your CGPA?':'CGPA', 
                            'Do you have Depression?':'Depression', 
                            'Do you have Anxiety?':'Anxiety', 
                            'Do you have Panic attack?':'Panic attack', 
                            'Did you seek any specialist for a treatment?':'M/H Treatment'}, inplace=True)
            st.write('• 칼럼명을 보기 쉽게 요약하기')
            st.dataframe(df)

            # 학년 정보의 표기를 통일해 준다. 
            df['S-Year'] = df['S-Year'].str.replace(' ', '')
            df['S-Year'] = df['S-Year'].str.title()
            st.write('• Year/year 등 학년 정보의 표기 통일하기 ')
            st.dataframe(df)

            #요일도 확인하여 넣어준다. 
            days = ['Mon', 'Tues', 'Weds', 'Thurs', 'Fri', 'Sat', 'Sun' ]
            df['Weekday'] = pd.to_datetime(df['Timestamp']).dt.weekday
            df['Weekday'] = df.apply(lambda x : days[x['Weekday']], axis = 1)
            st.write('• Timestamp에 찍힌 날짜의 요일 정보도 추가')
            st.dataframe(df)

            st.write('주로 금요일에 설문조사에 응했으며, 월요일에도 다.')
            fig = plt.figure()
            sb.countplot(data= df, x='Weekday')
            st.pyplot(fig)

            X = pd.DataFrame() # 빈 데이터프레임 생성. 
            # data가 가공된 분석에 필요한 모든 column을 포함하는 dataframe 으로 만들어 줄 것임

            for name in df.columns:
                if df[name].dtype == object: # 문자열 트루면 
                    if df[name].nunique() <= 2 or df[name].nunique() > 6:
                        label_encoder = LabelEncoder()
                        X[name] = label_encoder.fit_transform(df[name])
                    else:
                        col_names = sorted(df[name].unique())
                        X[col_names] = pd.get_dummies(df[name], columns = col_names)

                else: # 문자열 트루 아니면 
                    X[name] = df[name]

            
            if st.subheader('최대/최소 데이터 확인하기') :
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

