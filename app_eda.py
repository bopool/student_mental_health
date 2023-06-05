import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import date
from pandas.core.dtypes.api import is_bool


def run_app_eda() : 
    line1 = '<div style="border-top:1px solid #006D64; width:100%; height:10px;"></div>'    
    st.markdown(line1, unsafe_allow_html=True)
     
    st.header('데이터 분석') 
    # st.subheader('데이터 확인 및 전처리')
    df = pd.read_csv('data/Student_Mental_health.csv', encoding = 'ISO-8859-1')

    if st.checkbox('데이터 확인') : 
        st.dataframe(df)
        # st.((df.info()))
        
        line2 = '<div style="border-top:1px solid #ddd; width:100%; height:40px; margin-top:20px;"></div>'    
        line3 = '<div style="border-top:0px dashed #ddd; width:100%; height:20px; margin-top:0px;"></div>'    
        line4 = '<div style="border-top:1px dashed #ddd; width:100%; height:30px; margin-top:10px;"></div>'    
        box1 = '<div style="border-left:1px solid #eaeaeb; border-bottom:1px solid #eaeaeb; width:100%; height:400px; display:block; margin-top:10px;"></div>'    


        st.markdown(line2, unsafe_allow_html=True)


        if st.checkbox('데이터 전처리') : 
            def draw_color_cell(x,color):
                color = f'background-color:{color}'
                return color
            # 데이터 다듬기 
            
            # 결측치 제거
            df_info = pd.DataFrame({'Column': df.columns,
                                    'Data Type': df.dtypes,
                                    'Non-Null Count': df.count()}, )
            df_info = df_info.reset_index(drop=True)
            df_isna_sum = df.isna().sum()
            df_isna_frame = df_isna_sum.to_frame()
            df_isna_frame.columns = ['NA']
            df_isna_frame['Column']= df.columns
            df_info_isna = df_info.merge(df_isna_frame, on='Column', how='left')

            st.write('▼ '+'데이터 정보 및 결측치 확인')
            st.text("<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 101 entries, 0 to 100\ndtypes: float64(1), object(10)\nmemory usage: 8.8+ KB\nData columns (total 11 columns):")
            df_info_isna1 = df_info_isna.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[2,'NA'])
            st.dataframe(df_info_isna1, height=422, width=860)

            st.markdown(line3, unsafe_allow_html=True)


            # 결측치 제거
            st.write('▼ '+' 결측치 삭제')
            df.dropna(inplace=True)
            df_info = pd.DataFrame({'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count()}, )
            df_info = df_info.reset_index(drop=True)
            df_isna_sum = df.isna().sum()
            df_isna_frame = df_isna_sum.to_frame()
            df_isna_frame.columns = ['NA']
            df_isna_frame['Column']= df.columns
            df_info_isna = df_info.merge(df_isna_frame, on='Column', how='left')
            st.dataframe(df_info_isna, height=422, width=860)
            
            st.markdown(line3, unsafe_allow_html=True)

            # 칼럼명을 보기 쉽게 다듬기 
            df22 = pd.DataFrame([['Timestamp', 'Timestamp'],
                        ['Choose your gender','Gender'],
                        ['Age','Age'],
                        ['What is your course?','Course'],
                        ['Your current year of Study','S-Year'],
                        ['What is your CGPA?','CGPA'],
                        ['Marital status','Marital status'],
                        ['Do you have Depression?','Depression'],
                        ['Do you have Anxiety?','Anxiety'],
                        ['Do you have Panic attack?','Panic attack'],
                        ['Did you seek any specialist for a treatment?', 'MH Treatment']],
                        columns = ['제목 변경 전','제목 변경 후'], )

            def draw_color_at_maxmum(x,color):
                color = f'background-color:{color}'
                not_0 = x != 0
                return [color if b else '' for b in not_0]
            
            df_color = df22.style.apply(draw_color_at_maxmum, color='#ffffb3',subset=['제목 변경 후'], axis=1)
            st.write('▼ '+'컬럼명 보기 쉽게 줄이기')
            st.dataframe(df_color, height=422, width=860)
            df = df.rename(columns={'Choose your gender':'Gender', 'What is your course?':'Course', 'Your current year of Study':'S-Year', 'What is your CGPA?':'CGPA', 'Do you have Depression?':'Depression', 'Do you have Anxiety?':'Anxiety', 'Do you have Panic attack?':'Panic attack', 'Did you seek any specialist for a treatment?':'MH Treatment'})

            st.markdown(line3, unsafe_allow_html=True)

            # 학년 정보의 표기를 통일해 준다. 
            df['S-Year'] = df['S-Year'].str.replace(' ', '')
            df['S-Year'] = df['S-Year'].str.title()
            st.write('▼ '+'Year/year 등 학년 정보의 표기 통일하기 ')
            df_color3= df.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'S-Year':'S-Year'])
            st.dataframe(df_color3)

            st.markdown(line3, unsafe_allow_html=True)

            # 날짜 별로 정렬해주기 
            # object type이었던 Timestamp를 datetime64[ns] type으로 변경, 시간으로 인식하도록 처리함. 
            # 기록 시간별로 다시 정렬해 줌. 
            df['Timestamp'] = pd.Series(pd.to_datetime(df['Timestamp']))
            df['Timestamp'] = pd.Series(df['Timestamp']).astype('datetime64')
            df.sort_values('Timestamp', inplace= True, ascending=False) # 시간 값 통일하는 방법 확인하자. 
            st.write('▼ '+'Timestamp column값들을 datetime64[ns] 타입으로 변경하고 시간순 재정렬함')

            time_min = pd.to_datetime(df['Timestamp'], ).min()
            time_max = pd.to_datetime(df['Timestamp'], ).max()
            st.write(('• {}년 {}월 {}일부터 설문조사를 시작하고 {}년 {}월 {}일까지 설문조사를 마무리 했다.').format(time_min.year, time_min.month, time_min.day, time_max.year, time_max.month, time_max.day))

            df_color1= df.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'Timestamp':'Timestamp'])
            st.dataframe(df_color1)

            st.markdown(line3, unsafe_allow_html=True)


            #요일도 확인하여 넣어준다. 
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday' ]
            df['Weekday'] = pd.to_datetime(df['Timestamp']).dt.weekday
            df['Weekday'] = df.apply(lambda x : days[x['Weekday']], axis = 1)
            st.write('▼ '+'Timestamp에 찍힌 날짜의 요일 정보 추가')
            df_color4= df.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'Weekday':'Weekday'])
            st.dataframe(df_color4)

            st.markdown(line3, unsafe_allow_html=True)

            # 요일 정보. 
            df_weekday = df['Weekday'].sort_values().unique()
            df_weekday_count = df['Weekday'].value_counts().sort_values(ascending=False)
            df_count = df.shape[0]
            
            ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
            st.write(('총 {}명 중 {}명이 {}에 설문조사에 답변했으며, {}에도 {}명이 답했다. 또한 {}명의 사람(들)이 {}에 답했다.').format(df_count, df_weekday_count[0], df_weekday[0], df_weekday[1], df_weekday_count[1], df_weekday_count[2], df_weekday[2]))
            fig = plt.figure()
            sb.countplot(data= df, x='Weekday')
            st.pyplot(fig)

            st.markdown(line3, unsafe_allow_html=True)

            # 시간 나눠서 넣음 
            df['Date'] = df['Timestamp'].dt.date
            df['Time'] = df['Timestamp'].dt.time
            df = df.drop(['Timestamp'], axis=1)

            st.write('▼ '+'Timestamp 날짜 정보와 시간 정보 분리')
            df_weekday_count = df['Weekday'].value_counts().sort_values(ascending=False)
            df_color5= df.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'Date':'Time'])
            st.dataframe(df_color5)

            st.markdown(line3, unsafe_allow_html=True)

            #인덱스 초기화
            df = df.reset_index(drop=True)
            st.write('▼ '+'인덱스 값 리셋')
            th_props = [
                ('font-size', '14px'),
                ('text-align', 'left'),
                ('font-weight', 'normal'),
                ('color', '#6d6d6d'),
                ('background-color', '#f8f9fb'),
                ('height', '28px'),
                ('width', 'auto'),
                ('word-break', 'keep-all')
                ]
            tr_props = [('height', '28px'),('width', 'auto'),('word-break', 'keep-all')]                                            
            td_props = [('font-size', '14px'), ('width', 'auto'),('word-break', 'keep-all')]

            styles = [
                dict(selector="th", props=th_props),
                dict(selector="tr", props=tr_props),
                dict(selector="td", props=td_props)
                ]
            df_index = df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            
            st.dataframe(df_index)

            st.markdown(line4, unsafe_allow_html=True)

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

            
            if st.subheader('각 컬럼별 정보 확인하기') :
                column = st.selectbox('컬럼을 선택하세요.', df.columns[0 :]) 
                df_c = df[column]
                st.write(('▼ '+'최대 데이터 _ {} : {} 그룹 ( {} 명 / {} % )').format(column, df_c.max(), df.loc[df_c == df_c.max(), ].shape[0], df.loc[df_c == df_c.max(), ].shape[0] / df.shape[0] * 100))
                st.dataframe(df.loc[df[column] == df[column].max(), ])  # 변수로 처리하는 것. 일반화 
                st.write(('▼ '+'최소 데이터 _ {} : {} 그룹 ( {} 명 / {} % )').format(column, df_c.min(), df.loc[df_c == df_c.min(), ].shape[0], df.loc[df_c == df_c.min(), ].shape[0] / df.shape[0] * 100))
                st.dataframe(df.loc[df[column] == df[column].min(), ]) 


                # 그래프 시각화 구간. 시작은 fig = plt.figure(), 끝은 st.pyplot(fig)
                #                 if df_c.type() == float and 7 > df_c.nunique() >= 2: 데이터 타입이 실수이고 그룹의 숫자가 2 이하이면 : 원 그래프 그리기.  
                fig = plt.figure()
                plt.pie(df_c, labels=df_c.value_counts())
                st.pyplot(fig)
                #   elif df_c.type() == float and df_c.nunique() >= 7: 
                #   else : 
                    #

                bins_number = st.number_input('빈의 갯수를 입력하세요.', 10, 30, 20, 1)
                fig = plt.figure()
                df[column].hist(bins =bins_number)
                plt.title(column + 'Histogram')
                plt.xlabel(column)
                plt.ylabel('count')
                st.pyplot(fig)
                
                #elif 데이터 타입이 실수이고 그룹 숫자가 2 이상 6이하이면 : 히스토그램 그리기 
                #elif 데이터 타입이 실수이고 그룹 숫자가 7 이상이면 : 히스토그램 그리기 
                #else : 선그래프? scatter ?

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


