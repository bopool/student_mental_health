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
            st.write(('• {}년 {}월 {}일부터 설문조사를 시작하고 {}년 {}월 {}일까지 설문조사를 마무리 했습니다.').format(time_min.year, time_min.month, time_min.day, time_max.year, time_max.month, time_max.day))

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
            st.write(('• 총 {}명 중 {}명이 {}에 설문조사에 답변했으며, {}에도 {}명이 답했다. 또한 {}명의 사람(들)이 {}에 답했다.').format(df_count, df_weekday_count[0], df_weekday[0], df_weekday[1], df_weekday_count[1], df_weekday_count[2], df_weekday[2]))
            # fig = plt.figure()
            # sb.countplot(data= df, x='Weekday')
            # st.pyplot(fig)

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

            st.markdown(line2, unsafe_allow_html=True)

            
            if st.subheader('각 컬럼별 정보 확인하기') :
                column = st.selectbox('컬럼을 선택하세요.', df.columns[0 :]) 
                df_c = df[column]
                df_col = df_c.value_counts().sort_values(ascending=False).to_frame()
                for t in range(df.shape[0]):
                    df.loc[t, 'Time'] = df.loc[t, 'Time'].hour

                # 그래프 시각화 구간
                if df_c.nunique() <= 2: 
                    fig = plt.figure()
                    df_c_v_unique = df_c.sort_values().unique()
                    df_c_sv = df_c.value_counts().sort_values(ascending=False)
                    plt.axis('equal')
                    plt.pie(df_c_sv, labels =df_c_v_unique, shadow=False, autopct='%.1f%%')  
                    plt.gca().set_facecolor('#f5f5f5')
                    st.pyplot(fig)

                elif column == 'Time': 
                    fig1 = plt.figure()
                    df['Time'].hist(bins =24)
                    plt.title(column + 'Histogram')
                    plt.xlabel(column)
                    plt.ylabel('count')
                    plt.xticks(rotation=45)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().set_facecolor('#f5f5f5') 
                    st.pyplot(fig1)

                elif 2 < df_c.nunique() < 15: 
                    fig2 = plt.figure()
                    sb.countplot(data=df_c.to_frame(), x=column)
                    plt.xticks(rotation=45)
                    plt.gca().spines['top'].set_visible(False) 
                    plt.gca().spines['right'].set_visible(False) 
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().set_facecolor('#f5f5f5')
                    st.pyplot(fig2)
                                
                elif df_c.nunique() >= 15 : 
                    fig3 = plt.figure()
                    sb.countplot(data=df_c.to_frame(), x=column)
                    plt.gca().axes.xaxis.set_visible(False)
                    plt.gca().spines['top'].set_visible(False) 
                    plt.gca().spines['right'].set_visible(False) 
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().set_facecolor('#f5f5f5') 
                    st.pyplot(fig3)
                
                st.write('▼ '+'각 컬럼별 다수 그룹 확인')
                df_col_f= df_col.pivot_table(columns = df_col.index, values= df_col, sort=False)
                st.dataframe(df_col_f, width=860)

                st.markdown(line4, unsafe_allow_html=True)

                st.write(('▼ '+'최대 데이터 _ {} : {} 그룹 ( {} 명 / {} % )').format(column, df_c.max(), df.loc[df_c == df_c.max(), ].shape[0], df.loc[df_c == df_c.max(), ].shape[0] / df.shape[0] * 100))
                st.dataframe(df.loc[df[column] == df[column].max(), ]) # 변수로 처리하는 것. 일반화 
                st.write(('▼ '+'최소 데이터 _ {} : {} 그룹 ( {} 명 / {} % )').format(column, df_c.min(), df.loc[df_c == df_c.min(), ].shape[0], df.loc[df_c == df_c.min(), ].shape[0] / df.shape[0] * 100))
                st.dataframe(df.loc[df[column] == df[column].min(), ]) 

                st.markdown(line2, unsafe_allow_html=True)


            if st.subheader('상관분석') : 
                X_box = pd.DataFrame() # 빈 데이터프레임 생성. 
                # data가 가공된 분석에 필요한 모든 column을 포함하는 dataframe 으로 만들어 줄 것임

                for name in df.columns:
                    if df[name].dtype == object: # 문자열 트루면 
                        if df[name].nunique() <= 2 or df[name].nunique() > 6:
                            label_encoder = LabelEncoder()
                            X_box[name] = label_encoder.fit_transform(df[name])
                        else:
                            col_names = sorted(df[name].unique())
                            X_box[col_names] = pd.get_dummies(df[name], columns = col_names)

                    else: # 문자열 트루 아니면 
                        X_box[name] = df[name]
                
                X = X_box.loc[:, 'Gender':'MH Treatment']
                st.write('▼ '+'각 컬럼별 그룹 컬럼화')
                st.dataframe(X)

                st.markdown(line3, unsafe_allow_html=True)

                X['MH Point'] = X.loc[:, 'Depression':'MH Treatment'].sum(axis=1)
                st.write('▼ '+'요일, 날짜, 시간 정보 삭제 + Mental Health 관련 데이터 합산 포인트 컬럼 MH Point 추가')
                X_color1= X.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'MH Point':'MH Point'])
                st.dataframe(X_color1)

                st.markdown(line3, unsafe_allow_html=True)

                st.write('▼ '+'MH Point와 CGPA의 상관관계 확인')
                X['CGPA'] = df['CGPA']
                X_color2= X.style.applymap(draw_color_cell, color='#ffffb3', subset=pd.IndexSlice[:,'MH Point':'CGPA'])
                st.dataframe(X_color2)

                X_mhp_cgpa = X.loc[:,'MH Point':'CGPA']
                st.dataframe(X_mhp_cgpa, height=220, width=860)

               
                fig6 = plt.figure()
                sb.countplot(data=X_mhp_cgpa, x='CGPA')
                plt.xticks(rotation=45)
                plt.gca().spines['top'].set_visible(False) 
                plt.gca().spines['right'].set_visible(False) 
                plt.gca().spines['left'].set_visible(False)
                plt.gca().set_facecolor('#f5f5f5')
                st.pyplot(fig6)
                # # X['CGPA'].sort_values().unique()
                # X_mc_corr = X_mhp_cgpa.corr()
                # names = ['MH Point', 'CGPA']
                # X_mc = X_mc_corr.loc[:, names]
                # fig4 = plt.figure(figsize = (16, 6))
                # for i, name in enumerate(names[1:]):
                #     ax = plt.subplot(1, 2, i+1)
                #     sns.regplot(x=name, y=names[0], data = X_mc, ax=ax)
                # st.pyplot(fig4)




                #X_train = [:, '']
                
                # column_list = st.multiselect('상관분석하고 싶은 항목 선택', X.columns)
                # fig5 = plt.figure()
                # if len(column_list) == 1 : 
                #     st.write('2개 이상의 컬럼을 선택하세요.')
                # elif len(column_list) >= 2 : 
                #     sns.heatmap(data= X[column_list], cbar=True, annot=True, vmin=-1, vmax=1, cmap='coolwarm', fmt='.2f', linewidths= 0.5)
                # st.pyplot(fig5)

                # # sns.catplot(x=column,
                # # col = 'who', #캔버스 분리하기
                # # kind ='count',#빈도 막대그래프 그리기
                # # data = df_titanic) 
