import streamlit as st



def run_app_home() : 
    small_title = '<span style="color:#006D64"><b>A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA</b> dataset</span>'
    st.markdown(small_title, unsafe_allow_html=True)
    st.write('This Data set was collected by a survey conducted by Google forms from University student in order to examine their current academic situation and mental health.')

    img_url = 'img\dataset-cover.jpg'
    st.image(img_url)

    st.subheader('정신 건강이 학점이 미치는 영향에 관한 통계적 연구')
    st.write('이 앱은 Kaggle https://www.kaggle.com/ 에서 공유한 Student Mental health - A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA dataset을 분석하고 다양한 가설을 세워보는 공간입니다.')
    st.write('이 데이터 세트는 대학생의 현재 학업 상황과 정신 건강을 조사하기 위해 Google 양식으로 실시한 설문 조사에 의해 수집되었습니다. 데이터의 사이즈는 크지 않지만, 정신건강이 학점에 미치는 영향뿐만 아니라 이 데이터의 분석을 통해 볼 수 있는 다양한 상관관계를 파악해 보고자 합니다.')

