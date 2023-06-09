import streamlit as st

def run_app_home() : 
    small_title = '<span style="color:#006D64"><b>A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA</b>  dataset</span>'
    st.markdown(small_title, unsafe_allow_html=True)
    st.write('This Data set was collected by a survey conducted by Google forms from University student in order to examine their current academic situation and mental health.')

    img_url = 'img/dataset-cover.jpg'
    st.image(img_url)

    st.subheader('정신 건강이 학점에 미치는 영향에 관한 통계적 연구')
    st.write('본 프로젝트의 데이터는 Kaggle https://www.kaggle.com/datasets/shariful07/student-mental-health 에서 공유한 Student Mental health - A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA dataset입니다. 다양한 분석 활동이 가능한 항목들로 구성되어 있어서 이 데이터 셋을 선택하였습니다. ')
    st.write('데이터의 설명에서 언급된 정신건강이 학점에 미치는 영향뿐만 아니라, 이 데이터의 분석을 통해 예측볼 수 있는 다양한 상관관계를 검토해 보고자 합니다. ')


