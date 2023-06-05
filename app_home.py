import streamlit as st

def run_app_home() : 
    small_title = '<span style="color:#006D64"><b>A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA</b>  dataset</span>'
    st.markdown(small_title, unsafe_allow_html=True)
    st.write('This Data set was collected by a survey conducted by Google forms from University student in order to examine their current academic situation and mental health.')

    img_url = 'img/dataset-cover.jpg'
    st.image(img_url)

    st.subheader('정신 건강이 학점이 미치는 영향에 관한 통계적 연구')
    st.write('이 앱은 데이터분석 학습을 위한 프로젝트 공간입니다.')
    st.write('데이터는 Kaggle https://www.kaggle.com/ 에서 공유한 Student Mental health - A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA dataset입니다. 데이터의 양이 충분히 많지 않아 신뢰도면에서 유의미한 결과를 내기 어려울 것 같지만, 다양한 분석 활동이 가능한 항목으로 구성되어 있어서 이 데이터 셋을 선택하게 되었습니다. ')
    st.write('데이터의 설명에서 언급된 정신건강이 학점에 미치는 영향뿐만 아니라, 이 데이터의 분석을 통해 예측볼 수 있는 다양한 상관관계를 검토해 보고자 합니다. ')
    st.write('프로젝트 진행: 김하연 2023. 06. 23.')


