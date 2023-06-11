from sklearn.linear_model import LinearRegression
import streamlit as st
import numpy as np
import joblib
# 구체적으로 어떤 방법을 썼는지, 어떤 메서드를 썼는지 보이는 것이 좋다. NaN있는지 확인하고 꼼꼼하게 계산대로 써라. 
# 개발할 때는 각 필요한 위치를 찾아 오르락내리락하면서 작업을 진행해야 한다. cpu말고는 위에서 아래로 진행되는 것이 없다. 

def run_app_ml() : 
    line1 = '<div style="border-top:1px solid #006D64; width:100%; height:10px;"></div>'    
    st.markdown(line1, unsafe_allow_html=True)
    line2 = '<div style="border-top:1px solid #ddd; width:100%; height:40px; margin-top:20px;"></div>'    
    line3 = '<div style="border-top:0px dashed #ddd; width:100%; height:20px; margin-top:0px;"></div>'    
    line4 = '<div style="border-top:1px dashed #ddd; width:100%; height:30px; margin-top:10px;"></div>'    

    st.subheader('정신 건강과 학교생활 정보를 통한 학점 예측')
    st.markdown(line3, unsafe_allow_html=True)

    gender = st.radio('● 성별을 선택하세요.', ['남자', '여자'])
    if gender == '남자':
        gender = 0
    else:
        gender = 1
    st.markdown(line3, unsafe_allow_html=True)
    

    age = st.number_input('● 나이를 입력하세요', 18, 100)
    st.markdown(line3, unsafe_allow_html=True)

    course = st.selectbox('● 전공을 선택하세요.', ['Engineering', 'Islamic education', 'BIT', 'Laws', 'Mathemathics',
       'Pendidikan islam', 'BCS', 'Human Resources', 'Irkhs',
       'Psychology', 'KENMS', 'Accounting ', 'ENM', 'Marine science',
       'KOE', 'Banking Studies', 'Business Administration', 'Law',
       'KIRKHS', 'Usuluddin ', 'TAASL', 'Engine', 'ALA',
       'Biomedical science', 'koe', 'Kirkhs', 'BENL', 'Benl', 'IT', 'CTS',
       'engin', 'Econs', 'MHSC', 'Malcom', 'Kop', 'Human Sciences ',
       'Biotechnology', 'Communication ', 'Diploma Nursing',
       'Pendidikan Islam ', 'Radiography', 'psychology', 'Fiqh fatwa ',
       'DIPLOMA TESL', 'Koe', 'Fiqh', 'Islamic Education', 'Nursing ',
       'Pendidikan Islam'])
    course_ls = ['Engineering', 'Islamic education', 'BIT', 'Laws', 'Mathemathics',
       'Pendidikan islam', 'BCS', 'Human Resources', 'Irkhs',
       'Psychology', 'KENMS', 'Accounting ', 'ENM', 'Marine science',
       'KOE', 'Banking Studies', 'Business Administration', 'Law',
       'KIRKHS', 'Usuluddin ', 'TAASL', 'Engine', 'ALA',
       'Biomedical science', 'koe', 'Kirkhs', 'BENL', 'Benl', 'IT', 'CTS',
       'engin', 'Econs', 'MHSC', 'Malcom', 'Kop', 'Human Sciences ',
       'Biotechnology', 'Communication ', 'Diploma Nursing',
       'Pendidikan Islam ', 'Radiography', 'psychology', 'Fiqh fatwa ',
       'DIPLOMA TESL', 'Koe', 'Fiqh', 'Islamic Education', 'Nursing ',
       'Pendidikan Islam']
    for c in range(len(course_ls)):
        if course == course_ls[c]:
            course = c
    st.markdown(line3, unsafe_allow_html=True)

    year = st.selectbox('● 해당되는 학년을 선택하세요.', ['1학년', '2학년', '3학년', '4학년'])
    if year == '1학년':
        year = 0
    elif year == '2학년':
        year = 1
    elif year == '3학년':
        year = 2
    elif year == '4학년':
        year = 13
    
    st.markdown(line3, unsafe_allow_html=True)

    marital_status = st.radio('● 혼인 여부를 선택하세요.', ['미혼', '기혼'])
    if marital_status == '미혼':
        marital_status = 0
    else:
        marital_status = 1
    st.markdown(line3, unsafe_allow_html=True)

    depression = st.radio("● 최근 우울 증상을 느끼고 있나요?", ['예', '아니오'])
    if depression == '예':
        depression = 0
    else:
        depression = 1
    st.markdown(line3, unsafe_allow_html=True)

    anxiety = st.radio("● 최근 불안 증상을 느끼고 있나요?", ['예', '아니오'])
    if anxiety == '예':
        anxiety = 0
    else:
        anxiety = 1
    st.markdown(line3, unsafe_allow_html=True)

    panic_attack = st.radio("● 최근 공황발작 증상이 있었나요?", ['예', '아니오'])
    if panic_attack == '예':
        panic_attack = 0
    else:
        panic_attack = 1
    st.markdown(line3, unsafe_allow_html=True)

    mh_treatment = st.radio("● 정신건강과 관련한 진료를 받은 경험이 있나요?", ['예', '아니오'])
    if mh_treatment == '예':
        mh_treatment = 0
    else:
        mh_treatment = 1
    st.markdown(line3, unsafe_allow_html=True)
    # 실제로 예측할 때도 '학습 시킬 때 사용한 항목'을 입력해야 한다.     

    new_data = np.array([gender, age, course, marital_status, depression, anxiety, panic_attack, mh_treatment, year], )
    new_data = new_data.reshape(1, 9)
    
    regressor = joblib.load('model/regressor2.pkl')
    y_pred = regressor.predict(new_data)
        
    # 버튼을 누르면 예측한 정보를 표시한다.
    if st.button('CGPA 범위 예측'):
        pred_cgpa = round(y_pred[0])-1
        cgpa_ls = ["0 - 1.99", "2.00 - 2.49", "2.50-2.99", "3.00-3.49", "3.50-4.00"]
        for n in range(5):
            if pred_cgpa == n:
                pred_cgpa = cgpa_ls[n]
        # print(str(pred_cgp) + '')
        # print(f'예상 CGPA 점수는 {pred_cgpa}점 입니다.')
        # print('{}'.format(pred_cgp))
        st.write(f'예상 CGPA 점수 범위는 {pred_cgpa}점 입니다.')
        



