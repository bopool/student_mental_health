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

    st.subheader('정신 건강과 학교생활 정보에 의한 학점 예측')
    st.markdown(line3, unsafe_allow_html=True)

    gender = st.radio('● 성별을 선택하세요.', ['남자', '여자'])
    st.markdown(line3, unsafe_allow_html=True)

    age = st.number_input('● 나이를 입력하세요', 18, 100)
    st.markdown(line3, unsafe_allow_html=True)

    course = st.number_input('● 전공을 선택하세요.', 1, 100)
    st.markdown(line3, unsafe_allow_html=True)

    gender = st.radio('● 해당되는 학년을 선택하세요.', ['1학년', '2학년', '3학년', '4학년'])
    st.markdown(line3, unsafe_allow_html=True)

    marital_status = st.radio('● 혼인 여부를 선택하세요.', ['미혼', '기혼'])
    st.markdown(line3, unsafe_allow_html=True)

    depression = st.radio("● 최근 우울 증상을 느끼고 있나요?", ['예', '아니오'])
    st.markdown(line3, unsafe_allow_html=True)

    anxiety = st.radio("● 최근 불안 증상을 느끼고 있나요?", ['예', '아니오'])
    st.markdown(line3, unsafe_allow_html=True)

    panic_attack = st.radio("● 최근 공황발작 증상이 있었나요?", ['예', '아니오'])
    st.markdown(line3, unsafe_allow_html=True)

    mh_treatment = st.radio("● 정신건강과 관련한 진료를 받은 경험이 있나요?", ['예', '아니오'])
    st.markdown(line3, unsafe_allow_html=True)
    # 실제로 예측할 때도 '학습 시킬 때 사용한 항목'을 입력해야 한다.     

    new_data = np.array([gender,age,course,marital_status,depression,anxiety,panic_attack,mh_treatment])
    new_data = new_data.reshape(1, 8)
    
    regressor = joblib.load('model/regressor.pkl')
    y_pred = regressor.predict(new_data)
        
    # 버튼을 누르면 예측한 금액을 표시한다.
    if st.button('성적 예측'):
        print(y_pred)
        # print(y_pred[0])  
        # print(round(y_pred[0]))
        # price = round(y_pred[0])
        # print(str(price) + '')
        # print(f'{price}')
        # print('{}'.format(price))
        # st.text(f'{price}')
        


        

