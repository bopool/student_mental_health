import streamlit as st
import numpy as np
import joblib
# 구체적으로 어떤 방법을 썼는지, 어떤 메서드를 썼는지 보이는 것이 좋다. NaN있는지 확인하고 꼼꼼하게 계산대로 써라. 
# 개발할 때는 각 필요한 위치를 찾아 오르락내리락하면서 작업을 진행해야 한다. cpu말고는 위에서 아래로 진행되는 것이 없다. 

def run_app_ml() : 
    st.subheader('정신 건강과 학교생활 정보에 의한 학점 예측')
    # 성별, 나이, 연봉, 카드빚, 자산을 
    # 유저에게 입력받는다. 
    Gender = st.radio('성별', ['남자', '여자'])

    Age = st.number_input('나이 선택', 18, 100)
    
    Course = st.number_input('과목 선택', 18, 100)
    
    Year1 = st.checkbox('1학년')
    Year2 = st.checkbox('2학년')
    Year3 = st.checkbox('3학년')
    Year4 = st.checkbox('4학년')

    Marital_status = st.radio(
        "혼인 여부",
        ('미혼', '기혼'))
    Depression = st.radio(
        "최근 우울 증상을 느끼고 있나요?",
        ('예', '아니오'))
    Anxiety = st.radio(
        "최근 불안 증상을 느끼고 있나요?",
        ('예', '아니오'))
    Panic_attack = st.radio(
        "최근 공황발작 증상이 있었나요?",
        ('예', '아니오'))
    MH_Treatment = st.radio(
        "정신건강과 관련한 진료를 받은 경험이 있나요?",
        ('예', '아니오'))
    # 실제로 예측할 때도 '학습 시킬 때 사용한 항목'을 입력해야 한다.     

    new_data = np.array([Gender,Age,Course,Year1,Year2,Year3,Year4,Marital_status,Depression,Anxiety,Panic_attack,MH_Treatment])
    new_data = new_data.reshape(1, 12)
    
    regressor = joblib.load('model/regressor.pkl')
    y_pred = regressor.predict(new_data)
        
    # 버튼을 누르면 예측한 금액을 표시한다.
    if st.button('성적 예측'):
        print(y_pred)
        print(y_pred[0]) # = 
        print(round(y_pred[0]))
        price = round(y_pred[0])
        print(str(price) + '')
        print(f'{price}')
        print('{}'.format(price))
        st.text(f'{price}')
        


        

