import streamlit as st
import numpy as np
import joblib
# 구체적으로 어떤 방법을 썼는지, 어떤 메서드를 썼는지 보이는 것이 좋다. NaN있는지 확인하고 꼼꼼하게 계산대로 써라. 
# 개발할 때는 각 필요한 위치를 찾아 오르락내리락하면서 작업을 진행해야 한다. cpu말고는 위에서 아래로 진행되는 것이 없다. 

def run_app_ml() : 
    st.subheader('자동차 금액 예측')

    # 성별, 나이, 연봉, 카드빚, 자산을 
    # 유저에게 입력받는다. 
    gender = st.radio('성별 선택', ['남자', '여자'])
    if gender == '남자':
        gender = 0
    else:
        gender = 1
    
    # 실제로 예측할 때도 '학습 시킬 때 사용한 항목'을 입력해야 한다.     
    age = st.number_input('나이 입력', 18, 100)
    salary = st.number_input('연봉 입력', 5000, 1000000)
    debt = st.number_input('카드빚', 0, 1000000)
    worth = st.number_input('자산 입력', 1000, 10000000)

    new_data = np.array([gender, age, salary, debt, worth])
    new_data = new_data.reshape(1, 5)
    
    regressor = joblib.load('model/regressor.pkl')
    y_pred = regressor.predict(new_data)
        
    # 버튼을 누르면 예측한 금액을 표시한다.
    if st.button('금액 예측') :
        print(y_pred)
        print(y_pred[0]) # = [] 프로그래밍은 이 두 개 모르면 못 짬. 컴퓨터에 찍어봐야 는다. 직접 문제 다시 풀어보고, 직접 짜보기. 구현하는 방법을 검색으로 찾아보세요. 
        print(round(y_pred[0]))
        price = round(y_pred[0])
        print(str(price) + '달러 짜리 차량 구매 가능합니다.')
        print(f'{price}달러짜리 차량 구매 가능합니다.')
        print('{}달러짜리 차량 구매 가능합니다'.format(price))
        st.text(f'{price}달러짜리 차량 구매 가능합니다.')
        


        

