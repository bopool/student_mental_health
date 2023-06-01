import streamlit as st

from app_home import run_app_home
from app_eda import run_app_eda
from app_ml import run_app_ml


def main():
    title = '<div style="color:#006D64; font-size:44px; font-weight:900;">Student Mental Health</div>'
    st.markdown(title, unsafe_allow_html=True)

    menu = ['Introduce', 'EDA', 'ML']
    choice = st.sidebar.selectbox('', menu) # 세 개 중 하나를 선택하면 뭔가 작동해야 하니까 변수로 만들어 줌 

    if choice == menu[0] : 
        run_app_home()
    elif choice == menu[1] :
        run_app_eda()
        feature = '<div style="color:#666; font-size:15px; font-weight:300; padding:5px 10px 5px 10px;"><b style="color:#006D64; font-size:18px; font-weight:bold;">FEATURES</b></br>• Timestamp 입력한 시간</br>•  Choose your gender 성적정체성</br>•  Age 나이</br>•  What is your course? 전공</br>•  Your current year of Study 연차</br>•  What is your CGPA? 학업 성적</br>•  Marital status 혼인 여부</br>•  Do you have Depression? 우울 경험</br>•  Do you have Anxiety? 불안 경험</br>•  Do you have Panic attack? 공황발작 경험</br>•  Did you seek any specialist for a treatment? 진료경험</div>'
        st.sidebar.markdown(feature, unsafe_allow_html=True)
    else : 
        run_app_ml()

if __name__ == '__main__' : 
    main()