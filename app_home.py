import streamlit as st

def run_app_home() : 
    st.subheader('환영합니다~')
    st.text('좋은 서비스로 제공하겠습니다!')
    st.text('자동배포 처리된 앱으로 제작 중입니다!!!')

    st.text('이 앱은 고객 데이터와 자동차 구매 데이터에 대한 내용입니다. ')
    st.text('데이터 분석도 가능하고, 고객 정보를 넣으면 구매 차량 가격도 예측해 줍니다. ')
    img_url = 'https://dimg.donga.com/wps/ECONOMY/IMAGE/2023/03/15/118348082.2.edit.jpg'
    st.image(img_url)
