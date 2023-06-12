import streamlit as st

from app_home import run_app_home
from app_eda import run_app_eda
from app_ml import run_app_ml


def main():
    st.write('<style>div > .css-1544g2n{padding-top:3.1rem; padding-left:1.4rem;}</style>', unsafe_allow_html=True)
    img_url2 = 'img/smh_side_top.jpg'
    st.sidebar.image(img_url2)
    title = '<div style="color:#006D64; font-size:44px; font-weight:900;">Student Mental Health</div>'
    st.markdown(title, unsafe_allow_html=True)
    menu = ['Introduce', 'Exploratory Data Analysis', 'ML']
    choice = st.sidebar.selectbox('', menu) # 세 개 중 하나를 선택하면 뭔가 작동해야 하니까 변수로 만들어 줌 
    
    feature = '<div style="color:#666; font-size:14px; font-weight:300; padding:6px 10px 5px 10px;"><b style="color:#006D64; font-size:19px; font-weight:bold; line-height:40px;">FEATURES</b></br><b style="font-size:15px; font-weight:bold; color:#222;">• Timestamp</b> (Timestamp)</br> - 타임스탬프</br> - Date/Time/Weekday 날짜/시간/요일</br><b style="font-size:15px; font-weight:bold; color:#222;">• Choose your gender</b> (Gender)</br> - 성정체성을 선택하세요</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Age</b> (Age) - 나이</br><b style="font-size:15px; font-weight:bold; color:#222;">•  What is your course?</b> (Course)</br> -  당신의 전공은 무엇입니까?</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Your current year of Study</b> (S-Year)</br> - 당신의 연구 햇수</br><b style="font-size:15px; font-weight:bold; color:#222;">•  What is your CGPA?</b> (CGPA)</br> - 당신의 성적평점은 얼마인가요?</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Marital status</b> (Marital status)</br> - 혼인 여부</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Do you have Depression?</b> (Depression)</br> - 우울 증상이 있나요?</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Do you have Anxiety?</b> (Anxiety)</br> - 불안 증상이 있나요?</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Do you have Panic attack?</b> (Panic attack)</br> -  공황발작 증상이 있나요?</br><b style="font-size:15px; font-weight:bold; color:#222;">•  Did you seek any specialist for a treatment?</b></br>(MH Treatment) - 치료를 위해 전문가를 찾았었나요?</div>'
    st.sidebar.markdown(feature, unsafe_allow_html=True)
    
    if choice == menu[0] : 
        run_app_home()
    elif choice == menu[1] :
        run_app_eda()
    else : 
        run_app_ml()

if __name__ == '__main__' : 
    main()