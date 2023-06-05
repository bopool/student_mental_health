import streamlit as st

from app_home import run_app_home
from app_eda import run_app_eda
from app_ml import run_app_ml


def main():
    #sidebar
    img_url2 = 'img/smh_side_top.jpg'
    st.sidebar.image(img_url2)
    st.sidebar.markdown(feature, unsafe_allow_html=True)
    choice = st.sidebar.selectbox('', menu) # 세 개 중 하나를 선택하면 뭔가 작동해야 하니까 변수로 만들어 줌 
    if choice == menu[0] : 
        run_app_home()
    elif choice == menu[1] :
        run_app_eda()
    else : 
        run_app_ml()
    menu = ['Introduce', 'Exploratory Data Analysis', 'ML']
    feature = '<div style="color:#666; font-size:15px; font-weight:300; padding:6px 10px 5px 10px;"><b style="color:#006D64; font-size:18px; font-weight:bold;">FEATURES</b></br>• Timestamp (Date, Time, Weekday)</br>•  Choose your gender (Gender)</br>•  Age</br>•  What is your course? (Course)</br>•  Your current year of Study (S-Year)</br>•  What is your CGPA? (CGPA)</br>•  Marital status</br>•  Do you have Depression? (Depression)</br>•  Do you have Anxiety? (Anxiety)</br>•  Do you have Panic attack? (Panic attack)</br>•  Did you seek any specialist for a treatment? (MH Treatment)</div>'
    
    title = '<div style="color:#006D64; font-size:44px; font-weight:900;">Student Mental Health</div>'
    st.markdown(title, unsafe_allow_html=True)


if __name__ == '__main__' : 
    main()