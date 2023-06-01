import streamlit as st

from app_home import run_app_home
from app_eda import run_app_eda
from app_ml import run_app_ml


def main():
    st.title('Student Mental Health')
    menu = ['Home', 'EDA(탐색적데이터분석)', 'ML']
    choice = st.sidebar.selectbox('메뉴', menu) # 세 개 중 하나를 선택하면 뭔가 작동해야 하니까 변수로 만들어 줌 
    if choice == menu[0] : 
        run_app_home()
    elif choice == menu[1] :
        run_app_eda()
    else : 
        run_app_ml()

if __name__ == '__main__' : 
    main()