# Student Mental health Dataset
<b>A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA dataset</b><br>
정신건강이 학생의 CGPA 학업성적에 미치는 영향에 관한 통계적 연구를 위한 데이터셋<br><br>
<img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/googlecolab-F9AB00?style=flat&logo=googlecolab&logoColor=white"/> <img src="https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/amazonaws-232F3E?style=flat&logo=amazonaws&logoColor=white"/> <img src="https://img.shields.io/badge/linux-FCC624?style=flat&logo=linux&logoColor=white"/><br>
<br>
<img src="https://github.com/bopool/student_mental_health/blob/main/img/dataset-cover.jpg" alt="낮은 직사각형 초록색 바탕이미지 안에 인물의 옆모습으로 보이는 흰색 실루엣이 있다. 흰색 인물의 머리 부분에는 학생들의 생활을 상징하는 학사모와 연필 실험도구 차트 돋보기 등 다양한 아이콘 이미지들이 둥글게 배치되어 있다." style="border-radius:5px"><br><br>

## 프로젝트 소개
 
프로젝트로 선택한 데이터는 Kaggle에서 공유한 [Student Mental health - A STATISTICAL RESEARCH ON THE EFFECTS OF MENTAL HEALTH ON STUDENTS’ CGPA dataset](https://www.kaggle.com/datasets/shariful07/student-mental-health)입니다. 다양한 분석 활동이 가능한 항목으로 구성되어 있어서 이 데이터 셋을 선택하였습니다. 
데이터의 설명에서 언급된 정신건강이 학점에 미치는 영향뿐만 아니라, 이 데이터의 분석을 통해 예측볼 수 있는 다양한 상관관계를 검토해 보았습니다. 

### 1. 데이터 속성 <br>
<b>FEATURES</b><br>
- Timestamp (Date, Time, Weekday)<br>
- Choose your gender (Gender)<br>
- Age (Age)<br>
- What is your course? (Course)<br>
- Your current year of Study (S-Year)<br>
- What is your CGPA? (CGPA)<br>
  - CGPA: Cumulative Grade Point Average. 학생이 학기 중에 수강한 모든 과목의 평균 학점을 나타내는 지표
- Marital status (Marital status)<br>
- Do you have Depression? (Depression)<br>
- Do you have Anxiety? (Anxiety)<br>
- Do you have Panic attack? (Panic attack)<br>
- Did you seek any specialist for a treatment? (MH Treatment)<br>
- (MP Point)<br>
<br>

### 2. 탐색적 데이터 분석 
<b>Exploratory Data Analysis</b><br>
- 데이터의 정보 및 결측치 확인, 삭제
- 컬럼명을 보기 쉽게 줄여주기 
- 데이터의 대소문자, 띄어쓰기 정보 표기 확인 후 통일
- Timestamp로 기록된 값들을 datetime64 타입으로 변경하고 시간 순으로 재정렬
- Timestamp에 찍힌 날짜의 요일 정보 컬럼 추가 
- Timestamp의 날짜 정보와 시간 정보 분리하여 칼럼 추가
- 인덱스 값 리셋
- 각 컬럼별 정보 확인하기 
  - 하나의 컬럼을 선택하면 데이터의 성격에 맞는 차트 이미지와 함께 
  - 각 컬럼별 다수 그룹 확인할 수 있는 데이터프레임, 
  - 최대/최소 데이터의 정보를 확인할 수 있는 데이터프레임 함께 셋팅
- 상관분석 
  - One-hot encoding으로 각 컬럼별 그룹을 칼럼으로 변경
  - 대부분의 설문조사가 개인적 의미 없이 수업 시간에 진행된 것으로 보임. 요일, 날짜, 시간 정보 삭제
  - 우울, 불안, 공황 증상과 전문가의 치료를 받아보았는지 확인하는 칼럼들의 값을 합산하여 MH Point 칼럼으로 추가
  - 전체 데이터의 상관계수를 구하고, 히트맵 차트와 데이터프레임으로 시각화
  - 우울과 관련된 MH Point와 학업평균 성적인 CGPA와의 상관관계 확인
  - 정신건강과 직접 관련되지 않은 일반 정보들과 상관관계가 있는지 확인
  .... 
   
### 3. ....
💡
