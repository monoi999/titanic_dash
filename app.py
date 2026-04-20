import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. 페이지 설정
st.set_page_config(page_title="타이타닉 종합 대시보드", layout="wide")

# --- 2. 디자인 개선 (탭 글자 크기 및 스타일) ---
st.markdown("""
    <style>
        /* 탭 텍스트 크기 확대 및 강조 */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 16px;
            font-weight: bold;
            color: #31333F;
        }
        
        /* 탭 메뉴 간격 및 테두리 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            border-bottom: 2px solid #e0e0e0;
        }

        /* 탭 버튼 배경 및 라운딩 */
        .stTabs [data-baseweb="tab"] {
            height: 55px;
            background-color: #f9f9f9;
            border-radius: 8px 8px 0px 0px;
            padding: 0px 30px;
        }
        
        /* 선택된 활성 탭 스타일 */
        .stTabs [aria-selected="true"] {
            background-color: #f0f7ff !important;
            border-bottom: 4px solid #007bff !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    # 전처리: 결측치 및 변수 생성
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 항구 이름 가독성 개선
    embarked_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    df['Embarked_Full'] = df['Embarked'].map(embarked_map)
    
    # 연령대 그룹
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이', '청소년', '청년', '중년', '노년']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # 모델용 인코딩
    df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

df = load_data()

# 4. 머신러닝 모델 학습
@st.cache_resource
def train_model(data):
    X = data[['Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = data['Survived']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# --- 5. 사이드바 필터 설정 (객실 등급, 성별, 탑승 항구) ---
st.sidebar.header("🔍 데이터 분석 필터")
st.sidebar.write("대시보드 차트에 적용될 조건을 선택하세요.")

# 객실 등급 필터
selected_pclass = st.sidebar.multiselect(
    "1. 객실 등급 선택", 
    options=sorted(df['Pclass'].unique()), 
    default=df['Pclass'].unique()
)

# 성별 필터
selected_sex = st.sidebar.multiselect(
    "2. 성별 선택", 
    options=df['Sex'].unique(), 
    default=df['Sex'].unique()
)

# 탑승 항구 필터 (추가됨)
selected_embarked = st.sidebar.multiselect(
    "3. 탑승 항구 선택", 
    options=df['Embarked_Full'].unique(), 
    default=df['Embarked_Full'].unique()
)

# 데이터 필터링 적용
filtered_df = df[
    (df['Pclass'].isin(selected_pclass)) & 
    (df['Sex'].isin(selected_sex)) & 
    (df['Embarked_Full'].isin(selected_embarked))
]

# --- 6. 메인 레이아웃 및 탭 구성 ---
st.title("🚢 타이타닉 생존 인사이트 대시보드")

tab1, tab2 = st.tabs(["📊 데이터 분석 대시보드", "🔮 생존 예측 시뮬레이터"])

# --- Tab 1: 분석 대시보드 ---
with tab1:
    st.subheader("📍 필터 결과 요약")
    
    # 상단 지표(Metrics)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("승객 수", f"{len(filtered_df)}명")
    m2.metric("평균 생존율", f"{filtered_df['Survived'].mean():.1%}")
    m3.metric("평균 연령", f"{filtered_df['Age'].mean():.1f}세")
    m4.metric("평균 운임", f"${filtered_df['Fare'].mean():.1f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        # 객실 등급별 생존 현황
        fig1 = px.histogram(filtered_df, x="Pclass", color="Survived", barmode="group",
                            title="객실 등급별 생존 인원",
                            color_discrete_map={0: "#EF553B", 1: "#00CC96"},
                            labels={'Pclass': '객실 등급', 'count': '인원 수'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # 항구별 생존율 분석 (필터 추가 기념 차트)
        port_survival = filtered_df.groupby('Embarked_Full')['Survived'].mean().reset_index()
        fig2 = px.bar(port_survival, x="Embarked_Full", y="Survived", 
                      title="탑승 항구별 평균 생존율",
                      labels={'Embarked_Full': '탑승 항구', 'Survived': '생존율'},
                      color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: 생존 예측 시뮬레이터 ---
with tab2:
    st.subheader("🕵️ AI 생존 확률 시뮬레이션")
    st.write("승객의 세부 정보를 입력하면 AI가 생존 가능성을 실시간으로 계산합니다.")
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        # 시뮬레이터 입력창
        i_sex = st.radio("성별", ["여성", "남성"], horizontal=True)
        i_class = st.select_slider("객실 등급", options=[1, 2, 3], value=1)
        i_age = st.slider("나이", 0, 80, 25)
        i_fare = st.number_input("지불 요금 ($)", 0, 500, 30)
        i_fam = st.slider("동반 가족 수", 0, 10, 0)
        
        # 모델 입력 데이터 변환
        sex_val = 1 if i_sex == "여성" else 0
        input_data = pd.DataFrame([[i_class, sex_val, i_age, i_fam, 0, i_fare]], 
                                  columns=['Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare'])
        
        # 예측 수행
        prob = model.predict_proba(input_data)[0][1] * 100

    with c2:
        # 생존 확률 시각화 (게이지)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            title = {'text': f"예상 생존 확률", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': "#f8d7da"},
                    {'range': [50, 100], 'color': "#d4edda"}]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if prob >= 50:
            st.success(f"결과: 이 조건의 승객은 **생존 가능성**이 높습니다! ({prob:.1f}%)")
        else:
            st.error(f"결과: 이 조건의 승객은 **사망 가능성**이 높습니다. ({prob:.1f}%)")
