import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

# 페이지 설정
st.set_page_config(page_title="타이타닉 생존 분석 대시보드", layout="wide")

# 1. 데이터 로드 및 전처리 (캐싱을 통해 속도 향상)
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    # 호칭 추출 및 나이 결측치 채우기 (가장 효율적인 방법 적용)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # 항구 결측치 채우기
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # 연령대 생성
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['어린이', '청소년', '청년', '중년', '노년']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

df = load_data()

# --- 사이드바: 필터 설정 ---
st.sidebar.header("🔍 데이터 필터")
selected_pclass = st.sidebar.multiselect("객실 등급 선택", options=[1, 2, 3], default=[1, 2, 3])
selected_sex = st.sidebar.radio("성별 선택", options=['전체', 'male', 'female'])

# 필터링 적용
filtered_df = df[df['Pclass'].isin(selected_pclass)]
if selected_sex != '전체':
    filtered_df = filtered_df[filtered_df['Sex'] == selected_sex]

# --- 메인 화면: 대시보드 구성 ---
st.title("🚢 Titanic Survival Dashboard")
st.markdown("승객 데이터를 바탕으로 한 생존율 및 인구통계 분석 결과입니다.")

# 2. 상단 KPI 카드 (핵심 지표)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 승객 수", f"{len(filtered_df)}명")
with col2:
    survival_rate = filtered_df['Survived'].mean() * 100
    st.metric("평균 생존율", f"{survival_rate:.1f}%")
with col3:
    st.metric("평균 요금", f"${filtered_df['Fare'].mean():.2f}")
with col4:
    st.metric("평균 나이", f"{filtered_df['Age'].mean():.1f}세")

st.divider()

# 3. 차트 섹션
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("📊 객실 등급별 생존자 수")
    fig1 = px.histogram(filtered_df, x="Pclass", color="Survived", 
                        barmode="group", color_discrete_map={0: "red", 1: "green"},
                        labels={'Survived': '생존 여부 (0:사망, 1:생존)'})
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("👧 성별 및 연령대별 생존율")
    # 생존율 계산을 위한 집계
    survival_by_age = filtered_df.groupby(['AgeGroup', 'Sex'], as_index=False)['Survived'].mean()
    fig2 = px.bar(survival_by_age, x="AgeGroup", y="Survived", color="Sex",
                  barmode="group", title="연령대별 생존 확률")
    st.plotly_chart(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("💰 요금(Fare)과 생존의 관계")
    fig3 = px.box(filtered_df, x="Survived", y="Fare", color="Survived",
                  points="all", title="생존 여부에 따른 요금 분포")
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    st.subheader("📍 탑승 항구별 승객 분포")
    embarked_counts = filtered_df['Embarked'].value_counts().reset_index()
    fig4 = px.pie(embarked_counts, values='count', names='Embarked', hole=0.4)
    st.plotly_chart(fig4, use_container_width=True)

# 4. 데이터 테이블 확인
if st.checkbox("전체 데이터 보기"):
    st.dataframe(filtered_df)