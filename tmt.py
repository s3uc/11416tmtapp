import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="스마트팜 착과율 예측",
    page_icon="🍅",
    layout="wide"
)

# =========================
# CSS 스타일
# =========================
st.markdown("""
<style>

.main {
    background-color: #F4F8F4;
}

.stApp {
    background-color: #F4F8F4;
}

.header-box {
    background: linear-gradient(135deg, #2E7D32, #66BB6A);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}

.result-card {
    background: linear-gradient(135deg, #1B5E20, #43A047);
    color: white;
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    text-align: center;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    background-color: #2E7D32;
    color: white;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1B5E20;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 모델 불러오기
# =========================
MODEL_PATH = Path(__file__).parent / "tomato_model.pkl"

rf_model = joblib.load(MODEL_PATH)

# =========================
# 헤더
# =========================
st.markdown("""
<div class="header-box">
    <h1>🍅 스마트팜 착과율 예측 AI</h1>
    <h4>
    온도 · 습도 · 지온 데이터를 활용한
    실시간 착과율 분석 시스템
    </h4>
</div>
""", unsafe_allow_html=True)

# =========================
# 설명
# =========================
st.info(
    "환경 데이터를 입력하면 AI 모델이 예상 착과율을 예측합니다."
)

# =========================
# 입력 영역
# =========================
st.subheader("📋 환경 데이터 입력")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.number_input(
            "🌡️ 내부온도 (℃)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=0.1
        )

    with col2:
        humidity = st.number_input(
            "💧 내부습도 (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )

    with col3:
        soil_temp = st.number_input(
            "🌱 지온 (℃)",
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.1
        )

# =========================
# 현재 환경 상태
# =========================
st.subheader("📊 현재 입력 환경")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="내부온도",
        value=f"{temp:.1f}℃"
    )

with m2:
    st.metric(
        label="내부습도",
        value=f"{humidity:.1f}%"
    )

with m3:
    st.metric(
        label="지온",
        value=f"{soil_temp:.1f}℃"
    )

st.markdown("")

# =========================
# 예측 버튼
# =========================
if st.button("🔍 착과율 예측하기"):

    input_data = pd.DataFrame(
        [[temp, humidity, soil_temp]],
        columns=[
            "내부온도",
            "내부습도",
            "지온"
        ]
    )

    result = rf_model.predict(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # 결과 카드
    st.markdown(f"""
    <div class="result-card">

        <h2>📈 예측 착과율</h2>

        <h1 style="
        font-size:75px;
        margin-top:10px;
        margin-bottom:10px;
        ">
            {result:.1f}%
        </h1>

        <h4>
            AI 모델이 예측한 예상 착과율
        </h4>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 달성도 바
    st.progress(min(int(result), 100))

    st.caption(
        f"착과율 달성도 : {result:.1f}%"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 상태 판정
    if result >= 80:
        st.success(
            "🟢 매우 양호한 생육 환경입니다."
        )

    elif result >= 60:
        st.warning(
            "🟡 적정 수준의 생육 환경입니다."
        )

    else:
        st.error(
            "🔴 착과율 향상을 위해 환경 개선이 필요합니다."
        )

# =========================
# 하단
# =========================
st.markdown("---")

st.caption(
    "🍅 AI 기반 스마트팜 착과율 예측 시스템"
)
