from pathlib import Path

import pandas as pd
import streamlit as st

import next_bar_prediction_engine as engine


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
AVAILABLE_MODELS = list(
    getattr(engine, "AVAILABLE_MODELS", ("Logistic Regression", "Gradient Boosting"))
)
DEFAULT_MODEL = next(
    (
        model_name
        for model_name in ("Logistic Regression",)
        if model_name in AVAILABLE_MODELS
    ),
    AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "Logistic Regression",
)

st.set_page_config(
    page_title="Footprint Signal Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #edf4f0;
        --muted: #93a69d;
        --line: #2c3b35;
        --paper: #0d1512;
        --panel: #14201b;
        --green: #55d69a;
        --green-soft: #173d2d;
        --orange: #f3a35c;
        --orange-soft: #49301e;
        --red: #ff8177;
        --red-soft: #452321;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(13, 21, 18, .85); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: 2.6rem; line-height: 1.05; margin-bottom: .35rem; }
    h2 { font-size: 1.35rem; }
    [data-testid="stSidebar"] { background: #111c18; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stMetricValue"] div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    .eyebrow { color: var(--green); font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.04rem; margin-bottom: 1.5rem; }
    .status { display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: .78rem; font-weight: 700; }
    .status-ok { background: var(--green-soft); color: var(--green); }
    .status-danger { background: var(--red-soft); color: var(--red); }
    .result-card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 20px; margin-bottom: 14px; }
    .result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; margin-bottom: 12px; }
    .small-note { color: var(--muted); font-size: .84rem; }
    div[data-testid="stButton"] button[kind="primary"] { background: var(--green); border-color: var(--green); }
    div[data-testid="stButton"] button[kind="primary"] p { color: #07130d; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    return f"{value:,.2f} บาท"


def pct(value: float) -> str:
    return f"{value:.2%}"


def result_card(title: str, result: dict) -> None:
    status_class = "status-danger" if result["stopped_for_insufficient_balance"] else "status-ok"
    status_text = "พอร์ตแตก" if result["stopped_for_insufficient_balance"] else "ทดสอบครบ"
    profit_color = "#16835b" if result["net_profit"] >= 0 else "#bd4b43"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-title">{title} <span class="status {status_class}">{status_text}</span></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("ทุนสุดท้าย", money(result["final_balance"]))
    columns[1].metric("กำไร / ขาดทุน", money(result["net_profit"]))
    columns[2].metric("Win rate", pct(result["accuracy_on_signals"]))
    columns[3].metric("เข้าเงินจริง", f'{result["n_bets"]:,} ไม้')

    details = pd.DataFrame(
        [
            ["สัญญาณทั้งหมด", f'{result["n_signals"]:,} ครั้ง'],
            ["ทายถูก / ผิด", f'{result["n_correct_signals"]:,} / {result["n_wrong_signals"]:,}'],
            ["เดิมพันสูงสุด", money(result["max_stake"])],
            ["แพ้ติดกันสูงสุด", f'{result["max_loss_streak"]:,} ครั้ง'],
            ["ชนะติดกันสูงสุด", f'{result["max_win_streak"]:,} ครั้ง'],
            ["Maximum drawdown", money(result["max_drawdown"])],
        ],
        columns=["ตัวชี้วัด", "ค่า"],
    )
    st.dataframe(details, hide_index=True, width="stretch")
    st.markdown(
        f'<div class="small-note" style="color:{profit_color}">ผลนี้คำนวณจากสัญญาณทั้งหมด ไม่ใช่เฉพาะไม้ที่เงินยังพอเปิด</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="eyebrow">FOOTPRINT SIGNAL LAB</div>', unsafe_allow_html=True)
st.title("Next candle, made readable.")
st.markdown(
    '<div class="lede">แดชบอร์ดทดสอบสัญญาณแท่ง 5 นาทีถัดไปสำหรับ Binary Option และเปรียบเทียบระบบเดินเงินแบบ Fixed stake กับ Martingale</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ตั้งค่าการทดสอบ")
    local_files = sorted(
        [*APP_DIR.glob("*_ticks*.csv"), *DATA_DIR.glob("*_ticks*.csv")],
        key=lambda file: file.name.lower(),
    )
    file_options = {
        f"{file.name} ({file.parent.name}/)": file for file in local_files
    }
    source_mode = st.radio("แหล่งข้อมูล", ["ไฟล์ในโฟลเดอร์", "อัปโหลด CSV"], index=0)

    selected_path = None
    uploaded_file = None
    if source_mode == "ไฟล์ในโฟลเดอร์":
        if file_options:
            selected_label = st.selectbox("เลือกไฟล์ Backtest", list(file_options))
            selected_path = file_options[selected_label]
        else:
            st.warning("ยังไม่พบไฟล์ *_ticks.csv ในโฟลเดอร์โปรเจกต์")
    else:
        uploaded_file = st.file_uploader("เลือกไฟล์ tick CSV", type=["csv"])

    st.divider()
    starting_balance = st.number_input("ทุนเริ่มต้น (บาท)", min_value=100.0, value=2000.0, step=100.0)
    stake_per_trade = st.number_input("Fixed stake / เงินเริ่ม Martingale (บาท)", min_value=1.0, value=100.0, step=10.0)
    payout_rate = st.number_input("Payout ชนะ (%)", min_value=1.0, max_value=100.0, value=80.0, step=1.0) / 100
    martingale_multiplier = st.number_input("ตัวคูณ Martingale", min_value=1.0, value=2.0, step=0.1)
    threshold = st.slider("ความมั่นใจขั้นต่ำของ Signal", min_value=0.50, max_value=0.90, value=0.55, step=0.01)
    use_pattern_filter = st.checkbox("ใช้ Candle pattern filter", value=True)
    model_name = st.selectbox(
        "โมเดลหลัก",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
    )
    run_backtest = st.button("รัน Backtest", type="primary", width="stretch")

if not selected_path and uploaded_file is None:
    st.info("เลือกไฟล์ CSV จากแถบด้านซ้าย แล้วกด รัน Backtest เพื่อเริ่มวิเคราะห์")
    st.stop()

if run_backtest:
    source = uploaded_file if uploaded_file is not None else selected_path
    with st.spinner("กำลังสร้างแท่ง Footprint และทดสอบโมเดล..."):
        bars = engine.prepare_bars(source)
        comparisons = engine.run_selected_models(
                bars=bars,
                model_names=[model_name],
                starting_balance=starting_balance,
                stake_per_trade=stake_per_trade,
                payout_rate=payout_rate,
                martingale_multiplier=martingale_multiplier,
                threshold=threshold,
                use_pattern_filter=use_pattern_filter,
            )

    st.session_state["backtest"] = {
        "comparisons": comparisons,
        "file": uploaded_file.name if uploaded_file is not None else selected_path.name,
    }

if (
    "backtest" not in st.session_state
    or "comparisons" not in st.session_state["backtest"]
):
    st.info("ตั้งค่าทางซ้ายแล้วกด รัน Backtest")
    st.stop()

result = st.session_state["backtest"]
st.markdown(f"### {result['file']}")
model_results = result["comparisons"]
for current_model, comparison in model_results.items():
    st.markdown(f"## {current_model}")
    st.caption(
        f"แท่งทั้งหมด {comparison['bars']:,} · "
        f"ช่วงทดสอบ out-of-sample {comparison['split']:,} แท่ง · "
        "หมดอายุ 1 แท่ง (5 นาที)"
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric("Fixed stake", money(comparison["fixed"]["net_profit"]))
    summary_columns[1].metric("Martingale", money(comparison["martingale"]["net_profit"]))
    summary_columns[2].metric("Fixed win rate", pct(comparison["fixed"]["accuracy_on_signals"]))
    summary_columns[3].metric("Martingale win rate", pct(comparison["martingale"]["accuracy_on_signals"]))

    left, right = st.columns(2)
    with left:
        result_card(f"เดินเงินคงที่ · {stake_per_trade:,.0f} บาท", comparison["fixed"])
    with right:
        result_card(
            f"Martingale · x{martingale_multiplier:g} จาก {stake_per_trade:,.0f} บาท",
            comparison["martingale"],
        )

with st.expander("อ่านผลให้ถูกต้อง"):
    st.markdown(
        "Win rate คือความแม่นยำของสัญญาณทั้งหมด ส่วน `เข้าเงินจริง` คือจำนวนไม้ที่เปิดได้ตามทุนจริง "
        "Martingale อาจหยุดก่อนครบสัญญาณและแสดงสถานะ `พอร์ตแตก` จึงต้องดูจำนวนไม้ประกอบเสมอ"
    )
