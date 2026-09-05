"""Engine สำหรับทำนายทิศทางแท่ง 5 นาทีถัดไป.

จุดนี้คือเจ้าของ logic การเตรียมข้อมูล, train โมเดล และสร้างผลทำนาย
จึงเป็นไฟล์หลักที่ควรแก้เมื่อจะพัฒนา win rate หรือเพิ่ม signal filter.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

import footprint_ml_pipeline as pipeline


BAR_FREQ = "5min"
AVAILABLE_MODELS = ("Logistic Regression",)


def make_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")


def prepare_bars(csv_source) -> pd.DataFrame:
    """อ่าน tick CSV แล้วสร้างข้อมูลแท่งพร้อม feature สำหรับ next-bar prediction."""
    ticks = pipeline.load_ticks(csv_source)
    bars = pipeline.build_footprint_features(
        ticks,
        bar_freq=BAR_FREQ,
        bucket_size=None,
        imbalance_ratio=None,
        min_stacked_rows=3,
    )
    bars = pipeline.add_context_features(bars, n_lags=3, roll_window=5)
    bars = pipeline.add_labels(bars)
    return bars.dropna(subset=pipeline.FEATURE_COLUMNS).reset_index(drop=True)


def train_model(bars: pd.DataFrame, model_name: str, train_ratio: float = 0.80):
    """ฝึกโมเดลด้วยข้อมูลช่วงต้น และคืนโมเดลกับจุดเริ่ม out-of-sample."""
    split_point = int(len(bars) * train_ratio)
    model = make_model(model_name)
    model.fit(
        bars[pipeline.FEATURE_COLUMNS].values[:split_point],
        bars["label_up"].values[:split_point],
    )
    return model, split_point


def run_comparison(
    bars: pd.DataFrame,
    model_name: str,
    starting_balance: float,
    stake_per_trade: float,
    payout_rate: float,
    martingale_multiplier: float,
    threshold: float,
    use_pattern_filter: bool,
):
    """สร้างผลเปรียบเทียบ fixed stake และ Martingale จากสัญญาณชุดเดียวกัน."""
    model, split_point = train_model(bars, model_name)
    common = {
        "bars": bars,
        "model": model,
        "train_end_idx": split_point,
        "prob_threshold": threshold,
        "starting_balance": starting_balance,
        "stake_per_trade": stake_per_trade,
        "payout_rate": payout_rate,
        "use_pattern_filter": use_pattern_filter,
        "martingale_multiplier": martingale_multiplier,
    }
    return {
        "bars": len(bars),
        "split": split_point,
        "model": model_name,
        "fixed": pipeline.binary_option_backtest(
            **common, money_management="fixed_stake"
        ),
        "martingale": pipeline.binary_option_backtest(
            **common, money_management="martingale"
        ),
    }


def run_all_models(
    bars: pd.DataFrame,
    starting_balance: float,
    stake_per_trade: float,
    payout_rate: float,
    martingale_multiplier: float,
    threshold: float,
    use_pattern_filter: bool,
):
    """รันโมเดลทั้งหมดด้วยชุดแท่งและค่าทดสอบเดียวกันเพื่อเปรียบเทียบตรง ๆ."""
    model_names = list(AVAILABLE_MODELS)
    return run_selected_models(
        bars=bars,
        model_names=model_names,
        starting_balance=starting_balance,
        stake_per_trade=stake_per_trade,
        payout_rate=payout_rate,
        martingale_multiplier=martingale_multiplier,
        threshold=threshold,
        use_pattern_filter=use_pattern_filter,
    )


def run_selected_models(
    bars: pd.DataFrame,
    model_names: list[str],
    starting_balance: float,
    stake_per_trade: float,
    payout_rate: float,
    martingale_multiplier: float,
    threshold: float,
    use_pattern_filter: bool,
):
    """รันเฉพาะโมเดลที่ผู้ใช้เลือกเพื่อเปรียบเทียบกัน."""
    return {
        model_name: run_comparison(
            bars=bars,
            model_name=model_name,
            starting_balance=starting_balance,
            stake_per_trade=stake_per_trade,
            payout_rate=payout_rate,
            martingale_multiplier=martingale_multiplier,
            threshold=threshold,
            use_pattern_filter=use_pattern_filter,
        )
        for model_name in model_names
    }
