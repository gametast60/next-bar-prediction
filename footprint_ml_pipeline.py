"""
footprint_ml_pipeline.py
------------------------
Pipeline สำหรับสร้างโมเดลสถิติ/ML จากข้อมูล footprint ของ Dukascopy tick data

ขั้นตอน:
1. อ่าน tick CSV ที่ export มาจาก TickExporter.java
2. รวม tick เป็นแท่ง (bar) ตาม timeframe ที่กำหนด แล้วสร้าง footprint ต่อแท่ง
   (buy/sell แยกด้วย tick rule เหมือนโค้ด Java ต้นฉบับ)
3. แปลง footprint เป็น feature เชิงตัวเลข (delta, imbalance count, stacked run,
   ระยะห่างจาก POC ฯลฯ)
4. ตั้ง label = ทิศทางของแท่งถัดไป
5. เทรนโมเดล baseline ด้วย walk-forward validation (ไม่ใช่สุ่มแบ่ง train/test
   เพราะข้อมูลเป็น time series จะเกิด lookahead bias)
6. Backtest แบบจำลองการเข้าออกจริง พร้อมค่า spread ประมาณการ

ต้องติดตั้ง: pandas, numpy, scikit-learn
    pip install pandas numpy scikit-learn --break-system-packages

คำเตือน: นี่คือเครื่องมือวิจัย ไม่ใช่คำแนะนำการลงทุน ผลจาก backtest ในอดีต
ไม่ได้การันตีผลในอนาคต ควรทดสอบด้วยข้อมูล out-of-sample เพิ่มเติมก่อนใช้เงินจริงเสมอ
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


# ============================================================
# 1) โหลด tick data
# ============================================================
def load_ticks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_transaction_cost(csv_path: Path) -> tuple[str, float]:
    """เลือก spread จำลองต่อการเทรดจากสัญลักษณ์ในชื่อไฟล์"""
    symbol = csv_path.stem.upper()

    if "XAU" in symbol or "GOLD" in symbol:
        return "XAUUSD", 0.30
    if "EURUSD" in symbol:
        return "EURUSD", 0.00010
    if any(pair in symbol for pair in ("USDJPY", "EURJPY", "GBPJPY")):
        return "FOREX_JPY", 0.010
    return "FOREX", 0.00015


# ============================================================
# 1b) ประมาณ bucket_size / imbalance_ratio อัตโนมัติจากข้อมูลจริงของ pair นั้นๆ
#     แทนใส่ค่าตายตัว (0.0001, 3.0 เดิม ถูก fix ไว้สำหรับ forex ทั่วไป
#     ไม่เหมาะกับทองคำหรือ instrument ที่ scale ราคาต่างออกไป)
# ============================================================
def _nice_number(x: float) -> float:
    """ปัดค่าให้เป็นเลขกลมๆ (1/2/5 x 10^n) อ่านง่ายและไม่เล็ก/ใหญ่เกินไป"""
    if x <= 0 or not np.isfinite(x):
        return 0.0001
    exponent = np.floor(np.log10(x))
    fraction = x / (10 ** exponent)
    if fraction < 1.5:
        nice = 1
    elif fraction < 3:
        nice = 2
    elif fraction < 7:
        nice = 5
    else:
        nice = 10
    return float(nice * (10 ** exponent))


def estimate_bucket_size(
    ticks: pd.DataFrame,
    bar_freq: str = "5min",
    target_rows_per_bar: int = 20,
) -> float:
    """
    bucket_size = median(high-low ต่อแท่ง) / จำนวน price-level ที่อยากได้ต่อแท่ง
    แล้วปัดเป็นเลขกลม เพื่อให้ scale ตามราคาของ instrument โดยอัตโนมัติ
    (ทองคำ ~2600 -> bucket ใหญ่กว่า EURUSD ~1.05 หลายพันเท่าโดยอัตโนมัติ)
    """
    tmp = ticks[["timestamp", "mid"]].copy()
    tmp["bar_time"] = tmp["timestamp"].dt.floor(bar_freq)
    ranges = tmp.groupby("bar_time")["mid"].agg(lambda s: s.max() - s.min())
    ranges = ranges[ranges > 0]
    typical_range = ranges.median() if len(ranges) else ticks["mid"].std()
    if not typical_range or typical_range <= 0:
        typical_range = ticks["mid"].std() or 0.0001
    return _nice_number(typical_range / target_rows_per_bar)


def estimate_imbalance_ratio(
    ticks: pd.DataFrame,
    bucket_size: float,
    bar_freq: str = "5min",
    percentile: float = 85.0,
) -> float:
    """
    หา imbalance_ratio จาก percentile ของอัตราส่วน (vol ฝั่งมาก / vol ฝั่งน้อย)
    ที่เกิดขึ้นจริงระหว่าง bucket ราคาที่ติดกันในข้อมูลชุดนี้ แทนใช้ 3.0 ตายตัว
    ค่านี้ควบคุม "ต้องต่างกันแค่ไหนถึงเรียกว่า imbalance" ซึ่งพฤติกรรม
    การกระจาย volume ของแต่ละ pair ไม่เหมือนกัน
    """
    tmp = ticks.copy()
    tmp["bar_time"] = tmp["timestamp"].dt.floor(bar_freq)
    mid_diff = tmp["mid"].diff()
    direction = np.sign(mid_diff).replace(0, np.nan).ffill().fillna(1)
    total_vol = (tmp["bidVolume"] + tmp["askVolume"]).sum()
    vol = pd.Series(1.0, index=tmp.index) if total_vol <= 0 else (tmp["bidVolume"] + tmp["askVolume"]) / 2.0
    tmp["bucket_key"] = np.round(tmp["mid"] / bucket_size).astype(np.int64)
    tmp["buy_vol"] = np.where(direction > 0, vol, 0.0)
    tmp["sell_vol"] = np.where(direction < 0, vol, 0.0)

    ratios = []
    for _, g in tmp.groupby("bar_time", sort=True):
        row_stats = (
            g.groupby("bucket_key")
            .agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"))
            .reset_index()
            .sort_values("bucket_key")
            .reset_index(drop=True)
        )
        buy = row_stats["buy_vol"].values
        sell = row_stats["sell_vol"].values
        n = len(row_stats)
        for i in range(n):
            if i > 0 and sell[i - 1] > 0 and buy[i] > 0:
                ratios.append(buy[i] / sell[i - 1])
            if i < n - 1 and buy[i + 1] > 0 and sell[i] > 0:
                ratios.append(sell[i] / buy[i + 1])

    if not ratios:
        return 3.0
    return float(max(1.5, np.percentile(ratios, percentile)))


# ============================================================
# 2) สร้าง footprint ต่อแท่ง + แตกเป็น feature
# ============================================================
def build_footprint_features(
    ticks: pd.DataFrame,
    bar_freq: str = "5min",
    bucket_size: float = None,
    imbalance_ratio: float = None,
    min_stacked_rows: int = 3,
) -> pd.DataFrame:
    # ถ้าไม่ระบุมา ให้ประมาณอัตโนมัติจากข้อมูลจริงของ instrument นี้
    if bucket_size is None:
        bucket_size = estimate_bucket_size(ticks, bar_freq=bar_freq)
        print(f"   [auto] bucket_size = {bucket_size}")
    if imbalance_ratio is None:
        imbalance_ratio = estimate_imbalance_ratio(ticks, bucket_size, bar_freq=bar_freq)
        print(f"   [auto] imbalance_ratio = {imbalance_ratio:.2f}")

    # เช็คว่ามี volume จริงไหม (บาง instrument ของ Dukascopy ส่ง volume เป็น 0 ทั้งหมด)
    total_vol = (ticks["bidVolume"] + ticks["askVolume"]).sum()
    use_tick_count_fallback = total_vol <= 0.0

    ticks = ticks.copy()
    ticks["bar_time"] = ticks["timestamp"].dt.floor(bar_freq)

    # tick rule: ขึ้น = buy, ลง = sell, เท่าเดิม = ตามทิศทางก่อนหน้า
    mid_diff = ticks["mid"].diff()
    direction = np.sign(mid_diff)
    direction = direction.replace(0, np.nan).ffill().fillna(1)
    ticks["direction"] = direction

    if use_tick_count_fallback:
        ticks["vol"] = 1.0
    else:
        ticks["vol"] = (ticks["bidVolume"] + ticks["askVolume"]) / 2.0

    ticks["bucket_key"] = np.round(ticks["mid"] / bucket_size).astype(np.int64)
    ticks["buy_vol"] = np.where(ticks["direction"] > 0, ticks["vol"], 0.0)
    ticks["sell_vol"] = np.where(ticks["direction"] < 0, ticks["vol"], 0.0)

    rows_out = []

    for bar_time, g in ticks.groupby("bar_time", sort=True):
        row_stats = (
            g.groupby("bucket_key")
            .agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"))
            .reset_index()
            .sort_values("bucket_key")
            .reset_index(drop=True)
        )
        row_stats["price"] = row_stats["bucket_key"] * bucket_size
        row_stats["total_vol"] = row_stats["buy_vol"] + row_stats["sell_vol"]

        n_rows = len(row_stats)
        total_delta = row_stats["buy_vol"].sum() - row_stats["sell_vol"].sum()
        total_volume = row_stats["total_vol"].sum()

        poc_idx = row_stats["total_vol"].idxmax() if n_rows > 0 else None
        poc_price = row_stats.loc[poc_idx, "price"] if poc_idx is not None else np.nan

        # imbalance ต่อแถว เทียบกับแถวราคาติดกัน (เหมือน logic ใน Java)
        buy_imb = np.zeros(n_rows, dtype=bool)
        sell_imb = np.zeros(n_rows, dtype=bool)
        for i in range(n_rows):
            if i > 0 and row_stats.loc[i - 1, "sell_vol"] > 0:
                if row_stats.loc[i, "buy_vol"] >= row_stats.loc[i - 1, "sell_vol"] * imbalance_ratio:
                    buy_imb[i] = True
            if i < n_rows - 1 and row_stats.loc[i + 1, "buy_vol"] > 0:
                if row_stats.loc[i, "sell_vol"] >= row_stats.loc[i + 1, "buy_vol"] * imbalance_ratio:
                    sell_imb[i] = True

        def max_consecutive_run(flags: np.ndarray) -> int:
            best = run = 0
            for f in flags:
                run = run + 1 if f else 0
                best = max(best, run)
            return best

        max_buy_run = max_consecutive_run(buy_imb)
        max_sell_run = max_consecutive_run(sell_imb)
        n_buy_imb_rows = int(buy_imb.sum())
        n_sell_imb_rows = int(sell_imb.sum())
        has_stacked_buy = max_buy_run >= min_stacked_rows
        has_stacked_sell = max_sell_run >= min_stacked_rows

        bar_open = g["mid"].iloc[0]
        bar_close = g["mid"].iloc[-1]
        bar_high = g["mid"].max()
        bar_low = g["mid"].min()

        rows_out.append({
            "bar_time": bar_time,
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "n_price_rows": n_rows,
            "total_delta": total_delta,
            "total_volume": total_volume,
            "poc_price": poc_price,
            "dist_close_to_poc": bar_close - poc_price if poc_idx is not None else np.nan,
            "n_buy_imbalance_rows": n_buy_imb_rows,
            "n_sell_imbalance_rows": n_sell_imb_rows,
            "max_buy_stacked_run": max_buy_run,
            "max_sell_stacked_run": max_sell_run,
            "has_stacked_buy": has_stacked_buy,
            "has_stacked_sell": has_stacked_sell,
            "used_tick_count_fallback": use_tick_count_fallback,
        })

    bars = pd.DataFrame(rows_out).sort_values("bar_time").reset_index(drop=True)
    bars["bar_return"] = bars["close"] - bars["open"]
    return bars


# ============================================================
# 2b) เพิ่ม context feature ข้ามแท่ง — ของเดิมโมเดลเห็นแค่ footprint ของ
#     แท่งปัจจุบันแท่งเดียว ไม่มีความต่อเนื่องจากแท่งก่อนหน้าเลย จึงจับ
#     "โมเมนตัม" หรือ "แนวโน้มสะสม" ไม่ได้ ฟังก์ชันนี้เพิ่ม:
#       - lag ของ delta/return/stacked-run ย้อนหลัง n_lags แท่ง
#       - ค่าเฉลี่ย/ส่วนเบี่ยงเบนของ delta แบบ rolling (โมเมนตัมระยะสั้น)
#       - cumulative delta สะสมในหน้าต่าง roll_window แท่ง
#       - ATR-แบบง่าย (rolling mean ของ high-low) วัดความผันผวนล่าสุด
#       - session dummy (Asia/London/NY ตามเวลา UTC) เพราะพฤติกรรมราคา
#         มักต่างกันตามช่วงตลาดที่เปิดอยู่
#     หมายเหตุ: ทุกค่าที่ใช้เป็น feature ของแท่ง t คำนวณจากข้อมูลที่ "รู้แล้ว
#     ตอนแท่ง t ปิด" เท่านั้น (แท่งปัจจุบันเองรวมถึงแท่งก่อนหน้า) ไม่มีการ
#     มองไปข้างหน้า label ยังคงมาจาก shift(-1) ใน add_labels ตามเดิม
# ============================================================
def add_context_features(
    bars: pd.DataFrame,
    n_lags: int = 3,
    roll_window: int = 5,
) -> pd.DataFrame:
    bars = bars.copy()

    for lag in range(1, n_lags + 1):
        bars[f"lag_delta_{lag}"] = bars["total_delta"].shift(lag)
        bars[f"lag_return_{lag}"] = bars["bar_return"].shift(lag)
        bars[f"lag_stacked_buy_{lag}"] = bars["max_buy_stacked_run"].shift(lag)
        bars[f"lag_stacked_sell_{lag}"] = bars["max_sell_stacked_run"].shift(lag)

    bars[f"rolling_delta_mean_{roll_window}"] = bars["total_delta"].rolling(roll_window).mean()
    bars[f"rolling_delta_std_{roll_window}"] = bars["total_delta"].rolling(roll_window).std()
    bars[f"cum_delta_{roll_window}"] = bars["total_delta"].rolling(roll_window).sum()
    bars[f"rolling_range_{roll_window}"] = (bars["high"] - bars["low"]).rolling(roll_window).mean()
    bars[f"rolling_vol_mean_{roll_window}"] = bars["total_volume"].rolling(roll_window).mean()

    hour_utc = bars["bar_time"].dt.hour
    bars["session_asia"] = hour_utc.between(0, 7).astype(int)
    bars["session_london"] = hour_utc.between(7, 15).astype(int)
    bars["session_ny"] = hour_utc.between(13, 21).astype(int)

    candle_range = (bars["high"] - bars["low"]).replace(0, np.nan)
    bars["body_ratio"] = (bars["close"] - bars["open"]) / candle_range
    bars["close_location"] = (bars["close"] - bars["low"]) / candle_range
    bars["upper_wick_ratio"] = (
        bars["high"] - bars[["open", "close"]].max(axis=1)
    ) / candle_range
    bars["lower_wick_ratio"] = (
        bars[["open", "close"]].min(axis=1) - bars["low"]
    ) / candle_range
    bars["momentum_3"] = bars["close"] - bars["close"].shift(3)
    bars["momentum_6"] = bars["close"] - bars["close"].shift(6)
    bars["trend_12"] = bars["close"] - bars["close"].rolling(12).mean()
    bars["range_ratio_12"] = (
        (bars["high"] - bars["low"])
        / (bars["high"] - bars["low"]).rolling(12).mean()
    )

    return bars


# ============================================================
# 3) ตั้ง label = ทิศทางแท่งถัดไป (สำคัญ: ต้อง shift(-1) เพื่อไม่ให้เห็นอนาคต)
# ============================================================
def add_labels(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["next_close"] = bars["close"].shift(-1)
    bars["next_return"] = bars["next_close"] - bars["close"]
    bars["label_up"] = (bars["next_return"] > 0).astype(int)
    bars = bars.iloc[:-1]  # ตัดแท่งสุดท้ายทิ้ง เพราะไม่มี "แท่งถัดไป" ให้เป็น label
    return bars


BASE_FEATURE_COLUMNS = [
    "total_delta",
    "total_volume",
    "n_price_rows",
    "dist_close_to_poc",
    "n_buy_imbalance_rows",
    "n_sell_imbalance_rows",
    "max_buy_stacked_run",
    "max_sell_stacked_run",
    "bar_return",
]

CONTEXT_FEATURE_COLUMNS = (
    [f"lag_delta_{i}" for i in range(1, 4)]
    + [f"lag_return_{i}" for i in range(1, 4)]
    + [f"lag_stacked_buy_{i}" for i in range(1, 4)]
    + [f"lag_stacked_sell_{i}" for i in range(1, 4)]
    + [
        "rolling_delta_mean_5",
        "rolling_delta_std_5",
        "cum_delta_5",
        "rolling_range_5",
        "rolling_vol_mean_5",
        "session_asia",
        "session_london",
        "session_ny",
        "body_ratio",
        "close_location",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "momentum_3",
        "momentum_6",
        "trend_12",
        "range_ratio_12",
    ]
)

# ใช้ทุก feature (แท่งเดียว + context ข้ามแท่ง) เป็นค่าเริ่มต้น
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


# ============================================================
# 4) เทรน + ประเมินผลด้วย walk-forward (TimeSeriesSplit)
# ============================================================
def walk_forward_eval(bars: pd.DataFrame, n_splits: int = 5):
    X = bars[FEATURE_COLUMNS].fillna(0.0).values
    y = bars["label_up"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            fold_results.append({
                "fold": fold,
                "model": name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, pred),
                "auc": roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else np.nan,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
            })

    return pd.DataFrame(fold_results)


# ============================================================
# 5) Backtest สำหรับ Binary Option: ทายทิศทางแท่งถัดไป 1 แท่ง (หมดอายุ 5 นาที)
# ============================================================
def binary_option_backtest(
    bars: pd.DataFrame,
    model,
    train_end_idx: int,
    prob_threshold: float = 0.55,
    starting_balance: float = 2000.0,
    stake_per_trade: float = 100.0,
    payout_rate: float = 0.80,
    use_pattern_filter: bool = True,
    money_management: str = "fixed_stake",
    risk_fraction: float = 0.01,
    martingale_multiplier: float = 2.0,
):
    X = bars[FEATURE_COLUMNS].fillna(0.0).values
    y_true = bars["label_up"].values

    X_test = X[train_end_idx:]
    y_test = y_true[train_end_idx:]
    pattern_test = bars.iloc[train_end_idx:]

    proba = model.predict_proba(X_test)[:, 1]
    predictions = []
    actual = []
    for test_offset, (p, y) in enumerate(zip(proba, y_test)):
        if p >= prob_threshold:
            prediction = 1
        elif p <= (1 - prob_threshold):
            prediction = 0
        else:
            continue

        if use_pattern_filter:
            body_ratio = pattern_test.iloc[test_offset]["body_ratio"]
            if (prediction == 1 and body_ratio <= 0) or (
                prediction == 0 and body_ratio >= 0
            ):
                continue

        predictions.append(prediction)
        actual.append(y)

    balance = starting_balance
    equity_curve = [balance]
    n_signals = len(predictions)
    n_correct_signals = int(np.sum(np.array(predictions) == np.array(actual)))
    n_wrong_signals = n_signals - n_correct_signals
    n_bets = 0
    n_correct_bets = 0
    n_wrong_bets = 0
    current_stake = stake_per_trade
    current_loss_streak = 0
    current_win_streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    max_stake = 0.0
    stopped_for_insufficient_balance = False

    for prediction, outcome in zip(predictions, actual):
        if money_management == "fixed_fractional":
            current_stake = balance * risk_fraction
        elif money_management == "fixed_stake":
            current_stake = stake_per_trade
        elif money_management != "martingale":
            raise ValueError(
                "money_management ต้องเป็น fixed_fractional, fixed_stake หรือ martingale"
            )

        if balance < current_stake:
            stopped_for_insufficient_balance = True
            break

        n_bets += 1
        max_stake = max(max_stake, current_stake)
        if prediction == outcome:
            balance += current_stake * payout_rate
            n_correct_bets += 1
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
            if money_management == "martingale":
                current_stake = stake_per_trade
        else:
            balance -= current_stake
            n_wrong_bets += 1
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            if money_management == "martingale":
                current_stake *= martingale_multiplier
        equity_curve.append(balance)

    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.min(equity_curve - peak))

    return {
        "n_candles_tested": len(y_test),
        "n_signals": n_signals,
        "n_correct_signals": n_correct_signals,
        "n_wrong_signals": n_wrong_signals,
        "accuracy_on_signals": n_correct_signals / n_signals if n_signals else np.nan,
        "n_bets": n_bets,
        "n_correct_bets": n_correct_bets,
        "n_wrong_bets": n_wrong_bets,
        "accuracy_on_bets": n_correct_bets / n_bets if n_bets else np.nan,
        "coverage": n_signals / len(y_test) if len(y_test) else np.nan,
        "starting_balance": starting_balance,
        "money_management": money_management,
        "risk_fraction": risk_fraction if money_management == "fixed_fractional" else None,
        "martingale_multiplier": martingale_multiplier if money_management == "martingale" else None,
        "final_balance": balance,
        "net_profit": balance - starting_balance,
        "max_drawdown": max_drawdown,
        "max_stake": max_stake,
        "max_loss_streak": max_loss_streak,
        "max_win_streak": max_win_streak,
        "stopped_for_insufficient_balance": stopped_for_insufficient_balance,
    }


def print_backtest_summary(name: str, result: dict) -> None:
    status = "พอร์ตแตก" if result["stopped_for_insufficient_balance"] else "ทดสอบครบ"
    print(f"\n[{name}]")
    print(f"สัญญาณทั้งหมด: {result['n_signals']} ครั้ง")
    print(f"เข้าเงินจริง: {result['n_bets']} ไม้")
    print(f"ทายถูก: {result['n_correct_bets']} ไม้")
    print(f"ทายผิด: {result['n_wrong_bets']} ไม้")
    print(f"Accuracy ของไม้ที่เข้า: {result['accuracy_on_bets']:.2%}")
    print(f"ทุนเริ่มต้น: {result['starting_balance']:.2f} บาท")
    print(f"ทุนสุดท้าย: {result['final_balance']:.2f} บาท")
    print(f"กำไร/ขาดทุนสุทธิ: {result['net_profit']:+.2f} บาท")
    print(f"เดิมพันสูงสุด: {result['max_stake']:.2f} บาท")
    print(f"แพ้ติดกันสูงสุด: {result['max_loss_streak']} ครั้ง")
    print(f"ชนะติดกันสูงสุด: {result['max_win_streak']} ครั้ง")
    print(f"สถานะ: {status}")


# ============================================================
# ตัวอย่างการใช้งาน
# ============================================================
if __name__ == "__main__":
    CSV_PATH = Path(__file__).resolve().parent / "XAUUSD_ticks.csv"
    BAR_FREQ = "5min"
    MIN_STACKED_ROWS = 3
    # BUCKET_SIZE / IMBALANCE_RATIO: ไม่ระบุ -> ให้ประมาณอัตโนมัติจาก
    # ข้อมูลจริงของ pair นี้ (ดู estimate_bucket_size / estimate_imbalance_ratio)
    # ถ้าอยากบังคับค่าเองก็ยังใส่ตัวเลขตรงนี้ได้ตามปกติ

    print("1) โหลด tick data...")
    ticks = load_ticks(CSV_PATH)

    print("2) สร้าง footprint feature ต่อแท่ง (auto-tune bucket/imbalance)...")
    bars = build_footprint_features(
        ticks, bar_freq=BAR_FREQ, bucket_size=None,
        imbalance_ratio=None, min_stacked_rows=MIN_STACKED_ROWS,
    )

    print("2b) เพิ่ม context feature ข้ามแท่ง (lag/rolling/session)...")
    bars = add_context_features(bars, n_lags=3, roll_window=5)

    bars = add_labels(bars)
    bars = bars.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"   ได้ {len(bars)} แท่ง (หลังตัดแถวที่ยังไม่มี lag/rolling ครบ)")

    print("3) walk-forward evaluation (5 folds)...")
    results = walk_forward_eval(bars, n_splits=5)
    print(results.groupby("model")[["accuracy", "auc", "precision", "recall"]].mean())

    print("4) backtest บนช่วงท้ายสุดของข้อมูล (out-of-sample)...")
    split_point = int(len(bars) * 0.8)
    X_train = bars[FEATURE_COLUMNS].fillna(0.0).values[:split_point]
    y_train = bars["label_up"].values[:split_point]

    final_model = GradientBoostingClassifier(random_state=42)
    final_model.fit(X_train, y_train)

    print("   binary option: expiry = 1 แท่ง (5 นาที), threshold = 0.55")
    print("   pattern filter: candle body confirmation = ON")
    print("   เปรียบเทียบ money management: fixed stake 100 บาท vs martingale")
    fixed_bt = binary_option_backtest(
        bars, final_model, train_end_idx=split_point,
        prob_threshold=0.55, starting_balance=2000.0,
        stake_per_trade=100.0, payout_rate=0.80,
        money_management="fixed_stake",
    )
    martingale_bt = binary_option_backtest(
        bars, final_model, train_end_idx=split_point,
        prob_threshold=0.55, starting_balance=2000.0,
        stake_per_trade=100.0, payout_rate=0.80,
        money_management="martingale",
    )
    print_backtest_summary("fixed stake 100 บาท", fixed_bt)
    print_backtest_summary("martingale เริ่ม 100 บาท", martingale_bt)

    input("\nกด Enter เพื่อปิดโปรแกรม...")