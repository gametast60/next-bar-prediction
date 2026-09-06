package com.dukascopy.indicators;

import com.dukascopy.api.*;
import com.dukascopy.api.indicators.*;
import java.awt.*;
import java.text.*;
import java.util.*;
import java.util.List;

public class NextBarPredictionIndicator implements IIndicator, IDrawingIndicator {

    // ============================================================
    // 1. CONSTANTS (ค่าคงที่ของโมเดล)
    // ============================================================
    private static final int TREND_WINDOW = 13;
    private static final int N_FEATURES = 37;
    private static final long PERIOD_MS = 5 * 60 * 1000;

    // ---- Backtest money-management constants ----
    private static final double BT_STARTING_CAPITAL = 2000.0;
    private static final double BT_FIXED_STAKE = 100.0;
    private static final double BT_PAYOUT_RATIO = 0.8; // win = +stake*0.8, lose = -stake
    private static final double BT_MARTINGALE_BASE = 100.0;
    private static final double BT_MARTINGALE_MULTIPLIER = 2.0;

    // Model coefficients (exported from Python)
    private static final double INTERCEPT = 0.36239545361756081;
    private static final double[] COEFFICIENTS = {
        0.065908720668599735, 0.056006082341678837, 0.0088159937475023546,
        -0.0032968701198590649, -0.0014426245802901976, -0.0062006136801055385,
        0.026323345978475524, -0.049463358111213232, -0.0036460299765007343,
        -0.11289925473519764, -0.30204606314984178, -0.44275906033523821,
        -0.013935818239363996, -0.0063101336268783104, 0.0028618197608080713,
        -0.063165723712000396, -0.038093206158860339, 0.0600071526971118,
        -0.040142914150192603, 0.0038220534268405587, -0.031833376189938846,
        -0.1865363083292679, 0.093006040002456283, -0.93268154164627659,
        -0.016260980135582907, -0.1719632055238442, -0.0010104964198183425,
        0.010828705155967594, -0.013755554933018639, -0.014628324338586296,
        -0.12829741155125107, 0.049863223816356078, -0.32365752405226683,
        0.0087027254303474119, -0.00081910212045794672, -0.0082198103264951236,
        -0.15269924495124687
    };

    // ============================================================
    // 2. OUTPUT INDICES
    // ============================================================
    private static final int OUT_BUY_ARROW = 0;
    private static final int OUT_SELL_ARROW = 1;
    private static final int OUT_BUY_LABEL = 2;
    private static final int OUT_SELL_LABEL = 3;
    private final double[][] outputs = new double[4][];

    // ============================================================
    // 3. INSTANCE VARIABLES (สำหรับ Opt Inputs)
    // ============================================================
    private IHistory history;
    private IConsole console;
    private Instrument instrument;

    // พารามิเตอร์ที่ผู้ใช้ปรับได้ (ค่าเริ่มต้น)
    private double threshold = 0.5;
    private boolean patternFilter = true;
    private double bucketSize = 0.5;      // เหมาะกับ XAUUSD
    private double imbalanceRatio = 4.0;  // ratio (ไม่ใช่ %)
    private boolean backtestEnabled = false; // เปิด/ปิดโหมด backtest
    private int backtestLookback = 50;       // มองย้อนกลับไปกี่ "สัญญาณ" (ไม่ใช่แท่งเทียน)

    // แคชข้อมูลดิบ (เพื่อไม่ต้องดึง Tick ซ้ำ)
    private Map<Long, RawBarData> rawCache = new HashMap<>();

    // ============================================================
    // 4. INNER CLASSES
    // ============================================================
    /** เก็บข้อมูลดิบจาก Tick (ยังไม่คำนวณ imbalance/POC) */
    private static class RawBarData {
        long timestamp;
        double open, high, low, close;
        Map<Long, Double> buyVol;   // key = bucket index, value = volume
        Map<Long, Double> sellVol;
        Set<Long> keys;            // bucket keys ทั้งหมด

        RawBarData() {
            buyVol = new HashMap<>();
            sellVol = new HashMap<>();
            keys = new HashSet<>();
        }
    }

    /** เก็บข้อมูลที่คำนวณแล้ว (features, signal) สำหรับแท่งใดแท่งหนึ่ง */
    private static class BarData {
        long timestamp;
        double open, high, low, close;
        double buyVolume, sellVolume;
        int priceLevels;
        double pocPrice;
        int buyImbalanceRows, sellImbalanceRows;
        int maxBuyStackedRun, maxSellStackedRun;
        double[] features;
        double probability;
        String signal;
        BarData() {
            features = new double[N_FEATURES];
        }
    }

    // ============================================================
    // 5. INDICATOR LIFECYCLE
    // ============================================================
    @Override
    public void onStart(IIndicatorContext context) {
        this.history = context.getHistory();
        this.console = context.getConsole();
        this.instrument = context.getInstrument();

        // ตั้งค่า IndicatorInfo
        indicatorInfo = new IndicatorInfo("NBP", "Next Bar Prediction", "Custom",
                true, false, false, 1, 6, 4);

        inputParameterInfos = new InputParameterInfo[]{
                new InputParameterInfo("Price", InputParameterInfo.Type.PRICE)
        };

        // Opt Inputs (4 ตัว)
        optInputParameterInfos = new OptInputParameterInfo[]{
                new OptInputParameterInfo("Threshold", OptInputParameterInfo.Type.OTHER,
                        new DoubleRangeDescription(0.5, 0.1, 0.9, 0.01, 2)),
                new OptInputParameterInfo("Pattern Filter", OptInputParameterInfo.Type.OTHER,
                        new BooleanOptInputDescription(true)),
                new OptInputParameterInfo("Bucket Size", OptInputParameterInfo.Type.OTHER,
                        new DoubleRangeDescription(0.5, 0.001, 10.0, 0.001, 3)),
                new OptInputParameterInfo("Imbalance Ratio", OptInputParameterInfo.Type.OTHER,
                        new DoubleRangeDescription(4.0, 1.0, 10.0, 0.1, 1)),
                new OptInputParameterInfo("Backtest", OptInputParameterInfo.Type.OTHER,
                        new BooleanOptInputDescription(false)),
                new OptInputParameterInfo("Backtest Lookback (signals)", OptInputParameterInfo.Type.OTHER,
                        new IntegerRangeDescription(50, 5, 1000, 1))
        };

        outputParameterInfos = new OutputParameterInfo[]{
                new OutputParameterInfo("Buy Arrow", OutputParameterInfo.Type.DOUBLE, OutputParameterInfo.DrawingStyle.LINE),
                new OutputParameterInfo("Sell Arrow", OutputParameterInfo.Type.DOUBLE, OutputParameterInfo.DrawingStyle.LINE),
                new OutputParameterInfo("Buy Label", OutputParameterInfo.Type.DOUBLE, OutputParameterInfo.DrawingStyle.LINE),
                new OutputParameterInfo("Sell Label", OutputParameterInfo.Type.DOUBLE, OutputParameterInfo.DrawingStyle.LINE)
        };
        outputParameterInfos[OUT_BUY_ARROW].setColor(Color.GREEN);
        outputParameterInfos[OUT_SELL_ARROW].setColor(Color.RED);
        outputParameterInfos[OUT_BUY_LABEL].setColor(Color.GREEN);
        outputParameterInfos[OUT_SELL_LABEL].setColor(Color.RED);
        for (int i = 0; i < 4; i++) {
            outputParameterInfos[i].setDrawnByIndicator(true);
        }
    }

    @Override
    public IndicatorResult calculate(int startIndex, int endIndex) {
        // รับ output arrays
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] == null) {
                return new IndicatorResult(startIndex, endIndex - startIndex + 1);
            }
        }
        // ตั้งค่าเป็น NaN (เผื่อไว้)
        for (int i = startIndex; i <= endIndex; i++) {
            for (int out = 0; out < outputs.length; out++) {
                if (i < outputs[out].length) {
                    outputs[out][i] = Double.NaN;
                }
            }
        }
        return new IndicatorResult(startIndex, endIndex - startIndex + 1);
    }

    // ============================================================
    // 6. DRAWING – เรียกทุกครั้งที่มีการวาด
    // ============================================================
    @Override
    public Point drawOutput(Graphics g, int outputIdx, Object values, Color color, Stroke stroke,
                             IIndicatorDrawingSupport support, List<Shape> shapes,
                             Map<Color, List<Point>> handles) {

        IBar[] candles = support.getCandles();
        if (candles == null || candles.length < TREND_WINDOW + 2) {
            return null;
        }

        // คำนวณสัญญาณด้วยพารามิเตอร์ปัจจุบัน
        SignalResult result = calculateSignals(candles);
        if (result == null) {
            return null;
        }

        double[] buyPrices = result.buyPrices;
        double[] sellPrices = result.sellPrices;

        // วาดตาม outputIdx
        if (outputIdx == OUT_BUY_ARROW) {
            drawMarkers(g, true, buyPrices, support);
        } else if (outputIdx == OUT_SELL_ARROW) {
            drawMarkers(g, false, sellPrices, support);
        } else if (outputIdx == OUT_BUY_LABEL) {
            drawLabels(g, true, buyPrices, support);
        } else if (outputIdx == OUT_SELL_LABEL) {
            drawLabels(g, false, sellPrices, support);
        }

        // วาดกล่องสรุปผล backtest แค่ครั้งเดียวต่อรอบ (ตอน output สุดท้าย) กันทับซ้อน
        if (backtestEnabled && outputIdx == OUT_SELL_LABEL) {
            BacktestStats stats = computeBacktest(result.backtestSignals, backtestLookback);
            drawBacktestPanel(g, stats, candles, support);
        }
        return null;
    }

    // ============================================================
    // 7. CALCULATE SIGNALS (ใช้ Opt Inputs ปัจจุบัน)
    // ============================================================
    private SignalResult calculateSignals(IBar[] candles) {
        int len = candles.length;
        double[] buyPrices = new double[len];
        double[] sellPrices = new double[len];
        Arrays.fill(buyPrices, Double.NaN);
        Arrays.fill(sellPrices, Double.NaN);

        List<BarData> bars = new ArrayList<>();
        List<BacktestSignal> backtestSignals = new ArrayList<>();
        int buy = 0, sell = 0;

        for (int i = 0; i < len; i++) {
            IBar bar = candles[i];
            long t = bar.getTime();

            // สร้าง RawBarData (ถ้ายังไม่มี) โดยดึง Tick จริง (ไม่มี fallback)
            RawBarData raw = rawCache.get(t);
            if (raw == null) {
                raw = buildRawBarData(bar);
                if (raw != null) {
                    rawCache.put(t, raw);
                } else {
                    // ถ้าไม่สามารถสร้าง raw ได้ (ไม่มี tick) ให้ข้าม
                    continue;
                }
            }

            // แปลง RawBarData เป็น BarData (คำนวณ footprint ด้วย opt ปัจจุบัน)
            BarData data = convertRawToBarData(raw);
            bars.add(data);

            if (bars.size() >= TREND_WINDOW) {
                updateFeaturesForLast(bars); // ใช้ opt ปัจจุบัน (อ่านจาก instance)
            }

            int lookback = TREND_WINDOW + 1;
            if (i >= lookback && i < len - 1) {
                BarData cur = bars.get(bars.size() - 1);
                // ตรวจสอบว่ามีฟีเจอร์ที่ไม่เป็นศูนย์ (อย่างน้อยบางตัว)
                boolean hasFeatures = false;
                for (double v : cur.features) {
                    if (v != 0.0) { hasFeatures = true; break; }
                }
                if (!hasFeatures) continue;

                double prob = predictProbability(cur.features);
                cur.probability = prob;
                String signal = "NO_SIGNAL";
                if (prob >= threshold) signal = "UP";
                else if (prob <= 1.0 - threshold) signal = "DOWN";

                if (patternFilter && !"NO_SIGNAL".equals(signal)) {
                    double body = cur.close - cur.open;
                    if ("UP".equals(signal) && body <= 0) signal = "NO_SIGNAL";
                    else if ("DOWN".equals(signal) && body >= 0) signal = "NO_SIGNAL";
                }

                cur.signal = signal;
                if ("UP".equals(signal)) {
                    buy++;
                    buyPrices[i] = cur.low;
                } else if ("DOWN".equals(signal)) {
                    sell++;
                    sellPrices[i] = cur.high;
                }

                // ---- Backtest: เปรียบเทียบกับทิศทางของ "แท่งถัดไป" ----
                // ถัดไป = candles[i+1] (มีแน่นอนเพราะเงื่อนไข i < len-1 ด้านบน)
                // แท่งขึ้น = close > open, แท่งลง = close < open ของแท่งถัดไป
                if (("UP".equals(signal) || "DOWN".equals(signal))) {
                    IBar nextBar = candles[i + 1];
                    double nextBody = nextBar.getClose() - nextBar.getOpen();
                    if (nextBody != 0.0) { // แท่ง Doji (body=0) ไม่นับผลแพ้/ชนะ ข้ามไป
                        boolean win = ("UP".equals(signal) && nextBody > 0)
                                || ("DOWN".equals(signal) && nextBody < 0);
                        backtestSignals.add(new BacktestSignal(signal, win));
                    }
                }
            }
        }

        return new SignalResult(buyPrices, sellPrices, buy, sell, backtestSignals);
    }

    // ============================================================
    // 7b. BACKTEST — เดินเงินคงที่ vs Martingale
    // ============================================================
    /**
     * ใช้สัญญาณ N ตัวล่าสุด (backtestLookback) มาจำลองการเดินเงิน 2 แบบ:
     *  - เดินเงินคงที่ (Fixed): เดิมพัน 100 บาททุกครั้ง ชนะ +80 แพ้ -100
     *  - Martingale x2 จาก 100: แพ้ -> เดิมพัน*2 ไม้ถัดไป, ชนะ -> รีเซ็ตกลับ 100
     * ทุนเริ่มต้นของทั้ง 2 แบบ = 2000 บาท (BT_STARTING_CAPITAL)
     */
    private BacktestStats computeBacktest(List<BacktestSignal> allSignals, int lookback) {
        if (allSignals == null || allSignals.isEmpty()) return null;

        int fromIndex = Math.max(0, allSignals.size() - lookback);
        List<BacktestSignal> window = allSignals.subList(fromIndex, allSignals.size());
        if (window.isEmpty()) return null;

        BacktestStats st = new BacktestStats();
        st.totalSignals = window.size();

        double fixedCapital = BT_STARTING_CAPITAL;
        double fixedPeak = BT_STARTING_CAPITAL;
        int fixedWinStreak = 0, fixedLoseStreak = 0;

        double martCapital = BT_STARTING_CAPITAL;
        double martPeak = BT_STARTING_CAPITAL;
        double martStake = BT_MARTINGALE_BASE;
        double martMaxStakeUsed = BT_MARTINGALE_BASE;
        int martWinStreak = 0, martLoseStreak = 0;

        int wins = 0, losses = 0;

        for (BacktestSignal s : window) {
            if (s.win) {
                wins++;
                // Fixed
                fixedCapital += BT_FIXED_STAKE * BT_PAYOUT_RATIO;
                fixedWinStreak++; fixedLoseStreak = 0;
                st.fixedMaxWinStreak = Math.max(st.fixedMaxWinStreak, fixedWinStreak);
                // Martingale
                martCapital += martStake * BT_PAYOUT_RATIO;
                martWinStreak++; martLoseStreak = 0;
                st.martMaxWinStreak = Math.max(st.martMaxWinStreak, martWinStreak);
                martStake = BT_MARTINGALE_BASE; // รีเซ็ตกลับ 100 หลังชนะ
            } else {
                losses++;
                // Fixed
                fixedCapital -= BT_FIXED_STAKE;
                fixedLoseStreak++; fixedWinStreak = 0;
                st.fixedMaxLoseStreak = Math.max(st.fixedMaxLoseStreak, fixedLoseStreak);
                // Martingale
                martCapital -= martStake;
                martLoseStreak++; martWinStreak = 0;
                st.martMaxLoseStreak = Math.max(st.martMaxLoseStreak, martLoseStreak);
                martStake *= BT_MARTINGALE_MULTIPLIER; // แพ้ -> เพิ่มเดิมพันเป็น 2 เท่า
                martMaxStakeUsed = Math.max(martMaxStakeUsed, martStake);
            }

            fixedPeak = Math.max(fixedPeak, fixedCapital);
            st.fixedMaxDrawdown = Math.min(st.fixedMaxDrawdown, fixedCapital - fixedPeak);

            martPeak = Math.max(martPeak, martCapital);
            st.martMaxDrawdown = Math.min(st.martMaxDrawdown, martCapital - martPeak);
        }

        st.wins = wins;
        st.losses = losses;
        st.winRate = st.totalSignals > 0 ? (100.0 * wins / st.totalSignals) : 0.0;

        st.fixedFinalCapital = fixedCapital;
        st.fixedPnl = fixedCapital - BT_STARTING_CAPITAL;

        st.martFinalCapital = martCapital;
        st.martPnl = martCapital - BT_STARTING_CAPITAL;
        st.martMaxStakeUsed = martMaxStakeUsed;

        return st;
    }

    // ============================================================
    // 8. BUILD RAW DATA FROM TICKS (ไม่มี fallback)
    // ============================================================
    private RawBarData buildRawBarData(IBar bar) {
        long barStart = bar.getTime();
        long barEnd = barStart + PERIOD_MS;
        try {
            List<ITick> ticks = history.getTicks(instrument, barStart, barEnd);
            if (ticks == null || ticks.isEmpty()) {
                // ไม่มี Tick -> ไม่สามารถสร้าง RawBarData ได้
                return null;
            }

            // ตรวจสอบ volume รวม
            double totalVol = 0;
            for (ITick t : ticks) {
                totalVol += t.getAskVolume() + t.getBidVolume();
            }
            if (totalVol <= 0) {
                // ไม่มี volume จริง -> ไม่ใช้ fallback, ให้ return null (จะข้ามแท่งนี้)
                console.getInfo().println("NBP: No real volume for bar " + bar.getTime() + ", skipping.");
                return null;
            }

            RawBarData raw = new RawBarData();
            raw.timestamp = bar.getTime();
            raw.open = bar.getOpen();
            raw.high = bar.getHigh();
            raw.low = bar.getLow();
            raw.close = bar.getClose();

            double prevMid = Double.NaN;
            int direction = 1;

            for (ITick t : ticks) {
                double mid = (t.getBid() + t.getAsk()) / 2.0;
                double vol = (t.getAskVolume() + t.getBidVolume()) / 2.0;

                // ทิศทาง
                if (Double.isNaN(prevMid) || mid > prevMid) direction = 1;
                else if (mid < prevMid) direction = -1;
                // else keep previous direction

                long key = Math.round(mid / bucketSize);
                raw.keys.add(key);

                if (direction > 0) {
                    raw.buyVol.put(key, raw.buyVol.getOrDefault(key, 0.0) + vol);
                } else {
                    raw.sellVol.put(key, raw.sellVol.getOrDefault(key, 0.0) + vol);
                }
                prevMid = mid;
            }
            return raw;
        } catch (JFException e) {
            console.getErr().println("NBP: Error fetching ticks: " + e.getMessage());
            return null;
        }
    }

    // ============================================================
    // 9. CONVERT RAW → BAR DATA (คำนวณ Footprint ด้วย opt ปัจจุบัน)
    // ============================================================
    private BarData convertRawToBarData(RawBarData raw) {
        BarData data = new BarData();
        data.timestamp = raw.timestamp;
        data.open = raw.open;
        data.high = raw.high;
        data.low = raw.low;
        data.close = raw.close;

        // รวม volume
        double buyTotal = 0, sellTotal = 0;
        for (Map.Entry<Long, Double> e : raw.buyVol.entrySet()) {
            buyTotal += e.getValue();
        }
        for (Map.Entry<Long, Double> e : raw.sellVol.entrySet()) {
            sellTotal += e.getValue();
        }
        data.buyVolume = buyTotal;
        data.sellVolume = sellTotal;
        data.priceLevels = raw.keys.size();

        // คำนวณ POC
        Map<Long, Double> totalVol = new HashMap<>();
        for (Long k : raw.keys) {
            double b = raw.buyVol.getOrDefault(k, 0.0);
            double s = raw.sellVol.getOrDefault(k, 0.0);
            totalVol.put(k, b + s);
        }
        double maxVol = 0;
        long pocKey = 0;
        for (Map.Entry<Long, Double> e : totalVol.entrySet()) {
            if (e.getValue() > maxVol) {
                maxVol = e.getValue();
                pocKey = e.getKey();
            }
        }
        data.pocPrice = pocKey * bucketSize;

        // คำนวณ imbalance (ตาม opt ratio)
        List<Long> sortedKeys = new ArrayList<>(raw.keys);
        Collections.sort(sortedKeys);
        int buyImb = 0, sellImb = 0;
        int maxBuyRun = 0, maxSellRun = 0;
        int curBuyRun = 0, curSellRun = 0;

        for (int i = 0; i < sortedKeys.size(); i++) {
            Long k = sortedKeys.get(i);
            double buy = raw.buyVol.getOrDefault(k, 0.0);
            double sell = raw.sellVol.getOrDefault(k, 0.0);

            boolean isBuyImb = false;
            boolean isSellImb = false;

            if (i > 0) {
                Long prevKey = sortedKeys.get(i - 1);
                double prevSell = raw.sellVol.getOrDefault(prevKey, 0.0);
                if (prevSell > 0 && buy >= prevSell * imbalanceRatio) {
                    isBuyImb = true;
                }
            }

            if (i < sortedKeys.size() - 1) {
                Long nextKey = sortedKeys.get(i + 1);
                double nextBuy = raw.buyVol.getOrDefault(nextKey, 0.0);
                if (nextBuy > 0 && sell >= nextBuy * imbalanceRatio) {
                    isSellImb = true;
                }
            }

            if (isBuyImb) {
                buyImb++;
                curBuyRun++;
                curSellRun = 0;
                maxBuyRun = Math.max(maxBuyRun, curBuyRun);
            } else if (isSellImb) {
                sellImb++;
                curSellRun++;
                curBuyRun = 0;
                maxSellRun = Math.max(maxSellRun, curSellRun);
            } else {
                curBuyRun = 0;
                curSellRun = 0;
            }
        }

        data.buyImbalanceRows = buyImb;
        data.sellImbalanceRows = sellImb;
        data.maxBuyStackedRun = maxBuyRun;
        data.maxSellStackedRun = maxSellRun;

        return data;
    }

    // ============================================================
    // 10. UPDATE FEATURES (ใช้ opt ปัจจุบัน)
    // ============================================================
    private void updateFeaturesForLast(List<BarData> bars) {
        int i = bars.size() - 1;
        BarData cur = bars.get(i);
        double[] f = cur.features;

        f[0] = cur.buyVolume - cur.sellVolume;
        f[1] = cur.buyVolume + cur.sellVolume;
        f[2] = cur.priceLevels;
        f[3] = cur.close - cur.pocPrice;
        f[4] = cur.buyImbalanceRows;
        f[5] = cur.sellImbalanceRows;
        f[6] = cur.maxBuyStackedRun;
        f[7] = cur.maxSellStackedRun;
        f[8] = cur.close - cur.open;

        if (i >= 1) {
            f[9] = bars.get(i-1).features[0];
            f[12] = bars.get(i-1).features[8];
            f[15] = bars.get(i-1).features[6];
            f[18] = bars.get(i-1).features[7];
        }
        if (i >= 2) {
            f[10] = bars.get(i-2).features[0];
            f[13] = bars.get(i-2).features[8];
            f[16] = bars.get(i-2).features[6];
            f[19] = bars.get(i-2).features[7];
        }
        if (i >= 3) {
            f[11] = bars.get(i-3).features[0];
            f[14] = bars.get(i-3).features[8];
            f[17] = bars.get(i-3).features[6];
            f[20] = bars.get(i-3).features[7];
        }

        if (i >= 4) {
            double sumD = 0, sumSqD = 0, sumV = 0, sumHL = 0;
            for (int j = i-4; j <= i; j++) {
                sumD += bars.get(j).features[0];
                sumV += bars.get(j).features[1];
                sumHL += bars.get(j).high - bars.get(j).low;
            }
            double meanD = sumD / 5;
            for (int j = i-4; j <= i; j++) {
                double diff = bars.get(j).features[0] - meanD;
                sumSqD += diff * diff;
            }
            f[21] = meanD;
            f[22] = Math.sqrt(sumSqD / 5);
            f[23] = sumD;
            f[24] = sumHL / 5;
            f[25] = sumV / 5;
        }

        Calendar cal = Calendar.getInstance(TimeZone.getTimeZone("UTC"));
        cal.setTimeInMillis(cur.timestamp);
        int hour = cal.get(Calendar.HOUR_OF_DAY);
        f[26] = (hour >= 0 && hour <= 7) ? 1.0 : 0.0;
        f[27] = (hour >= 7 && hour <= 15) ? 1.0 : 0.0;
        f[28] = (hour >= 13 && hour <= 21) ? 1.0 : 0.0;

        double range = cur.high - cur.low;
        double body = cur.close - cur.open;
        double uw = cur.high - Math.max(cur.open, cur.close);
        double lw = Math.min(cur.open, cur.close) - cur.low;
        f[29] = range > 0 ? body / range : 0.0;
        f[30] = range > 0 ? (cur.close - cur.low) / range : 0.5;
        f[31] = range > 0 ? uw / range : 0.0;
        f[32] = range > 0 ? lw / range : 0.0;

        if (i >= 3) f[33] = cur.close - bars.get(i-3).close;
        if (i >= 6) f[34] = cur.close - bars.get(i-6).close;

        if (i >= 12) {
            double sumC = 0, sumR = 0;
            for (int j = i-11; j <= i; j++) {
                sumC += bars.get(j).close;
                sumR += bars.get(j).high - bars.get(j).low;
            }
            double ma12 = sumC / 12;
            double meanR12 = sumR / 12;
            f[35] = cur.close - ma12;
            f[36] = meanR12 > 0 ? (cur.high - cur.low) / meanR12 : 1.0;
        }
    }

    // ============================================================
    // 11. PREDICT PROBABILITY
    // ============================================================
    private double predictProbability(double[] features) {
        double linear = INTERCEPT;
        for (int i = 0; i < Math.min(features.length, COEFFICIENTS.length); i++) {
            linear += COEFFICIENTS[i] * features[i];
        }
        return 1.0 / (1.0 + Math.exp(-linear));
    }

    // ============================================================
    // 13. DRAWING FUNCTIONS
    // ============================================================
    private static final int SIGNAL_GAP_FROM_PRICE = 22;
    private static final int SIGNAL_TRI_HALF_WIDTH = 8;
    private static final int SIGNAL_TRI_HEIGHT = 14;
    private static final int SIGNAL_LABEL_PADDING = 10;

    private void drawMarkers(Graphics g, boolean isBuy, double[] prices, IIndicatorDrawingSupport support) {
        if (prices == null) return;
        g.setColor(isBuy ? Color.GREEN : Color.RED);

        int first = support.getIndexOfFirstCandleOnScreen();
        int visible = support.getNumberOfCandlesOnScreen();
        int last = Math.min(first + visible, prices.length - 1);

        for (int i = Math.max(first, 0); i <= last; i++) {
            double price = prices[i];
            if (Double.isNaN(price)) continue;

            int x = (int) support.getMiddleOfCandle(i);
            int y = (int) support.getYForValue(price);

            int[] xPoints, yPoints;
            if (isBuy) {
                int tipY = y + SIGNAL_GAP_FROM_PRICE;
                int baseY = tipY + SIGNAL_TRI_HEIGHT;
                xPoints = new int[]{x, x - SIGNAL_TRI_HALF_WIDTH, x + SIGNAL_TRI_HALF_WIDTH};
                yPoints = new int[]{tipY, baseY, baseY};
            } else {
                int tipY = y - SIGNAL_GAP_FROM_PRICE;
                int baseY = tipY - SIGNAL_TRI_HEIGHT;
                xPoints = new int[]{x, x - SIGNAL_TRI_HALF_WIDTH, x + SIGNAL_TRI_HALF_WIDTH};
                yPoints = new int[]{tipY, baseY, baseY};
            }
            g.fillPolygon(xPoints, yPoints, 3);
        }
    }

    private void drawLabels(Graphics g, boolean isBuy, double[] prices, IIndicatorDrawingSupport support) {
        if (prices == null) return;
        String text = "Signal";
        Font font = new Font("SansSerif", Font.BOLD, 11);
        g.setFont(font);
        FontMetrics fm = g.getFontMetrics(font);
        int textWidth = fm.stringWidth(text);
        int textHeight = fm.getAscent();

        g.setColor(isBuy ? Color.GREEN : Color.RED);

        int first = support.getIndexOfFirstCandleOnScreen();
        int visible = support.getNumberOfCandlesOnScreen();
        int last = Math.min(first + visible, prices.length - 1);

        for (int i = Math.max(first, 0); i <= last; i++) {
            double price = prices[i];
            if (Double.isNaN(price)) continue;

            float x = support.getMiddleOfCandle(i);
            float y = support.getYForValue(price);

            int drawX = (int) (x - textWidth / 2.0);
            int drawY;
            if (isBuy) {
                int tipY = (int) (y + SIGNAL_GAP_FROM_PRICE);
                int baseY = tipY + SIGNAL_TRI_HEIGHT;
                drawY = baseY + SIGNAL_LABEL_PADDING + textHeight;
            } else {
                int tipY = (int) (y - SIGNAL_GAP_FROM_PRICE);
                int baseY = tipY - SIGNAL_TRI_HEIGHT;
                drawY = baseY - SIGNAL_LABEL_PADDING;
            }
            g.drawString(text, drawX, drawY);
        }
    }

    private void drawBacktestPanel(Graphics g, BacktestStats st, IBar[] candles, IIndicatorDrawingSupport support) {
        Font titleFont = new Font("SansSerif", Font.BOLD, 12);
        Font rowFont = new Font("SansSerif", Font.PLAIN, 11);

        String[] labels = {
                "Signals", "Win rate", "Start capital", "Final capital", "P/L", "Max stake",
                "Max win streak", "Max lose streak", "Max drawdown"
        };
        String[] fixedVals;
        String[] martVals;
        String title = "BACKTEST (no data)";

        if (st == null) {
            fixedVals = new String[]{"-", "-", "-", "-", "-", "-", "-", "-", "-"};
            martVals = fixedVals;
        } else {
            title = "BACKTEST — last " + st.totalSignals + " signals";
            fixedVals = new String[]{
                    String.valueOf(st.totalSignals),
                    String.format("%.2f%%", st.winRate),
                    String.format("%.2f", BT_STARTING_CAPITAL),
                    String.format("%.2f", st.fixedFinalCapital),
                    String.format("%+.2f", st.fixedPnl),
                    String.format("%.2f", BT_FIXED_STAKE),
                    String.valueOf(st.fixedMaxWinStreak),
                    String.valueOf(st.fixedMaxLoseStreak),
                    String.format("%.2f", st.fixedMaxDrawdown)
            };
            martVals = new String[]{
                    String.valueOf(st.totalSignals),
                    String.format("%.2f%%", st.winRate),
                    String.format("%.2f", BT_STARTING_CAPITAL),
                    String.format("%.2f", st.martFinalCapital),
                    String.format("%+.2f", st.martPnl),
                    String.format("%.2f", st.martMaxStakeUsed),
                    String.valueOf(st.martMaxWinStreak),
                    String.valueOf(st.martMaxLoseStreak),
                    String.format("%.2f", st.martMaxDrawdown)
            };
        }

        g.setFont(titleFont);
        FontMetrics tfm = g.getFontMetrics(titleFont);
        g.setFont(rowFont);
        FontMetrics rfm = g.getFontMetrics(rowFont);

        int colLabelW = 0, colFixedW = 0, colMartW = 0;
        for (String s : labels) colLabelW = Math.max(colLabelW, rfm.stringWidth(s));
        for (String s : fixedVals) colFixedW = Math.max(colFixedW, rfm.stringWidth(s));
        for (String s : martVals) colMartW = Math.max(colMartW, rfm.stringWidth(s));
        colFixedW = Math.max(colFixedW, rfm.stringWidth("Fixed"));
        colMartW = Math.max(colMartW, rfm.stringWidth("Martingale"));

        int pad = 8;
        int colGap = 18;
        int rowH = rfm.getHeight() + 3;
        int panelW = pad * 2 + colLabelW + colGap + colFixedW + colGap + colMartW;
        int panelH = pad * 2 + tfm.getHeight() + 6 + rowH * (labels.length + 1);

        // ---- หาตำแหน่งวาง โดยไม่พึ่ง g.getClipBounds() (บางแพลตฟอร์ม/สกินคืนค่า null) ----
        // ยึดตำแหน่งจาก "แท่งเทียนที่มองเห็นอยู่" แทน: ขวาสุด = หลังแท่งสุดท้ายที่มองเห็น,
        // บนสุด = เหนือราคาสูงสุดที่มองเห็นอยู่บนจอ
        int first = Math.max(0, support.getIndexOfFirstCandleOnScreen());
        int visible = support.getNumberOfCandlesOnScreen();
        int last = Math.min(first + visible - 1, candles.length - 1);
        if (last < first) last = Math.min(candles.length - 1, first);

        double maxHigh = -Double.MAX_VALUE;
        for (int i = first; i <= last && i >= 0 && i < candles.length; i++) {
            if (candles[i].getHigh() > maxHigh) maxHigh = candles[i].getHigh();
        }
        if (maxHigh == -Double.MAX_VALUE && candles.length > 0) {
            maxHigh = candles[candles.length - 1].getHigh();
        }

        float rightX = support.getMiddleOfCandle(Math.max(last, 0));
        int x = (int) (rightX - panelW - 10); // ให้ขอบขวาของกล่องอยู่ก่อนแท่งเทียนสุดท้ายที่มองเห็น (การันตีว่าอยู่ในจอ)
        int topY = (int) support.getYForValue(maxHigh);
        int y = Math.max(10, topY - 10);

        g.setColor(new Color(0, 0, 0, 200));
        g.fillRect(x, y, panelW, panelH);
        g.setColor(new Color(255, 255, 255, 80));
        g.drawRect(x, y, panelW, panelH);

        int curY = y + pad + tfm.getAscent();
        g.setFont(titleFont);
        g.setColor(Color.WHITE);
        g.drawString(title, x + pad, curY);
        curY += rowH + 6;

        int labelX = x + pad;
        int fixedX = labelX + colLabelW + colGap;
        int martX = fixedX + colFixedW + colGap;

        g.setFont(rowFont);
        g.setColor(Color.GREEN);
        g.drawString("Fixed", fixedX, curY);
        g.setColor(Color.ORANGE);
        g.drawString("Martingale", martX, curY);
        curY += rowH;

        for (int i = 0; i < labels.length; i++) {
            g.setColor(Color.LIGHT_GRAY);
            g.drawString(labels[i], labelX, curY);
            g.setColor(Color.WHITE);
            g.drawString(fixedVals[i], fixedX, curY);
            g.drawString(martVals[i], martX, curY);
            curY += rowH;
        }
    }

    // ============================================================
    // 14. HELPER CLASS สำหรับผลลัพธ์
    // ============================================================
    private static class SignalResult {
        double[] buyPrices;
        double[] sellPrices;
        int buyCount;
        int sellCount;
        List<BacktestSignal> backtestSignals;
        SignalResult(double[] buy, double[] sell, int b, int s, List<BacktestSignal> bt) {
            buyPrices = buy;
            sellPrices = sell;
            buyCount = b;
            sellCount = s;
            backtestSignals = bt;
        }
    }

    /** ผลลัพธ์ ชนะ/แพ้ ของสัญญาณหนึ่งตัว เทียบกับทิศทางแท่งถัดไป */
    private static class BacktestSignal {
        String signal; // "UP" หรือ "DOWN"
        boolean win;
        BacktestSignal(String signal, boolean win) {
            this.signal = signal;
            this.win = win;
        }
    }

    /** สรุปผล backtest ของการเดินเงินทั้ง 2 แบบ */
    private static class BacktestStats {
        int totalSignals, wins, losses;
        double winRate;

        double fixedFinalCapital, fixedPnl, fixedMaxDrawdown;
        int fixedMaxWinStreak, fixedMaxLoseStreak;

        double martFinalCapital, martPnl, martMaxDrawdown, martMaxStakeUsed;
        int martMaxWinStreak, martMaxLoseStreak;
    }

    // ============================================================
    // 15. METADATA GETTERS & SETTERS
    // ============================================================
    private IndicatorInfo indicatorInfo;
    private InputParameterInfo[] inputParameterInfos;
    private OptInputParameterInfo[] optInputParameterInfos;
    private OutputParameterInfo[] outputParameterInfos;

    @Override public int getLookback() { return TREND_WINDOW + 1; }
    @Override public int getLookforward() { return 0; }
    @Override public IndicatorInfo getIndicatorInfo() { return indicatorInfo; }
    @Override public InputParameterInfo getInputParameterInfo(int index) { return inputParameterInfos[index]; }
    @Override public OptInputParameterInfo getOptInputParameterInfo(int index) { return optInputParameterInfos[index]; }
    @Override public OutputParameterInfo getOutputParameterInfo(int index) { return outputParameterInfos[index]; }

    @Override
    public void setInputParameter(int index, Object array) {}

    @Override
    public void setOptInputParameter(int index, Object value) {
        switch (index) {
            case 0: threshold = (Double) value; break;
            case 1: patternFilter = (Boolean) value; break;
            case 2: bucketSize = (Double) value; break;
            case 3: imbalanceRatio = (Double) value; break;
            case 4: backtestEnabled = (Boolean) value; break;
            case 5: backtestLookback = (Integer) value; break;
        }
    }

    @Override
    public void setOutputParameter(int index, Object array) {
        outputs[index] = (double[]) array;
    }
}