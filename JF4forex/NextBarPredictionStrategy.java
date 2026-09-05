package com.dukascopy.strategies;

import com.dukascopy.api.*;
import java.io.*;
import java.text.*;
import java.util.*;

public class NextBarPredictionStrategy implements IStrategy {

    // ============================================================
    // 1. CONSTANTS
    // ============================================================
    
    private static final String DATE_PATTERN = "yyyy-MM-dd HH:mm:ss";
    private static final String OUTPUT_DIR = "C:/jforex_backtest/";
    private static final int TIMEFRAME_MINUTES = 5;
    private static final long PERIOD_MS = 5 * 60 * 1000;
    private static final double TRAIN_RATIO = 0.80;
    private static final int MIN_STACKED_ROWS = 3;
    private static final int TREND_WINDOW = 13;  // ต้องมีอย่างน้อย 12 แท่งก่อนหน้าสำหรับ MA12
    // ============================================================
    // 2. @Configurable PARAMETERS
    // ============================================================
    
    @Configurable("Instrument")
    public Instrument instrument = Instrument.XAUUSD;
    
    @Configurable("Start date (yyyy-MM-dd HH:mm:ss, UTC)")
    public String startDateStr = "2026-04-01 00:00:00";
    
    @Configurable("End date (yyyy-MM-dd HH:mm:ss, UTC)")
    public String endDateStr = "2026-04-30 23:59:59";
    
    @Configurable("Probability threshold")
    public double probabilityThreshold = 0.55;
    
    @Configurable("Use candle pattern filter")
    public boolean usePatternFilter = true;
    
    @Configurable("Starting balance (บาท)")
    public double startingBalance = 2000.0;
    
    @Configurable("Stake per trade (บาท)")
    public double stakePerTrade = 100.0;
    
    @Configurable("Payout rate (0.0-1.0)")
    public double payoutRate = 0.80;
    
    @Configurable("Use Martingale")
    public boolean useMartingale = true;
    
    @Configurable("Martingale multiplier")
    public double martingaleMultiplier = 2.0;
    
    // ============================================================
    // 3. MODEL COEFFICIENTS
    // ใส่ค่าที่ export จาก Python ตรงนี้!!!
    // ============================================================
    
    // Exported from LogisticRegression trained on the first 80% of
    // data/XAUUSD_ticks.csv with the current Python feature order.
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
    
    private static final int N_FEATURES = 37;
    
    // ============================================================
    // 4. INNER CLASSES
    // ============================================================
    
    private static class BarData {
        long timestamp;
        double open, high, low, close;
        double volume;
        double buyVolume, sellVolume;
        int priceLevels;
        double pocPrice;
        int buyImbalanceRows, sellImbalanceRows;
        int maxBuyStackedRun, maxSellStackedRun;
        double[] features;
        double probability;
        String signal;
        double nextClose;
        boolean isCorrect;
        
        BarData() {
            features = new double[N_FEATURES];
        }
    }
    
    private static class SignalResult {
        long timestamp;
        String signal;
        double probability;
        double entryPrice;
        double exitPrice;
        boolean isCorrect;
    }
    
    private static class BacktestResult {
        int totalBars;
        int totalSignals;
        int correctSignals;
        int wrongSignals;
        double accuracyOnSignals;
        double coverage;
        int totalBets;
        int correctBets;
        int wrongBets;
        double accuracyOnBets;
        double finalBalance;
        double netProfit;
        double maxDrawdown;
        double maxStake;
        int maxLossStreak;
        int maxWinStreak;
        boolean isStopped;
    }
    
    // ============================================================
    // 5. STATE
    // ============================================================
    
    private IContext context;
    private IHistory history;
    private IConsole console;
    private SimpleDateFormat sdf;
    
    private List<BarData> bars = new ArrayList<>();
    private List<SignalResult> signals = new ArrayList<>();
    private int trainEndIdx;
    private double globalBucketSize;
    private double globalImbalanceRatio;
    
    // ============================================================
    // 6. IStrategy METHODS
    // ============================================================
    
    @Override
    public void onStart(IContext context) throws JFException {
        this.context = context;
        this.history = context.getHistory();
        this.console = context.getConsole();

        sdf = new SimpleDateFormat(DATE_PATTERN);
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));

        console.getOut().println("=================================================");
        console.getOut().println("  NEXT BAR PREDICTION STRATEGY");
        console.getOut().println("  Instrument: " + instrument);
        console.getOut().println("  Start: " + startDateStr);
        console.getOut().println("  End: " + endDateStr);
        console.getOut().println("  Threshold: " + probabilityThreshold);
        console.getOut().println("  Pattern filter: " + usePatternFilter);
        console.getOut().println("  Starting balance: " + String.format("%.2f", startingBalance));
        console.getOut().println("  Stake: " + String.format("%.2f", stakePerTrade));
        console.getOut().println("  Payout: " + String.format("%.0f%%", payoutRate * 100));
        console.getOut().println("  Martingale: " + useMartingale);
        console.getOut().println("=================================================");
        
        long startTime = parseTimestamp(startDateStr);
        long endTime = parseTimestamp(endDateStr);
        long now = System.currentTimeMillis();
        
        if (endTime > now) endTime = now;
        endTime = (endTime / PERIOD_MS) * PERIOD_MS;
        
        if (startTime >= endTime) {
            console.getErr().println("Invalid date range.");
            context.stop();
            return;
        }
        
        // Load bars
        List<IBar> historicalBars = history.getBars(
            instrument, Period.FIVE_MINS, OfferSide.ASK, startTime, endTime
        );
        
        if (historicalBars == null || historicalBars.isEmpty()) {
            console.getErr().println("No bars found!");
            context.stop();
            return;
        }
        
        console.getOut().println("Loaded " + historicalBars.size() + " bars.");
        
        // ============================================================
        // ESTIMATE GLOBAL BUCKET_SIZE (เหมือน Python)
        // ============================================================
        List<Double> ranges = new ArrayList<>();
        for (IBar bar : historicalBars) {
            double r = bar.getHigh() - bar.getLow();
            if (r > 0) ranges.add(r);
        }
        Collections.sort(ranges);
        double medianRange = ranges.get(ranges.size() / 2);
        int targetRows = 20;
        // Match Python auto-tuned bucket for the XAUUSD dataset used by CMD.
        globalBucketSize = instrument == Instrument.XAUUSD ? 0.2 : niceNumber(medianRange / targetRows);
        
        if (instrument == Instrument.XAUUSD) {
            globalImbalanceRatio = 2.01;
        } else {
            globalImbalanceRatio = 3.0;
        }
        
        console.getOut().println("Global bucket_size: " + globalBucketSize);
        console.getOut().println("Global imbalance_ratio: " + globalImbalanceRatio);
        
        // ============================================================
        // BUILD FOOTPRINT BARS
        // ============================================================
        console.getOut().println("Building footprint bars...");
        for (IBar bar : historicalBars) {
            long barTime = bar.getTime();
            if (barTime < startTime || barTime > endTime) continue;
            BarData data = createBarData(bar, barTime);
            bars.add(data);
        }
        console.getOut().println("Total bars: " + bars.size());
        
        // ============================================================
        // CALCULATE FEATURES
        // ============================================================
        console.getOut().println("Calculating features...");
        updateFeatures();

        // Python drops rows without a complete 12-bar context before the
        // 80/20 split. Keep the final bar as the expiry bar, but remove the
        // first 11 warm-up rows from the training/test population.
        if (bars.size() > TREND_WINDOW - 1) {
            bars = new ArrayList<>(bars.subList(TREND_WINDOW - 1, bars.size()));
        }
        console.getOut().println("Bars after context warm-up: " + bars.size());
        
        // ============================================================
        // TRAIN/TEST SPLIT (80%/20%)
        // ============================================================
        int totalBars = bars.size();
        // The last bar is used only as the next-bar expiry for the final
        // prediction, matching Python's shift(-1) label construction.
        trainEndIdx = (int)((totalBars - 1) * TRAIN_RATIO);
        console.getOut().println("Train/Test split: " + trainEndIdx + " / " + (totalBars - trainEndIdx));
        
        // ============================================================
        // GENERATE SIGNALS (เฉพาะ Test Set)
        // ============================================================
        console.getOut().println("Generating signals on test set...");
        for (int i = trainEndIdx; i < totalBars - 1; i++) {
            BarData current = bars.get(i);
            BarData next = bars.get(i + 1);
            
            double prob = predictProbability(current.features);
            current.probability = prob;
            
            String signal = "NO_SIGNAL";
            if (prob >= probabilityThreshold) signal = "UP";
            else if (prob <= 1.0 - probabilityThreshold) signal = "DOWN";
            
            // Pattern filter
            if (usePatternFilter && !"NO_SIGNAL".equals(signal)) {
                double body = current.close - current.open;
                if ("UP".equals(signal) && body <= 0) signal = "NO_SIGNAL";
                else if ("DOWN".equals(signal) && body >= 0) signal = "NO_SIGNAL";
            }
            
            current.signal = signal;
            
            if (!"NO_SIGNAL".equals(signal)) {
                double nextClose = next.close;
                current.nextClose = nextClose;
                boolean correct = false;
                if ("UP".equals(signal) && nextClose > current.close) correct = true;
                else if ("DOWN".equals(signal) && nextClose <= current.close) correct = true;
                current.isCorrect = correct;
                
                SignalResult r = new SignalResult();
                r.timestamp = current.timestamp;
                r.signal = signal;
                r.probability = prob;
                r.entryPrice = current.close;
                r.exitPrice = nextClose;
                r.isCorrect = correct;
                signals.add(r);
            }
        }
        
        console.getOut().println("Total signals: " + signals.size());
        
        // ============================================================
        // BACKTEST
        // ============================================================
        console.getOut().println("\n=== Fixed Stake ===");
        BacktestResult fixed = runBacktest(false);
        printResult("Fixed Stake", fixed);
        
        if (useMartingale) {
            console.getOut().println("\n=== Martingale ===");
            BacktestResult martingale = runBacktest(true);
            printResult("Martingale", martingale);
            exportResults(fixed, martingale);
        } else {
            exportResults(fixed, null);
        }
        
        context.stop();
    }
    
    @Override public void onStop() throws JFException {
        console.getOut().println("Strategy stopped.");
    }
    @Override public void onTick(Instrument instrument, ITick tick) throws JFException {}
    @Override public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) throws JFException {}
    @Override public void onMessage(IMessage message) throws JFException {}
    @Override public void onAccount(IAccount account) throws JFException {}
    
    // ============================================================
    // 7. HELPER METHODS
    // ============================================================
    
    private BarData createBarData(IBar bar, long barTime) throws JFException {
        BarData data = new BarData();
        data.timestamp = barTime;
        data.open = bar.getOpen();
        data.high = bar.getHigh();
        data.low = bar.getLow();
        data.close = bar.getClose();
        data.volume = bar.getVolume();
        
        // IBar time is the opening time. Python builds each bar from ticks
        // in [barTime, barTime + 5 minutes), using mid=(bid+ask)/2.
        long barStart = barTime;
        long barEnd = barTime + PERIOD_MS;
        List<ITick> ticks = history.getTicks(instrument, barStart, barEnd);
        if (ticks != null && !ticks.isEmpty()) {
            data.open = Double.POSITIVE_INFINITY;
            data.high = Double.NEGATIVE_INFINITY;
            data.low = Double.POSITIVE_INFINITY;
            data.close = Double.NaN;
            for (ITick tick : ticks) {
                double mid = (tick.getBid() + tick.getAsk()) / 2.0;
                data.open = Math.min(data.open, mid);
                data.high = Math.max(data.high, mid);
                data.low = Math.min(data.low, mid);
                data.close = mid;
            }
            calculateFootprint(data, ticks);
        }
        return data;
    }
    
    private void calculateFootprint(BarData data, List<ITick> ticks) {
        double totalBidVol = 0, totalAskVol = 0;
        for (ITick t : ticks) {
            totalBidVol += t.getBidVolume();
            totalAskVol += t.getAskVolume();
        }
        boolean useTickCountFallback = (totalBidVol + totalAskVol) <= 0;
        
        Map<Long, Double> buyVol = new HashMap<>();
        Map<Long, Double> sellVol = new HashMap<>();
        Set<Long> keys = new HashSet<>();
        
        double prevMid = 0;
        int direction = 1;
        
        for (int i = 0; i < ticks.size(); i++) {
            ITick t = ticks.get(i);
            double mid = (t.getBid() + t.getAsk()) / 2.0;
            double vol = useTickCountFallback ? 1.0 : (t.getBidVolume() + t.getAskVolume()) / 2.0;
            
            if (i > 0) {
                double diff = mid - prevMid;
                if (diff > 0) direction = 1;
                else if (diff < 0) direction = -1;
            }
            
            long key = Math.round(mid / globalBucketSize);
            keys.add(key);
            
            if (direction > 0) {
                buyVol.put(key, buyVol.getOrDefault(key, 0.0) + vol);
                data.buyVolume += vol;
            } else {
                sellVol.put(key, sellVol.getOrDefault(key, 0.0) + vol);
                data.sellVolume += vol;
            }
            prevMid = mid;
        }
        
        data.priceLevels = keys.size();
        
        Map<Long, Double> totalVol = new HashMap<>();
        for (Long k : keys) {
            totalVol.put(k, buyVol.getOrDefault(k, 0.0) + sellVol.getOrDefault(k, 0.0));
        }
        
        double maxVol = 0;
        long pocKey = 0;
        for (Map.Entry<Long, Double> e : totalVol.entrySet()) {
            if (e.getValue() > maxVol) {
                maxVol = e.getValue();
                pocKey = e.getKey();
            }
        }
        data.pocPrice = pocKey * globalBucketSize;
        
        List<Long> sortedKeys = new ArrayList<>(keys);
        Collections.sort(sortedKeys);
        
        int buyImb = 0, sellImb = 0;
        int maxBuyRun = 0, maxSellRun = 0;
        int curBuyRun = 0, curSellRun = 0;
        
        for (int i = 0; i < sortedKeys.size(); i++) {
            Long k = sortedKeys.get(i);
            double buy = buyVol.getOrDefault(k, 0.0);
            double sell = sellVol.getOrDefault(k, 0.0);
            
            boolean isBuyImb = false;
            boolean isSellImb = false;
            
            if (i > 0) {
                Long prevKey = sortedKeys.get(i - 1);
                double prevSell = sellVol.getOrDefault(prevKey, 0.0);
                if (prevSell > 0 && buy >= prevSell * globalImbalanceRatio) {
                    isBuyImb = true;
                }
            }
            
            if (i < sortedKeys.size() - 1) {
                Long nextKey = sortedKeys.get(i + 1);
                double nextBuy = buyVol.getOrDefault(nextKey, 0.0);
                if (nextBuy > 0 && sell >= nextBuy * globalImbalanceRatio) {
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
    }
    
    private double niceNumber(double x) {
        if (x <= 0) return 0.0001;
        double exp = Math.floor(Math.log10(x));
        double frac = x / Math.pow(10, exp);
        if (frac < 1.5) return 1.0 * Math.pow(10, exp);
        else if (frac < 3.0) return 2.0 * Math.pow(10, exp);
        else if (frac < 7.0) return 5.0 * Math.pow(10, exp);
        else return 10.0 * Math.pow(10, exp);
    }
    
    private void updateFeatures() {
        if (bars.size() < 13) return;
        
        for (int i = 12; i < bars.size(); i++) {
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
    }
    
    private double predictProbability(double[] features) {
        double linear = INTERCEPT;
        for (int i = 0; i < Math.min(features.length, COEFFICIENTS.length); i++) {
            linear += COEFFICIENTS[i] * features[i];
        }
        return 1.0 / (1.0 + Math.exp(-linear));
    }
    
    private BacktestResult runBacktest(boolean isMartingale) {
        double balance = startingBalance;
        List<Double> equity = new ArrayList<>();
        equity.add(balance);
        
        int bets = 0, correctBets = 0, wrongBets = 0;
        double stake = stakePerTrade;
        int lossStreak = 0, winStreak = 0;
        int maxLossStreak = 0, maxWinStreak = 0;
        double maxStake = 0;
        boolean stopped = false;
        
        for (SignalResult s : signals) {
            if (isMartingale) {
                if (balance < stake) { stopped = true; break; }
                bets++;
                maxStake = Math.max(maxStake, stake);
                if (s.isCorrect) {
                    balance += stake * payoutRate;
                    correctBets++;
                    lossStreak = 0;
                    winStreak++;
                    maxWinStreak = Math.max(maxWinStreak, winStreak);
                    stake = stakePerTrade;
                } else {
                    balance -= stake;
                    wrongBets++;
                    winStreak = 0;
                    lossStreak++;
                    maxLossStreak = Math.max(maxLossStreak, lossStreak);
                    stake *= martingaleMultiplier;
                }
            } else {
                if (balance < stakePerTrade) { stopped = true; break; }
                bets++;
                maxStake = Math.max(maxStake, stakePerTrade);
                if (s.isCorrect) {
                    balance += stakePerTrade * payoutRate;
                    correctBets++;
                    lossStreak = 0;
                    winStreak++;
                    maxWinStreak = Math.max(maxWinStreak, winStreak);
                } else {
                    balance -= stakePerTrade;
                    wrongBets++;
                    winStreak = 0;
                    lossStreak++;
                    maxLossStreak = Math.max(maxLossStreak, lossStreak);
                }
            }
            equity.add(balance);
        }
        
        double peak = startingBalance;
        double maxDD = 0;
        for (double e : equity) {
            if (e > peak) peak = e;
            double dd = e - peak;
            if (dd < maxDD) maxDD = dd;
        }
        
        BacktestResult r = new BacktestResult();
        r.totalBars = bars.size() - trainEndIdx - 1;
        r.totalSignals = signals.size();
        r.correctSignals = (int) signals.stream().filter(s -> s.isCorrect).count();
        r.wrongSignals = r.totalSignals - r.correctSignals;
        r.accuracyOnSignals = r.totalSignals > 0 ? (double) r.correctSignals / r.totalSignals : 0;
        r.coverage = r.totalBars > 0 ? (double) r.totalSignals / r.totalBars : 0;
        r.totalBets = bets;
        r.correctBets = correctBets;
        r.wrongBets = wrongBets;
        r.accuracyOnBets = bets > 0 ? (double) correctBets / bets : 0;
        r.finalBalance = balance;
        r.netProfit = balance - startingBalance;
        r.maxDrawdown = maxDD;
        r.maxStake = maxStake;
        r.maxLossStreak = maxLossStreak;
        r.maxWinStreak = maxWinStreak;
        r.isStopped = stopped;
        
        return r;
    }
    
    private void printResult(String name, BacktestResult r) {
        console.getOut().println("[" + name + "]");
        console.getOut().println("  สัญญาณทั้งหมด: " + r.totalSignals + " ครั้ง");
        console.getOut().println("  ทายถูก / ผิด: " + r.correctSignals + " / " + r.wrongSignals);
        console.getOut().println("  Win rate: " + String.format("%.2f%%", r.accuracyOnSignals * 100));
        console.getOut().println("  Coverage: " + String.format("%.2f%%", r.coverage * 100));
        console.getOut().println("  เข้าเงินจริง: " + r.totalBets + " ไม้");
        console.getOut().println("  ทายถูก / ผิด (ไม้): " + r.correctBets + " / " + r.wrongBets);
        console.getOut().println("  Win rate (ไม้): " + String.format("%.2f%%", r.accuracyOnBets * 100));
        console.getOut().println("  ทุนเริ่มต้น: " + String.format("%.2f", startingBalance));
        console.getOut().println("  ทุนสุดท้าย: " + String.format("%.2f", r.finalBalance));
        console.getOut().println("  กำไร/ขาดทุน: " + String.format("%+.2f", r.netProfit));
        console.getOut().println("  Max Drawdown: " + String.format("%.2f", r.maxDrawdown));
        console.getOut().println("  แพ้ติดกันสูงสุด: " + r.maxLossStreak);
        console.getOut().println("  ชนะติดกันสูงสุด: " + r.maxWinStreak);
        console.getOut().println("  สถานะ: " + (r.isStopped ? "พอร์ตแตก" : "ทดสอบครบ"));
    }
    
    private void exportResults(BacktestResult fixed, BacktestResult martingale) {
        try {
            String filename = instrument.name() + "_backtest_" + System.currentTimeMillis() + ".txt";
            String path = OUTPUT_DIR + filename;
            File f = new File(path);
            f.getParentFile().mkdirs();
            
            try (BufferedWriter w = new BufferedWriter(new FileWriter(f, false))) {
                w.write("=== NEXT BAR PREDICTION BACKTEST ===\n");
                w.write("Instrument: " + instrument + "\n");
                w.write("Timeframe: 5 minutes\n");
                w.write("Start: " + startDateStr + "\n");
                w.write("End: " + endDateStr + "\n");
                w.write("Threshold: " + probabilityThreshold + "\n");
                w.write("Pattern filter: " + usePatternFilter + "\n");
                w.write("Stake: " + stakePerTrade + "\n");
                w.write("Payout: " + (payoutRate * 100) + "%\n");
                w.write("Martingale: " + useMartingale + "\n\n");
                
                w.write("=== FIXED STAKE ===\n");
                writeResult(w, fixed);
                
                if (useMartingale && martingale != null) {
                    w.write("\n=== MARTINGALE ===\n");
                    writeResult(w, martingale);
                }
                
                w.write("\n--- SIGNAL DETAILS ---\n");
                w.write("Timestamp,Signal,Probability,EntryPrice,ExitPrice,Correct\n");
                for (SignalResult s : signals) {
                    String ts = sdf.format(new Date(s.timestamp));
                    w.write(String.format("%s,%s,%.4f,%.4f,%.4f,%s\n",
                        ts, s.signal, s.probability, s.entryPrice, s.exitPrice, s.isCorrect));
                }
            }
            console.getOut().println("Exported: " + path);
        } catch (IOException e) {
            console.getErr().println("Export error: " + e.getMessage());
        }
    }
    
    private void writeResult(BufferedWriter w, BacktestResult r) throws IOException {
        w.write("Total signals: " + r.totalSignals + "\n");
        w.write("Correct/Wrong: " + r.correctSignals + "/" + r.wrongSignals + "\n");
        w.write("Win rate: " + String.format("%.2f%%", r.accuracyOnSignals * 100) + "\n");
        w.write("Coverage: " + String.format("%.2f%%", r.coverage * 100) + "\n");
        w.write("Total bets: " + r.totalBets + "\n");
        w.write("Correct/Wrong bets: " + r.correctBets + "/" + r.wrongBets + "\n");
        w.write("Win rate (bets): " + String.format("%.2f%%", r.accuracyOnBets * 100) + "\n");
        w.write("Starting balance: " + String.format("%.2f", startingBalance) + "\n");
        w.write("Final balance: " + String.format("%.2f", r.finalBalance) + "\n");
        w.write("Net profit: " + String.format("%.2f", r.netProfit) + "\n");
        w.write("Max drawdown: " + String.format("%.2f", r.maxDrawdown) + "\n");
        w.write("Max stake: " + String.format("%.2f", r.maxStake) + "\n");
        w.write("Max loss streak: " + r.maxLossStreak + "\n");
        w.write("Max win streak: " + r.maxWinStreak + "\n");
        w.write("Stopped: " + r.isStopped + "\n");
    }
    
    private long parseTimestamp(String ts) {
        try {
            return sdf.parse(ts).getTime();
        } catch (ParseException e) {
            return System.currentTimeMillis();
        }
    }
}