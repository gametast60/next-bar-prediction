package com.dukascopy.strategies;

import com.dukascopy.api.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.TimeZone;

/**
 * TickExporter
 * ------------
 * ดึง tick ดิบ (bid, ask, bidVolume, askVolume) ของ instrument/ช่วงเวลาที่กำหนด
 * ออกมาเป็นไฟล์ CSV เพื่อเอาไปทำ feature engineering + เทรนโมเดลต่อใน Python
 *
 * เวอร์ชันนี้: พารามิเตอร์หลัก (ช่วงเวลา, instrument) เป็น @Configurable
 * ทั้งหมด แก้ได้จากหน้า "Set parameters" ตอนรัน strategy ใน JForex
 * โดยไม่ต้องเปิด editor แก้โค้ดทุกครั้ง
 *
 * วิธีใช้:
 * 1. รันเป็น Strategy ใน JForex (ไม่ใช่ Indicator)
 * 2. หน้าต่าง "Set parameters" จะโชว์ช่องให้กรอก: Use chart instrument,
 *    Fallback instrument, Use custom date range, Start date, End date,
 *    Days back
 * 3. ถ้า "Use custom date range" = true จะใช้ Start date / End date
 *    ถ้า false จะใช้ Days back (ย้อนหลัง N วันจากตอนนี้)
 * 4. ถ้า "Use chart instrument" = true จะดึง instrument จาก chart ที่
 *    active ล่าสุดตอนรัน (ผ่าน context.getLastActiveChart())
 */
public class TickExporter implements IStrategy {

    // ===== ปรับได้จากไฟล์ (ค่าคงที่ระดับ path เท่านั้น) =====
    private static final String OUTPUT_DIR = "C:/dukascopy_export/";
    private static final String DATE_PATTERN = "yyyy-MM-dd HH:mm:ss";
    // ========================================================

    // ===== พารามิเตอร์ที่ปรับได้จากหน้า UI ตอนรัน strategy =====

    @Configurable("Use chart instrument")
    public boolean useChartContext = true;

    @Configurable("Fallback instrument (ถ้าไม่ใช้ chart)")
    public Instrument fallbackInstrument = Instrument.EURUSD;

    @Configurable("Use custom date range (false = ใช้ Days back)")
    public boolean useCustomDateRange = true;

    @Configurable("Start date (yyyy-MM-dd HH:mm:ss, UTC)")
    public String startDateStr = "2025-01-01 00:00:00";

    @Configurable("End date (yyyy-MM-dd HH:mm:ss, UTC)")
    public String endDateStr = "2025-01-08 00:00:00";

    @Configurable("Days back (ใช้ตอน Use custom date range = false)")
    public int daysBack = 7;

    // ============================================================

    private IHistory history;
    private IConsole console;
    private IContext context;

    @Override
    public void onStart(IContext context) throws JFException {
        this.context = context;
        this.history = context.getHistory();
        this.console = context.getConsole();

        SimpleDateFormat sdf = new SimpleDateFormat(DATE_PATTERN);
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));

        try {
            Instrument instrument = fallbackInstrument;

            IChart chart = useChartContext ? context.getLastActiveChart() : null;
            if (chart != null) {
                instrument = chart.getInstrument();
                console.getOut().println("อ่าน instrument จาก chart ที่ active: " + instrument);
            } else {
                console.getOut().println("ไม่พบ chart ที่ active ใช้ค่า fallback แทน: " + instrument);
            }

            long fromTime;
            long toTime;

            if (useCustomDateRange) {
                try {
                    fromTime = sdf.parse(startDateStr).getTime();
                    toTime = sdf.parse(endDateStr).getTime();
                } catch (ParseException pe) {
                    console.getErr().println("รูปแบบวันที่ไม่ถูกต้อง ต้องเป็น '" + DATE_PATTERN +
                            "' (UTC) — ที่ได้รับ: startDateStr='" + startDateStr +
                            "', endDateStr='" + endDateStr + "'");
                    return;
                }
            } else {
                toTime = System.currentTimeMillis();
                fromTime = toTime - (daysBack * 24L * 60 * 60 * 1000);
            }

            // ---- clamp "to" ไม่ให้เกินเวลาของ tick ล่าสุดที่มีจริงของ instrument นี้ ----
            // แก้ error "to parameter can't be greater than time of the last tick"
            // หมายเหตุ: getTimeOfLastTick() อาจ return -1 ได้ตอนเพิ่งเริ่มรัน หรือช่วงตลาดปิด
            // (เสาร์-อาทิตย์) กรณีนี้ข้ามการ clamp ไปเลย ไม่งั้นจะเผลอตัดช่วงเวลาผิด
            long lastTickTime = history.getTimeOfLastTick(instrument);
            if (lastTickTime > 0 && toTime > lastTickTime) {
                console.getOut().println("ปรับ 'to' จาก " + sdf.format(toTime) +
                        " เป็นเวลา tick ล่าสุดที่มีจริง " + sdf.format(lastTickTime));
                toTime = lastTickTime;
            } else if (lastTickTime <= 0) {
                console.getOut().println("getTimeOfLastTick() คืนค่า " + lastTickTime +
                        " (อาจเพราะเพิ่งเริ่ม subscribe หรือตลาดปิด) ข้ามการ clamp ไปก่อน");
            }

            if (fromTime >= toTime) {
                console.getErr().println("ช่วงเวลาที่ขอไม่ถูกต้อง (from >= to หลัง clamp แล้ว) " +
                        "from=" + sdf.format(fromTime) + ", to=" + sdf.format(toTime) +
                        " ลองปรับ Start date/End date หรือ Days back ใหม่");
                return;
            }

            String outputPath = OUTPUT_DIR + instrument.name() + "_ticks.csv";
            java.io.File outFile = new java.io.File(outputPath);
            outFile.getParentFile().mkdirs();

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outFile, false))) {
                writer.write("timestamp,bid,ask,bidVolume,askVolume");
                writer.newLine();

                long dayMillis = 24L * 60 * 60 * 1000;
                long chunkStart = fromTime;
                long totalTicks = 0;

                while (chunkStart < toTime) {
                    long chunkEnd = Math.min(chunkStart + dayMillis, toTime);

                    List<ITick> ticks = history.getTicks(instrument, chunkStart, chunkEnd);
                    if (ticks != null) {
                        for (ITick t : ticks) {
                            writer.write(t.getTime() + "," + t.getBid() + "," + t.getAsk() + ","
                                    + t.getBidVolume() + "," + t.getAskVolume());
                            writer.newLine();
                        }
                        totalTicks += ticks.size();
                    }

                    console.getOut().println("Exported chunk " + sdf.format(chunkStart) +
                            " -> " + sdf.format(chunkEnd) + " (" + (ticks == null ? 0 : ticks.size()) + " ticks)");

                    chunkStart = chunkEnd;
                }

                console.getOut().println("DONE. Total ticks exported: " + totalTicks +
                        " -> " + outputPath);
            }
        } catch (Exception e) {
            console.getErr().println("TickExporter error: " + e.getMessage());
        } finally {
            context.stop();
        }
    }

    @Override public void onTick(Instrument instrument, ITick tick) {}
    @Override public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) {}
    @Override public void onMessage(IMessage message) {}
    @Override public void onAccount(IAccount account) {}
    @Override public void onStop() throws JFException {
        console.getOut().println("TickExporter stopped.");
    }
}