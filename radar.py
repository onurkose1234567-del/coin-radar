import os
import time
import threading
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# AYARLAR
# =========================================================

BASE = "https://fapi.binance.com"

MIN_SCORE = float(os.getenv("MIN_SCORE", "90"))
INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

ALERT_COOLDOWN = 3600

last_alert = {}

bot_started = time.time()
last_scan_time = None
last_scan_duration = None
last_scan_count = 0
last_signal_count = 0

scan_running = False
scan_lock = threading.Lock()

telegram_offset = 0


# =========================================================
# BINANCE API
# =========================================================

def get(path, params=None):
    r = requests.get(
        BASE + path,
        params=params,
        timeout=15
    )

    r.raise_for_status()

    return r.json()


# =========================================================
# TUM AKTIF USDT PERPETUAL COINLER
# =========================================================

def symbols():
    info = get("/fapi/v1/exchangeInfo")

    result = []

    for x in info["symbols"]:

        if (
            x.get("quoteAsset") == "USDT"
            and x.get("contractType") == "PERPETUAL"
            and x.get("status") == "TRADING"
        ):
            result.append(x["symbol"])

    return result


# =========================================================
# MUMLAR
# =========================================================

def klines(sym, interval, limit=220):

    data = get(
        "/fapi/v1/klines",
        {
            "symbol": sym,
            "interval": interval,
            "limit": limit
        }
    )

    d = pd.DataFrame(
        data,
        columns=[
            "t",
            "o",
            "h",
            "l",
            "c",
            "v",
            "T",
            "q",
            "n",
            "tb",
            "tq",
            "x"
        ]
    )

    for col in [
        "o",
        "h",
        "l",
        "c",
        "v",
        "q",
        "tb"
    ]:
        d[col] = pd.to_numeric(
            d[col],
            errors="coerce"
        )

    return d


# =========================================================
# RSI
# =========================================================

def rsi(s, n=14):

    delta = s.diff()

    up = delta.clip(lower=0)

    down = -delta.clip(upper=0)

    avg_up = up.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    avg_down = down.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    rs = (
        avg_up
        /
        avg_down.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# ATR
# =========================================================

def atr(d, n=14):

    previous_close = d.c.shift()

    tr = pd.concat(
        [
            d.h - d.l,
            (d.h - previous_close).abs(),
            (d.l - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()


# =========================================================
# ADX
# =========================================================

def adx(d, n=14):

    up = d.h.diff()

    down = -d.l.diff()

    tr = atr(d, 1)

    smooth_tr = tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    plus = 100 * (
        up.where(
            (up > down) & (up > 0),
            0
        ).ewm(
            alpha=1 / n,
            adjust=False
        ).mean()
        /
        smooth_tr
    )

    minus = 100 * (
        down.where(
            (down > up) & (down > 0),
            0
        ).ewm(
            alpha=1 / n,
            adjust=False
        ).mean()
        /
        smooth_tr
    )

    dx = (
        100
        *
        (plus - minus).abs()
        /
        (plus + minus).replace(
            0,
            np.nan
        )
    )

    adx_value = dx.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    return (
        adx_value,
        plus,
        minus
    )


# =========================================================
# TEKNIK VERILER
# =========================================================

def features(d):

    c = d.c
    v = d.v

    e9 = c.ewm(
        span=9,
        adjust=False
    ).mean()

    e21 = c.ewm(
        span=21,
        adjust=False
    ).mean()

    e50 = c.ewm(
        span=50,
        adjust=False
    ).mean()

    e200 = c.ewm(
        span=200,
        adjust=False
    ).mean()

    rr = rsi(c)

    ema12 = c.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = c.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    ax, di_plus, di_minus = adx(d)

    volume_average = (
        v.rolling(20).mean()
    )

    volume_ratio = (
        v
        /
        volume_average.replace(
            0,
            np.nan
        )
    )

    middle = (
        c.rolling(20).mean()
    )

    std = (
        c.rolling(20).std()
    )

    lower = middle - 2 * std

    upper = middle + 2 * std

    bb_position = (
        (c - lower)
        /
        (upper - lower).replace(
            0,
            np.nan
        )
    )

    taker_buy_ratio = (
        d.tb
        /
        d.v.replace(
            0,
            np.nan
        )
    )

    previous_high = (
        d.h.shift(1)
        .rolling(20)
        .max()
    )

    previous_low = (
        d.l.shift(1)
        .rolling(20)
        .min()
    )

    return {
        "price": c.iloc[-1],

        "e9": e9.iloc[-1],
        "e21": e21.iloc[-1],
        "e50": e50.iloc[-1],
        "e200": e200.iloc[-1],

        "rsi": rr.iloc[-1],

        "mac": macd.iloc[-1],
        "sig": macd_signal.iloc[-1],

        "adx": ax.iloc[-1],

        "dip": di_plus.iloc[-1],
        "dim": di_minus.iloc[-1],

        "vr": volume_ratio.iloc[-1],

        "bb": bb_position.iloc[-1],

        "br": taker_buy_ratio.iloc[-1],

        "break_hi": (
            c.iloc[-1]
            >
            previous_high.iloc[-1]
        ),

        "break_lo": (
            c.iloc[-1]
            <
            previous_low.iloc[-1]
        )
    }


# =========================================================
# PUANLAMA
# =========================================================

def score(f, direction):

    total = 0

    reasons = []

    if direction == "LONG":

        checks = [

            (
                f["e9"]
                >
                f["e21"]
                >
                f["e50"],
                18,
                "EMA trend"
            ),

            (
                f["price"]
                >
                f["e200"],
                8,
                "EMA200"
            ),

            (
                f["mac"]
                >
                f["sig"],
                12,
                "MACD"
            ),

            (
                50
                <=
                f["rsi"]
                <=
                70,
                10,
                "RSI"
            ),

            (
                f["adx"] >= 22
                and
                f["dip"] > f["dim"],
                12,
                "ADX/DI"
            ),

            (
                f["vr"] >= 1.5,
                14,
                "Hacim"
            ),

            (
                f["br"] >= 0.52,
                8,
                "Taker Buy"
            ),

            (
                f["break_hi"],
                10,
                "Breakout"
            ),

            (
                0.45
                <=
                f["bb"]
                <=
                1.15,
                8,
                "Bollinger"
            )
        ]

    else:

        checks = [

            (
                f["e9"]
                <
                f["e21"]
                <
                f["e50"],
                18,
                "EMA trend"
            ),

            (
                f["price"]
                <
                f["e200"],
                8,
                "EMA200"
            ),

            (
                f["mac"]
                <
                f["sig"],
                12,
                "MACD"
            ),

            (
                30
                <=
                f["rsi"]
                <=
                50,
                10,
                "RSI"
            ),

            (
                f["adx"] >= 22
                and
                f["dim"] > f["dip"],
                12,
                "ADX/DI"
            ),

            (
                f["vr"] >= 1.5,
                14,
                "Hacim"
            ),

            (
                f["br"] <= 0.48,
                8,
                "Taker Sell"
            ),

            (
                f["break_lo"],
                10,
                "Breakdown"
            ),

            (
                -0.15
                <=
                f["bb"]
                <=
                0.55,
                8,
                "Bollinger"
            )
        ]

    for ok, weight, reason in checks:

        try:
            valid = bool(ok)

        except Exception:
            valid = False

        if valid:

            total += weight

            reasons.append(reason)

    return (
        min(total, 100),
        reasons
    )


# =========================================================
# COIN ANALIZI
# BTC VERISI DISARIDAN GELIYOR
# =========================================================

def analyze(sym, btc):

    f15 = features(
        klines(
            sym,
            "15m"
        )
    )

    f1h = features(
        klines(
            sym,
            "1h"
        )
    )

    output = []

    for direction in [
        "LONG",
        "SHORT"
    ]:

        score15, reasons15 = score(
            f15,
            direction
        )

        score1h, reasons1h = score(
            f1h,
            direction
        )

        if direction == "LONG":

            btc_ok = (
                btc["e9"]
                >=
                btc["e21"]
            )

        else:

            btc_ok = (
                btc["e9"]
                <=
                btc["e21"]
            )

        total = (
            0.58 * score15
            +
            0.32 * score1h
            +
            (10 if btc_ok else 0)
        )

        total = min(
            round(total, 1),
            100
        )

        if total >= MIN_SCORE:

            price = float(
                f15["price"]
            )

            # =============================================
            # SABIT YUZDE SL / TP
            #
            # SL  = %2
            # TP1 = %5
            # TP2 = %10
            # =============================================

            if direction == "LONG":

                sl = (
                    price * 0.98
                )

                tp1 = (
                    price * 1.05
                )

                tp2 = (
                    price * 1.10
                )

            else:

                sl = (
                    price * 1.02
                )

                tp1 = (
                    price * 0.95
                )

                tp2 = (
                    price * 0.90
                )

            reasons = sorted(
                set(
                    reasons15
                    +
                    reasons1h
                )
            )

            output.append(
                (
                    total,
                    direction,
                    price,
                    sl,
                    tp1,
                    tp2,
                    reasons,
                    f15["rsi"],
                    f15["vr"],
                    f15["adx"]
                )
            )

    return output


# =========================================================
# TELEGRAM MESAJ GONDERME
# =========================================================

def telegram(msg, chat_id=None):

    target = (
        chat_id
        if chat_id is not None
        else CHAT
    )

    if not TOKEN or not target:

        print(
            "Telegram ayarlari eksik",
            flush=True
        )

        return

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={
            "chat_id": target,
            "text": msg
        },
        timeout=15
    )

    r.raise_for_status()


# =========================================================
# CALISMA SURESI
# =========================================================

def uptime_text():

    seconds = int(
        time.time()
        -
        bot_started
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    return (
        f"{hours} saat "
        f"{minutes} dakika"
    )


# =========================================================
# DURUM
# =========================================================

def send_status(chat_id):

    if scan_running:

        state = (
            "🔎 Tarama yapılıyor"
        )

    else:

        state = (
            "🟢 Çalışıyor"
        )

    if last_scan_time:

        seconds_ago = int(
            time.time()
            -
            last_scan_time
        )

        last_text = (
            f"{seconds_ago} saniye önce"
        )

    else:

        last_text = (
            "Henüz tamamlanmadı"
        )

    if last_scan_duration is not None:

        duration_text = (
            f"{last_scan_duration:.1f} sn"
        )

    else:

        duration_text = "-"

    msg = (
        "📊 COIN RADAR V3 DURUMU\n\n"

        f"Durum: {state}\n"
        f"Çalışma süresi: {uptime_text()}\n\n"

        f"Minimum skor: {MIN_SCORE}/100\n"
        "Tarama: Tüm aktif USDT Perpetual\n"
        f"Tarama aralığı: {INTERVAL} sn\n\n"

        f"Son tarama: {last_text}\n"
        f"Son tarama süresi: {duration_text}\n"
        f"Taranan coin: {last_scan_count}\n"
        f"Yeni sinyal: {last_signal_count}\n\n"

        "Zaman dilimleri: 15m + 1h\n"
        "Yönler: LONG + SHORT\n\n"

        "🛑 SL: %2\n"
        "🎯 TP1: %5\n"
        "🎯 TP2: %10"
    )

    telegram(
        msg,
        chat_id
    )


# =========================================================
# PIYASA TARAMASI
# =========================================================

def scan_market(manual_chat=None):

    global scan_running
    global last_scan_time
    global last_scan_duration
    global last_scan_count
    global last_signal_count

    if not scan_lock.acquire(
        blocking=False
    ):

        if manual_chat:

            telegram(
                "⏳ Zaten bir tarama devam ediyor.",
                manual_chat
            )

        return

    scan_running = True

    start_time = time.time()

    try:

        coin_list = symbols()

        print(
            f"Tarama basladi: "
            f"{len(coin_list)} coin",
            flush=True
        )

        if manual_chat:

            telegram(
                "🔎 Manuel tarama başladı.\n\n"
                f"{len(coin_list)} aktif USDT "
                "Perpetual coin kontrol ediliyor.",
                manual_chat
            )

        # BTC VERISINI SADECE 1 KEZ CEK
        btc = features(
            klines(
                "BTCUSDT",
                "15m"
            )
        )

        signal_count = 0

        for sym in coin_list:

            try:

                results = analyze(
                    sym,
                    btc
                )

                for result in results:

                    (
                        sc,
                        direction,
                        price,
                        sl,
                        tp1,
                        tp2,
                        reasons,
                        current_rsi,
                        volume_ratio,
                        current_adx
                    ) = result

                    key = (
                        sym,
                        direction
                    )

                    now = time.time()

                    if (
                        now
                        -
                        last_alert.get(
                            key,
                            0
                        )
                        <
                        ALERT_COOLDOWN
                    ):

                        continue

                    if direction == "LONG":

                        direction_icon = "🟢"

                    else:

                        direction_icon = "🔴"

                    msg = (
                        f"🔥 {sym}\n\n"

                        f"{direction_icon} "
                        f"Yön: {direction}\n"

                        f"⭐ Radar skoru: "
                        f"{sc}/100\n\n"

                        f"💰 Giriş: "
                        f"{price:.8g}\n\n"

                        f"🛑 SL (%2): "
                        f"{sl:.8g}\n"

                        f"🎯 TP1 (%5): "
                        f"{tp1:.8g}\n"

                        f"🚀 TP2 (%10): "
                        f"{tp2:.8g}\n\n"

                        f"RSI: "
                        f"{current_rsi:.1f}\n"

                        f"Hacim: "
                        f"x{volume_ratio:.2f}\n"

                        f"ADX: "
                        f"{current_adx:.1f}\n\n"

                        f"Onaylar:\n"
                        f"{', '.join(reasons)}\n\n"

                        "⚠️ Radar skoru gerçek "
                        "kazanma olasılığı değildir."
                    )

                    telegram(msg)

                    last_alert[key] = now

                    signal_count += 1

            except Exception as e:

                print(
                    f"{sym} hata: {e}",
                    flush=True
                )

        duration = (
            time.time()
            -
            start_time
        )

        last_scan_time = time.time()

        last_scan_duration = duration

        last_scan_count = len(
            coin_list
        )

        last_signal_count = (
            signal_count
        )

        print(
            f"Tarama tamamlandi: "
            f"{len(coin_list)} coin | "
            f"{signal_count} sinyal | "
            f"{duration:.1f} saniye",
            flush=True
        )

        if manual_chat:

            telegram(
                "✅ Manuel tarama tamamlandı.\n\n"
                f"Taranan coin: "
                f"{len(coin_list)}\n"
                f"90+ yeni sinyal: "
                f"{signal_count}\n"
                f"Süre: "
                f"{duration:.1f} saniye",
                manual_chat
            )

    except Exception as e:

        print(
            "Tarama genel hatasi:",
            e,
            flush=True
        )

        if manual_chat:

            telegram(
                "❌ Tarama hatası:\n"
                f"{e}",
                manual_chat
            )

    finally:

        scan_running = False

        scan_lock.release()


# =========================================================
# OTOMATIK TARAMA
# =========================================================

def radar_loop():

    while True:

        scan_market()

        time.sleep(
            INTERVAL
        )


# =========================================================
# TELEGRAM KOMUTLARI
# =========================================================

def telegram_commands():

    global telegram_offset

    print(
        "Telegram komut sistemi aktif",
        flush=True
    )

    while True:

        try:

            r = requests.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={
                    "timeout": 25,
                    "offset": telegram_offset
                },
                timeout=35
            )

            r.raise_for_status()

            data = r.json()

            for update in data.get(
                "result",
                []
            ):

                telegram_offset = (
                    update["update_id"]
                    +
                    1
                )

                message = update.get(
                    "message",
                    {}
                )

                text = (
                    message.get(
                        "text",
                        ""
                    )
                    .strip()
                    .lower()
                )

                chat_id = (
                    message.get(
                        "chat",
                        {}
                    ).get("id")
                )

                if not chat_id:
                    continue

                # Sadece ayarlanan Telegram
                # sohbetinden komut kabul et
                if (
                    CHAT
                    and
                    str(chat_id)
                    !=
                    str(CHAT)
                ):

                    continue

                if text in [
                    "/start",
                    "/yardim",
                    "/help"
                ]:

                    telegram(
                        "🤖 COIN RADAR V3\n\n"

                        "Komutlar:\n\n"

                        "/durum - Bot durumunu göster\n"

                        "/tara - Hemen tüm piyasayı tara\n"

                        "/yardim - Komutları göster\n\n"

                        "Filtre:\n"
                        "Minimum 90/100\n\n"

                        "Risk seviyeleri:\n"
                        "SL %2\n"
                        "TP1 %5\n"
                        "TP2 %10",
                        chat_id
                    )

                elif text == "/durum":

                    send_status(
                        chat_id
                    )

                elif text == "/tara":

                    if scan_running:

                        telegram(
                            "⏳ Şu anda bir tarama "
                            "devam ediyor.\n\n"
                            "Tamamlanınca tekrar dene.",
                            chat_id
                        )

                    else:

                        threading.Thread(
                            target=scan_market,
                            args=(chat_id,),
                            daemon=True
                        ).start()

        except Exception as e:

            print(
                "Telegram komut hatasi:",
                e,
                flush=True
            )

            time.sleep(5)


# =========================================================
# BASLAT
# =========================================================

def run():

    print(
        "Coin Radar V3 started",
        flush=True
    )

    try:

        telegram(
            "🟢 COIN RADAR V3 AKTİF\n\n"

            "Tüm aktif Binance USDT "
            "Perpetual coinleri taranıyor.\n\n"

            f"Minimum skor: "
            f"{MIN_SCORE}/100\n"

            "Zaman: 15m + 1h\n"

            "Yön: LONG + SHORT\n\n"

            "🛑 SL: %2\n"
            "🎯 TP1: %5\n"
            "🚀 TP2: %10\n\n"

            "Komutlar:\n"
            "/durum\n"
            "/tara\n"
            "/yardim"
        )

    except Exception as e:

        print(
            "Telegram baslangic hatasi:",
            e,
            flush=True
        )

    radar_thread = threading.Thread(
        target=radar_loop,
        daemon=True
    )

    radar_thread.start()

    telegram_commands()


if __name__ == "__main__":
    run()
