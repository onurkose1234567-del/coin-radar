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

MIN_SCORE = float(os.getenv("MIN_SCORE", "80"))
MIN_VOLUME = 10_000_000          # Minimum 24h Futures hacmi
INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

MAX_SL_PERCENT = 2.0             # SL maksimum %2
MIN_TP1_RR = 1.0                 # TP1 en az 1R
MIN_TP2_RR = 2.0                 # TP2 en az 2R

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
        timeout=20
    )

    r.raise_for_status()

    return r.json()


# =========================================================
# COIN LISTESI
# SADECE 24H HACMI 10M+ OLANLAR
# =========================================================

def symbols():

    info = get("/fapi/v1/exchangeInfo")

    tickers = get("/fapi/v1/ticker/24hr")

    active = set()

    for x in info["symbols"]:

        if (
            x.get("quoteAsset") == "USDT"
            and x.get("contractType") == "PERPETUAL"
            and x.get("status") == "TRADING"
        ):
            active.add(x["symbol"])

    result = []

    for ticker in tickers:

        sym = ticker.get("symbol")

        if sym not in active:
            continue

        try:
            volume = float(
                ticker.get("quoteVolume", 0)
            )
        except:
            volume = 0

        if volume >= MIN_VOLUME:

            result.append(
                (
                    sym,
                    volume
                )
            )

    result.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return result


# =========================================================
# KLINE
# =========================================================

def klines(symbol, interval, limit=220):

    data = get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    df = pd.DataFrame(
        data,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy",
            "taker_quote",
            "ignore"
        ]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# =========================================================
# RSI
# =========================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# =========================================================
# ADX
# =========================================================

def calculate_adx(df, period=14):

    up = df["high"].diff()

    down = -df["low"].diff()

    plus_dm = up.where(
        (up > down) & (up > 0),
        0
    )

    minus_dm = down.where(
        (down > up) & (down > 0),
        0
    )

    atr = calculate_atr(
        df,
        period
    )

    plus_di = (
        100
        *
        plus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr.replace(0, np.nan)
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        atr.replace(0, np.nan)
    )

    dx = (
        100
        *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return (
        adx,
        plus_di,
        minus_di
    )


# =========================================================
# TEKNIK VERILER
# =========================================================

def features(df):

    close = df["close"]

    volume = df["volume"]

    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False
    ).mean()

    rsi = calculate_rsi(close)

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    adx, plus_di, minus_di = calculate_adx(df)

    volume_average = volume.rolling(
        20
    ).mean()

    volume_ratio = (
        volume
        /
        volume_average.replace(
            0,
            np.nan
        )
    )

    middle = close.rolling(20).mean()

    std = close.rolling(20).std()

    lower = middle - (2 * std)

    upper = middle + (2 * std)

    bollinger = (
        (close - lower)
        /
        (upper - lower).replace(
            0,
            np.nan
        )
    )

    taker_ratio = (
        df["taker_buy"]
        /
        volume.replace(
            0,
            np.nan
        )
    )

    previous_high = (
        df["high"]
        .shift(1)
        .rolling(20)
        .max()
    )

    previous_low = (
        df["low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return {

        "price": close.iloc[-1],

        "ema9": ema9.iloc[-1],
        "ema21": ema21.iloc[-1],
        "ema50": ema50.iloc[-1],
        "ema200": ema200.iloc[-1],

        "rsi": rsi.iloc[-1],

        "macd": macd.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],

        "adx": adx.iloc[-1],

        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],

        "volume_ratio": volume_ratio.iloc[-1],

        "bollinger": bollinger.iloc[-1],

        "taker_ratio": taker_ratio.iloc[-1],

        "breakout":
            close.iloc[-1]
            >
            previous_high.iloc[-1],

        "breakdown":
            close.iloc[-1]
            <
            previous_low.iloc[-1]
    }


# =========================================================
# DESTEK / DIRENC SWING NOKTALARI
# =========================================================

def swing_levels(df, window=3):

    highs = df["high"].values

    lows = df["low"].values

    resistances = []

    supports = []

    # Son açık/aktif mumu kullanma
    end = len(df) - 1

    for i in range(
        window,
        end - window
    ):

        high_area = highs[
            i - window:
            i + window + 1
        ]

        low_area = lows[
            i - window:
            i + window + 1
        ]

        if highs[i] >= np.max(high_area):

            resistances.append(
                float(highs[i])
            )

        if lows[i] <= np.min(low_area):

            supports.append(
                float(lows[i])
            )

    return (
        supports,
        resistances
    )


# =========================================================
# BENZER SEVIYELERI BIRLESTIR
# =========================================================

def merge_levels(levels, tolerance=0.003):

    if not levels:
        return []

    levels = sorted(levels)

    groups = []

    current = [
        levels[0]
    ]

    for level in levels[1:]:

        average = sum(current) / len(current)

        if (
            abs(level - average)
            /
            average
            <=
            tolerance
        ):

            current.append(level)

        else:

            groups.append(
                sum(current)
                /
                len(current)
            )

            current = [
                level
            ]

    groups.append(
        sum(current)
        /
        len(current)
    )

    return groups


# =========================================================
# 15M + 1H DESTEK DIRENC
# =========================================================

def support_resistance(df15, df1h):

    s15, r15 = swing_levels(
        df15,
        3
    )

    s1h, r1h = swing_levels(
        df1h,
        2
    )

    supports = merge_levels(
        s15 + s1h
    )

    resistances = merge_levels(
        r15 + r1h
    )

    return (
        supports,
        resistances
    )


# =========================================================
# SKOR
# =========================================================

def score(f, direction):

    total = 0

    reasons = []

    if direction == "LONG":

        checks = [

            (
                f["ema9"]
                >
                f["ema21"]
                >
                f["ema50"],
                18,
                "EMA trend"
            ),

            (
                f["price"] > f["ema200"],
                8,
                "EMA200"
            ),

            (
                f["macd"]
                >
                f["macd_signal"],
                12,
                "MACD"
            ),

            (
                50 <= f["rsi"] <= 70,
                10,
                "RSI"
            ),

            (
                f["adx"] >= 22
                and
                f["plus_di"]
                >
                f["minus_di"],
                12,
                "ADX/DI"
            ),

            (
                f["volume_ratio"] >= 1.5,
                14,
                "Hacim"
            ),

            (
                f["taker_ratio"] >= 0.52,
                8,
                "Alıcı baskısı"
            ),

            (
                f["breakout"],
                10,
                "Breakout"
            ),

            (
                0.45
                <=
                f["bollinger"]
                <=
                1.15,
                8,
                "Bollinger"
            )
        ]

    else:

        checks = [

            (
                f["ema9"]
                <
                f["ema21"]
                <
                f["ema50"],
                18,
                "EMA trend"
            ),

            (
                f["price"] < f["ema200"],
                8,
                "EMA200"
            ),

            (
                f["macd"]
                <
                f["macd_signal"],
                12,
                "MACD"
            ),

            (
                30 <= f["rsi"] <= 50,
                10,
                "RSI"
            ),

            (
                f["adx"] >= 22
                and
                f["minus_di"]
                >
                f["plus_di"],
                12,
                "ADX/DI"
            ),

            (
                f["volume_ratio"] >= 1.5,
                14,
                "Hacim"
            ),

            (
                f["taker_ratio"] <= 0.48,
                8,
                "Satıcı baskısı"
            ),

            (
                f["breakdown"],
                10,
                "Breakdown"
            ),

            (
                -0.15
                <=
                f["bollinger"]
                <=
                0.55,
                8,
                "Bollinger"
            )
        ]

    for condition, points, reason in checks:

        try:

            if bool(condition):

                total += points

                reasons.append(reason)

        except:
            pass

    return (
        min(total, 100),
        reasons
    )


# =========================================================
# SL / TP HESAPLAMA
# =========================================================

def calculate_trade_levels(
    price,
    direction,
    supports,
    resistances
):

    # Fiyatın %0.15 dibindeki seviyeleri
    # hedef olarak kullanma
    min_distance = price * 0.0015

    if direction == "LONG":

        below = sorted(
            [
                x
                for x in supports
                if x < price - min_distance
            ],
            reverse=True
        )

        above = sorted(
            [
                x
                for x in resistances
                if x > price + min_distance
            ]
        )

        # En az 2 direnç lazım
        if len(above) < 2:
            return None

        # SL teknik desteğin biraz altı
        max_sl = (
            price * 0.98
        )

        if below:

            technical_sl = (
                below[0] * 0.997
            )

            # %2'den fazla risk olamaz
            sl = max(
                technical_sl,
                max_sl
            )

        else:

            sl = max_sl

        tp1 = above[0]

        # TP2 TP1'e aşırı yakın olmasın
        tp2 = None

        for resistance in above[1:]:

            if (
                resistance - tp1
                >=
                price * 0.003
            ):

                tp2 = resistance
                break

        if tp2 is None:
            return None

        risk = price - sl

        reward1 = tp1 - price

        reward2 = tp2 - price

    else:

        above = sorted(
            [
                x
                for x in resistances
                if x > price + min_distance
            ]
        )

        below = sorted(
            [
                x
                for x in supports
                if x < price - min_distance
            ],
            reverse=True
        )

        # En az 2 destek lazım
        if len(below) < 2:
            return None

        max_sl = (
            price * 1.02
        )

        if above:

            technical_sl = (
                above[0] * 1.003
            )

            # %2'den fazla risk olamaz
            sl = min(
                technical_sl,
                max_sl
            )

        else:

            sl = max_sl

        tp1 = below[0]

        tp2 = None

        for support in below[1:]:

            if (
                tp1 - support
                >=
                price * 0.003
            ):

                tp2 = support
                break

        if tp2 is None:
            return None

        risk = sl - price

        reward1 = price - tp1

        reward2 = price - tp2

    if risk <= 0:
        return None

    rr1 = reward1 / risk

    rr2 = reward2 / risk

    # Kötü risk/getiri ise sinyal verme
    if rr1 < MIN_TP1_RR:
        return None

    if rr2 < MIN_TP2_RR:
        return None

    tp1_percent = (
        reward1
        /
        price
        *
        100
    )

    tp2_percent = (
        reward2
        /
        price
        *
        100
    )

    sl_percent = (
        risk
        /
        price
        *
        100
    )

    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,

        "sl_percent": sl_percent,
        "tp1_percent": tp1_percent,
        "tp2_percent": tp2_percent,

        "rr1": rr1,
        "rr2": rr2
    }


# =========================================================
# COIN ANALIZI
# =========================================================

def analyze(symbol, btc):

    df15 = klines(
        symbol,
        "15m"
    )

    df1h = klines(
        symbol,
        "1h"
    )

    f15 = features(df15)

    f1h = features(df1h)

    supports, resistances = support_resistance(
        df15,
        df1h
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
                btc["ema9"]
                >=
                btc["ema21"]
            )

        else:

            btc_ok = (
                btc["ema9"]
                <=
                btc["ema21"]
            )

        final_score = (
            0.58 * score15
            +
            0.32 * score1h
            +
            (10 if btc_ok else 0)
        )

        final_score = min(
            round(final_score, 1),
            100
        )

        if final_score < MIN_SCORE:
            continue

        price = float(
            f15["price"]
        )

        levels = calculate_trade_levels(
            price,
            direction,
            supports,
            resistances
        )

        # Mantıklı destek/direnç hedefi yoksa
        # sinyal gönderme
        if levels is None:
            continue

        reasons = sorted(
            set(
                reasons15
                +
                reasons1h
            )
        )

        output.append({

            "symbol": symbol,

            "direction": direction,

            "score": final_score,

            "price": price,

            "sl": levels["sl"],

            "tp1": levels["tp1"],

            "tp2": levels["tp2"],

            "sl_percent":
                levels["sl_percent"],

            "tp1_percent":
                levels["tp1_percent"],

            "tp2_percent":
                levels["tp2_percent"],

            "rr1":
                levels["rr1"],

            "rr2":
                levels["rr2"],

            "rsi":
                f15["rsi"],

            "volume_ratio":
                f15["volume_ratio"],

            "adx":
                f15["adx"],

            "reasons":
                reasons
        })

    return output


# =========================================================
# TELEGRAM
# =========================================================

def telegram(message, chat_id=None):

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

    response = requests.post(

        f"https://api.telegram.org/bot{TOKEN}/sendMessage",

        json={
            "chat_id": target,
            "text": message
        },

        timeout=20
    )

    response.raise_for_status()


# =========================================================
# UPTIME
# =========================================================

def uptime():

    seconds = int(
        time.time()
        -
        bot_started
    )

    hours = seconds // 3600

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

def status(chat_id):

    if scan_running:

        state = "🔎 Tarama yapılıyor"

    else:

        state = "🟢 Çalışıyor"

    if last_scan_time:

        ago = int(
            time.time()
            -
            last_scan_time
        )

        last = (
            f"{ago} saniye önce"
        )

    else:

        last = "Henüz tamamlanmadı"

    if last_scan_duration is not None:

        duration = (
            f"{last_scan_duration:.1f} saniye"
        )

    else:

        duration = "-"

    message = (

        "📊 COIN RADAR DURUMU\n\n"

        f"Durum: {state}\n"

        f"Çalışma süresi: "
        f"{uptime()}\n\n"

        f"Minimum skor: "
        f"{MIN_SCORE}/100\n"

        "Minimum 24h hacim: $10M\n"

        "Analiz: 15m + 1h\n"

        "Yön: LONG + SHORT\n\n"

        "🛑 SL: Teknik / maksimum %2\n"

        "🎯 TP1: Destek/direnç\n"

        "🚀 TP2: Destek/direnç\n\n"

        "Minimum R:R:\n"

        f"TP1: {MIN_TP1_RR:.1f}R\n"

        f"TP2: {MIN_TP2_RR:.1f}R\n\n"

        f"Son tarama: {last}\n"

        f"Tarama süresi: {duration}\n"

        f"Taranan coin: "
        f"{last_scan_count}\n"

        f"Yeni sinyal: "
        f"{last_signal_count}"
    )

    telegram(
        message,
        chat_id
    )


# =========================================================
# TARAMA
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
                "⏳ Tarama zaten devam ediyor.",
                manual_chat
            )

        return

    scan_running = True

    started = time.time()

    try:

        coin_list = symbols()

        print(
            f"10M+ hacimli "
            f"{len(coin_list)} coin bulundu.",
            flush=True
        )

        if manual_chat:

            telegram(
                "🔎 Tarama başladı.\n\n"
                f"{len(coin_list)} coin "
                "analiz ediliyor.",
                manual_chat
            )

        # BTC SADECE 1 KEZ CEKILIR
        btc = features(
            klines(
                "BTCUSDT",
                "15m"
            )
        )

        signals = 0

        processed = 0

        for symbol, volume24 in coin_list:

            try:

                results = analyze(
                    symbol,
                    btc
                )

                processed += 1

                for result in results:

                    key = (
                        result["symbol"],
                        result["direction"]
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

                    if result["direction"] == "LONG":

                        icon = "🟢"

                        sl_sign = "-"
                        tp_sign = "+"

                    else:

                        icon = "🔴"

                        sl_sign = "+"
                        tp_sign = "-"

                    volume_m = (
                        volume24
                        /
                        1_000_000
                    )

                    message = (

                        f"🔥 {result['symbol']}\n\n"

                        f"{icon} "
                        f"{result['direction']}\n"

                        f"⭐ Radar skoru: "
                        f"{result['score']}/100\n\n"

                        f"💰 Giriş: "
                        f"{result['price']:.8g}\n\n"

                        f"🛑 SL: "
                        f"{result['sl']:.8g} "
                        f"({sl_sign}"
                        f"%{result['sl_percent']:.2f})\n\n"

                        f"🎯 TP1: "
                        f"{result['tp1']:.8g} "
                        f"({tp_sign}"
                        f"%{result['tp1_percent']:.2f})\n"

                        f"R:R TP1: "
                        f"1:{result['rr1']:.2f}\n\n"

                        f"🚀 TP2: "
                        f"{result['tp2']:.8g} "
                        f"({tp_sign}"
                        f"%{result['tp2_percent']:.2f})\n"

                        f"R:R TP2: "
                        f"1:{result['rr2']:.2f}\n\n"

                        f"24h Futures hacmi: "
                        f"${volume_m:.1f}M\n"

                        f"RSI: "
                        f"{result['rsi']:.1f}\n"

                        f"Hacim oranı: "
                        f"x{result['volume_ratio']:.2f}\n"

                        f"ADX: "
                        f"{result['adx']:.1f}\n\n"

                        "Onaylar:\n"

                        f"{', '.join(result['reasons'])}\n\n"

                        "TP seviyeleri 15m + 1h "
                        "destek/dirençlerden hesaplandı.\n"

                        "⚠️ Radar skoru kazanma "
                        "olasılığı değildir."
                    )

                    telegram(message)

                    last_alert[key] = now

                    signals += 1

            except Exception as e:

                print(
                    f"{symbol} hata: {e}",
                    flush=True
                )

        duration = (
            time.time()
            -
            started
        )

        last_scan_time = time.time()

        last_scan_duration = duration

        last_scan_count = processed

        last_signal_count = signals

        print(
            f"Tarama tamamlandi | "
            f"{processed} coin | "
            f"{signals} sinyal | "
            f"{duration:.1f} saniye",
            flush=True
        )

        if manual_chat:

            telegram(
                "✅ Tarama tamamlandı.\n\n"
                f"Taranan: {processed}\n"
                f"Yeni sinyal: {signals}\n"
                f"Süre: {duration:.1f} saniye",
                manual_chat
            )

    except Exception as e:

        print(
            f"Genel tarama hatasi: {e}",
            flush=True
        )

        if manual_chat:

            telegram(
                f"❌ Tarama hatası:\n{e}",
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

            response = requests.get(

                f"https://api.telegram.org/bot{TOKEN}/getUpdates",

                params={
                    "timeout": 25,
                    "offset": telegram_offset
                },

                timeout=35
            )

            response.raise_for_status()

            updates = response.json()

            for update in updates.get(
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

                        "🤖 COIN RADAR\n\n"

                        "/durum - Bot durumu\n"

                        "/tara - Hemen tara\n"

                        "/yardim - Yardım\n\n"

                        "Filtreler:\n"

                        "24h hacim ≥ $10M\n"

                        f"Skor ≥ {MIN_SCORE}\n"

                        "15m + 1h\n"

                        "LONG + SHORT\n\n"

                        "SL: Teknik, max %2\n"

                        "TP1: Destek/direnç\n"

                        "TP2: Destek/direnç",

                        chat_id
                    )

                elif text == "/durum":

                    status(
                        chat_id
                    )

                elif text == "/tara":

                    if scan_running:

                        telegram(
                            "⏳ Şu anda tarama "
                            "devam ediyor.",
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
                f"Telegram komut hatasi: {e}",
                flush=True
            )

            time.sleep(5)


# =========================================================
# BASLAT
# =========================================================

def run():

    print(
        "COIN RADAR BASLADI",
        flush=True
    )

    try:

        telegram(

            "🟢 COIN RADAR AKTİF\n\n"

            "24h Futures hacmi ≥ $10M\n"

            f"Minimum skor: "
            f"{MIN_SCORE}/100\n"

            "Analiz: 15m + 1h\n"

            "LONG + SHORT\n\n"

            "🛑 SL: Teknik, maksimum %2\n"

            "🎯 TP1: Destek/direnç\n"

            "🚀 TP2: Destek/direnç\n\n"

            "Kötü R:R olan sinyaller elenir.\n\n"

            "/durum\n"
            "/tara\n"
            "/yardim"
        )

    except Exception as e:

        print(
            f"Telegram baslangic hatasi: {e}",
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
