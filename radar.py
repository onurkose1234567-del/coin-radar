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
TOP = int(os.getenv("TOP_SYMBOLS", "80"))
INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

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
# BINANCE
# =========================================================

def get(path, params=None):
    r = requests.get(
        BASE + path,
        params=params,
        timeout=15
    )
    r.raise_for_status()
    return r.json()


def symbols():
    tick = get("/fapi/v1/ticker/24hr")

    xs = [
        x for x in tick
        if x["symbol"].endswith("USDT")
        and float(x.get("quoteVolume", 0)) > 0
    ]

    xs.sort(
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    result = []

    for x in xs:
        sym = x["symbol"]

        if sym == "USDCUSDT":
            continue

        result.append(sym)

        if len(result) >= TOP:
            break

    return result


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
            "t", "o", "h", "l", "c", "v",
            "T", "q", "n", "tb", "tq", "x"
        ]
    )

    for col in [
        "o", "h", "l", "c",
        "v", "q", "tb"
    ]:
        d[col] = pd.to_numeric(d[col])

    return d


# =========================================================
# INDIKATORLER
# =========================================================

def rsi(s, n=14):
    x = s.diff()

    up = x.clip(lower=0).ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    dn = (-x.clip(upper=0)).ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    return 100 - (
        100 / (
            1 + up / dn.replace(0, np.nan)
        )
    )


def atr(d, n=14):
    pc = d.c.shift()

    tr = pd.concat(
        [
            d.h - d.l,
            (d.h - pc).abs(),
            (d.l - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()


def adx(d, n=14):
    up = d.h.diff()
    down = -d.l.diff()

    tr = atr(d, 1)

    plus = 100 * (
        up.where(
            (up > down) & (up > 0),
            0
        ).ewm(
            alpha=1 / n,
            adjust=False
        ).mean()
        /
        tr.ewm(
            alpha=1 / n,
            adjust=False
        ).mean()
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
        tr.ewm(
            alpha=1 / n,
            adjust=False
        ).mean()
    )

    dx = (
        100
        * (plus - minus).abs()
        / (plus + minus).replace(0, np.nan)
    )

    return (
        dx.ewm(
            alpha=1 / n,
            adjust=False
        ).mean(),
        plus,
        minus
    )


# =========================================================
# ANALIZ VERILERI
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

    mac = (
        c.ewm(
            span=12,
            adjust=False
        ).mean()
        -
        c.ewm(
            span=26,
            adjust=False
        ).mean()
    )

    sig = mac.ewm(
        span=9,
        adjust=False
    ).mean()

    aa = atr(d)

    ax, di_p, di_m = adx(d)

    vm = v.rolling(20).mean()

    vol_ratio = v / vm

    mid = c.rolling(20).mean()

    sd = c.rolling(20).std()

    lower = mid - 2 * sd
    upper = mid + 2 * sd

    bbpos = (
        (c - lower)
        /
        (upper - lower)
    )

    buy_ratio = (
        d.tb
        /
        d.v.replace(0, np.nan)
    )

    hi20 = (
        d.h.shift(1)
        .rolling(20)
        .max()
    )

    lo20 = (
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
        "mac": mac.iloc[-1],
        "sig": sig.iloc[-1],
        "atr": aa.iloc[-1],
        "adx": ax.iloc[-1],
        "dip": di_p.iloc[-1],
        "dim": di_m.iloc[-1],
        "vr": vol_ratio.iloc[-1],
        "bb": bbpos.iloc[-1],
        "br": buy_ratio.iloc[-1],
        "break_hi": (
            c.iloc[-1] > hi20.iloc[-1]
        ),
        "break_lo": (
            c.iloc[-1] < lo20.iloc[-1]
        )
    }


# =========================================================
# PUANLAMA
# =========================================================

def score(f, direction):
    s = 0
    reasons = []

    if direction == "LONG":

        checks = [
            (
                f["e9"] > f["e21"] > f["e50"],
                18,
                "EMA trend"
            ),
            (
                f["price"] > f["e200"],
                8,
                "EMA200"
            ),
            (
                f["mac"] > f["sig"],
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
                and f["dip"] > f["dim"],
                12,
                "ADX/DI"
            ),
            (
                f["vr"] >= 1.5,
                14,
                "Volume spike"
            ),
            (
                f["br"] >= 0.52,
                8,
                "Taker buy"
            ),
            (
                f["break_hi"],
                10,
                "Breakout"
            ),
            (
                0.45 <= f["bb"] <= 1.15,
                8,
                "Bollinger"
            )
        ]

    else:

        checks = [
            (
                f["e9"] < f["e21"] < f["e50"],
                18,
                "EMA trend"
            ),
            (
                f["price"] < f["e200"],
                8,
                "EMA200"
            ),
            (
                f["mac"] < f["sig"],
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
                and f["dim"] > f["dip"],
                12,
                "ADX/DI"
            ),
            (
                f["vr"] >= 1.5,
                14,
                "Volume spike"
            ),
            (
                f["br"] <= 0.48,
                8,
                "Taker sell"
            ),
            (
                f["break_lo"],
                10,
                "Breakdown"
            ),
            (
                -0.15 <= f["bb"] <= 0.55,
                8,
                "Bollinger"
            )
        ]

    for ok, weight, name in checks:
        if bool(ok):
            s += weight
            reasons.append(name)

    return min(s, 100), reasons


# =========================================================
# COIN ANALIZI
# =========================================================

def analyze(sym):
    f15 = features(
        klines(sym, "15m")
    )

    f1 = features(
        klines(sym, "1h")
    )

    btc = features(
        klines("BTCUSDT", "15m")
    )

    output = []

    for direction in [
        "LONG",
        "SHORT"
    ]:

        score15, reason15 = score(
            f15,
            direction
        )

        score1h, reason1h = score(
            f1,
            direction
        )

        if direction == "LONG":
            btc_ok = (
                btc["e9"] >= btc["e21"]
            )

        else:
            btc_ok = (
                btc["e9"] <= btc["e21"]
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

            price = f15["price"]
            current_atr = f15["atr"]

            mult = (
                1
                if direction == "LONG"
                else -1
            )

            sl = (
                price
                -
                mult * 1.35 * current_atr
            )

            tp1 = (
                price
                +
                mult * 1.5 * current_atr
            )

            tp2 = (
                price
                +
                mult * 2.5 * current_atr
            )

            output.append(
                (
                    total,
                    direction,
                    price,
                    sl,
                    tp1,
                    tp2,
                    sorted(
                        set(
                            reason15
                            +
                            reason1h
                        )
                    ),
                    f15["rsi"],
                    f15["vr"],
                    f15["adx"]
                )
            )

    return output


# =========================================================
# TELEGRAM MESAJ GONDER
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
# SURE YAZISI
# =========================================================

def uptime_text():

    seconds = int(
        time.time() - bot_started
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
# DURUM KOMUTU
# =========================================================

def send_status(chat_id):

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

        last_text = (
            f"{ago} saniye önce"
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
        "📊 COIN RADAR DURUMU\n\n"
        f"Durum: {state}\n"
        f"Çalışma süresi: {uptime_text()}\n\n"
        f"Minimum skor: {MIN_SCORE}/100\n"
        f"Hedef coin sayısı: {TOP}\n"
        f"Tarama aralığı: {INTERVAL} sn\n\n"
        f"Son tarama: {last_text}\n"
        f"Son tarama süresi: {duration_text}\n"
        f"Taranan coin: {last_scan_count}\n"
        f"Yeni sinyal: {last_signal_count}\n\n"
        "Zaman dilimleri: 15m + 1h\n"
        "Yönler: LONG + SHORT"
    )

    telegram(
        msg,
        chat_id
    )


# =========================================================
# TEK TARAMA
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

    start = time.time()

    try:

        coin_list = symbols()

        print(
            f"🔎 Tarama başladı - "
            f"{len(coin_list)} coin",
            flush=True
        )

        if manual_chat:

            telegram(
                "🔎 Manuel tarama başladı.\n"
                f"{len(coin_list)} coin "
                "kontrol ediliyor.",
                manual_chat
            )

        signal_count = 0

        for sym in coin_list:

            try:

                results = analyze(sym)

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
                        3600
                    ):
                        continue

                    msg = (
                        f"🔥 {sym} — {direction}\n\n"
                        f"Radar skoru: {sc}/100\n"
                        f"Giriş: {price:.8g}\n"
                        f"SL: {sl:.8g}\n"
                        f"TP1: {tp1:.8g}\n"
                        f"TP2: {tp2:.8g}\n\n"
                        f"15m RSI: {current_rsi:.1f}\n"
                        f"Hacim: x{volume_ratio:.2f}\n"
                        f"ADX: {current_adx:.1f}\n\n"
                        f"Onaylar: "
                        f"{', '.join(reasons)}\n\n"
                        "⚠️ Radar skoru gerçekleşmiş "
                        "kazanma oranı değildir."
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
            time.time() - start
        )

        last_scan_time = time.time()
        last_scan_duration = duration
        last_scan_count = len(coin_list)
        last_signal_count = signal_count

        print(
            f"✅ Tarama tamamlandı - "
            f"{len(coin_list)} coin - "
            f"{signal_count} yeni sinyal - "
            f"{duration:.1f} sn",
            flush=True
        )

        if manual_chat:

            telegram(
                "✅ Manuel tarama tamamlandı.\n\n"
                f"Taranan coin: "
                f"{len(coin_list)}\n"
                f"Yeni sinyal: "
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
                f"❌ Tarama hatası:\n{e}",
                manual_chat
            )

    finally:

        scan_running = False
        scan_lock.release()


# =========================================================
# OTOMATIK RADAR THREAD
# =========================================================

def radar_loop():

    while True:

        scan_market()

        time.sleep(INTERVAL)


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
                    + 1
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

                # Sadece bizim ayarladığımız
                # Telegram sohbetinden komut kabul et
                if CHAT and str(chat_id) != str(CHAT):
                    continue

                if text in [
                    "/start",
                    "/yardim",
                    "/help"
                ]:

                    telegram(
                        "🤖 COIN RADAR\n\n"
                        "Komutlar:\n\n"
                        "/durum - Bot durumunu göster\n"
                        "/tara - Hemen piyasa taraması yap\n"
                        "/yardim - Komutları göster\n\n"
                        "Radar ayrıca piyasayı otomatik "
                        "olarak taramaya devam eder.",
                        chat_id
                    )

                elif text == "/durum":

                    send_status(
                        chat_id
                    )

                elif text == "/tara":

                    if scan_running:

                        telegram(
                            "⏳ Şu anda zaten bir "
                            "tarama devam ediyor.\n\n"
                            "Biraz sonra tekrar dene.",
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
        "Coin Radar v2 started",
        flush=True
    )

    try:

        telegram(
            "🟢 COIN RADAR V2 AKTİF\n\n"
            "Binance Futures radar başladı.\n\n"
            f"Minimum skor: {MIN_SCORE}/100\n"
            f"Coin sayısı: {TOP}\n"
            f"Tarama aralığı: {INTERVAL} saniye\n\n"
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
