import os
import time
import math
import json
import threading
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# AYARLAR
# ============================================================

BASE = "https://fapi.binance.com"

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Minimum TOPLAM skor
MIN_SCORE = float(os.getenv("MIN_SCORE", "80"))

# Haber olmasa bile teknik tarafın minimum kalitesi
MIN_TECH_SCORE = float(os.getenv("MIN_TECH_SCORE", "72"))

# Binance Futures minimum 24 saatlik quote hacmi
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "10000000"))

# Tarama aralığı
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "180"))

# Aynı coin/yön tekrar alarm süresi
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600"))

# Maksimum SL uzaklığı
MAX_SL_PERCENT = 2.0

# Haberlerin kaç saat yeni sayılacağı
NEWS_MAX_AGE_HOURS = 12

# Haberleri kaç dakikada bir yenile
NEWS_REFRESH_SECONDS = 300

# Her taramada maksimum kaç coin analiz edilsin.
# Hacmi en yüksek olanlardan başlanır.
MAX_SYMBOLS_PER_SCAN = int(os.getenv("MAX_SYMBOLS_PER_SCAN", "120"))

# HTTP

REQUEST_TIMEOUT = 12

# ============================================================
# SANAL İŞLEM (PAPER TRADING)
# ============================================================

PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "200"))
PAPER_RISK_PERCENT = float(os.getenv("PAPER_RISK_PERCENT", "1"))
PAPER_TAKER_FEE = float(os.getenv("PAPER_TAKER_FEE", "0.0005"))  # %0.05
PAPER_FILE = os.getenv("PAPER_FILE", "paper_trades.json")
PAPER_MAX_OPEN = int(os.getenv("PAPER_MAX_OPEN", "5"))
paper_lock = threading.Lock()

session = requests.Session()
session.headers.update({
    "User-Agent": "CoinRadarV5/1.0"
})

# ============================================================
# HABER KAYNAKLARI
# ============================================================

NEWS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "CryptoSlate": "https://cryptoslate.com/feed/",
}

# Güçlü katalizör kelimeleri
STRONG_NEWS_WORDS = [
    "hack",
    "hacked",
    "exploit",
    "breach",
    "stolen",
    "attack",
    "listing",
    "listed",
    "delisting",
    "delisted",
    "etf",
    "approval",
    "approved",
    "lawsuit",
    "sec",
    "regulation",
    "partnership",
    "partners",
    "launch",
    "mainnet",
    "upgrade",
    "token unlock",
    "unlock",
    "burn",
    "airdrop",
    "acquisition",
    "acquires",
    "bankruptcy",
    "investigation",
    "settlement",
]

MEDIUM_NEWS_WORDS = [
    "whale",
    "institutional",
    "fund",
    "investment",
    "exchange",
    "network",
    "protocol",
    "treasury",
    "adoption",
    "staking",
    "validator",
    "futures",
    "open interest",
]

# Coin sembolü -> haberlerde aranabilecek isimler
COIN_NAMES = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "ether", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
    "BNB": ["bnb", "binance coin"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"],
    "AVAX": ["avalanche", "avax"],
    "LINK": ["chainlink", "link"],
    "DOT": ["polkadot", "dot"],
    "LTC": ["litecoin", "ltc"],
    "BCH": ["bitcoin cash", "bch"],
    "UNI": ["uniswap", "uni"],
    "AAVE": ["aave"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism", "op"],
    "SUI": ["sui"],
    "APT": ["aptos", "apt"],
    "NEAR": ["near protocol", "near"],
    "ATOM": ["cosmos", "atom"],
    "FIL": ["filecoin", "fil"],
    "INJ": ["injective", "inj"],
    "TIA": ["celestia", "tia"],
    "SEI": ["sei"],
    "ENA": ["ethena", "ena"],
    "WIF": ["dogwifhat", "wif"],
    "PEPE": ["pepe"],
    "SHIB": ["shiba inu", "shib"],
    "TRX": ["tron", "trx"],
    "TON": ["toncoin", "ton"],
    "ARK": ["ark", "ark ecosystem"],
}

# ============================================================
# GLOBAL DURUM
# ============================================================

start_time = time.time()

last_scan_time = 0
last_scan_duration = 0
last_scan_count = 0
last_signal_count = 0

alerts = {}

news_cache = []
news_last_update = 0
news_lock = threading.Lock()

scan_lock = threading.Lock()

# ============================================================
# HTTP
# ============================================================

def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"429 rate limit. {wait} sn bekleniyor...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            print(f"HTTP hata: {url} -> {e}")

            if attempt < retries - 1:
                time.sleep(2 + attempt * 2)

    return None


def get_text(url):
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code == 200:
            return r.text

    except Exception as e:
        print("RSS hata:", url, e)

    return None

# ============================================================
# TELEGRAM
# ============================================================

def telegram(text):
    if not TOKEN or not CHAT:
        print("Telegram ayarları eksik.")
        print(text)
        return

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        session.post(
            url,
            data={
                "chat_id": CHAT,
                "text": text,
                "disable_web_page_preview": True
            },
            timeout=12
        )

    except Exception as e:
        print("Telegram hata:", e)


def uptime():
    sec = int(time.time() - start_time)

    h = sec // 3600
    m = (sec % 3600) // 60

    return f"{h}s {m}dk"

# ============================================================
# BINANCE SYMBOL / HACİM
# ============================================================

def get_symbols():
    exchange = get_json(BASE + "/fapi/v1/exchangeInfo")
    tickers = get_json(BASE + "/fapi/v1/ticker/24hr")

    if not exchange or not tickers:
        return []

    valid = {}

    for s in exchange.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
        ):
            valid[s["symbol"]] = s.get("baseAsset", "")

    result = []

    for t in tickers:
        sym = t.get("symbol")

        if sym not in valid:
            continue

        try:
            qv = float(t.get("quoteVolume", 0))
        except:
            qv = 0

        if qv >= MIN_VOLUME:
            result.append({
                "symbol": sym,
                "base": valid[sym],
                "volume": qv
            })

    # Hacmi yüksek coinlerden başla
    result.sort(
        key=lambda x: x["volume"],
        reverse=True
    )

    return result[:MAX_SYMBOLS_PER_SCAN]

# ============================================================
# KLINE
# ============================================================

def klines(symbol, interval, limit=220):
    data = get_json(
        BASE + "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if not data:
        return None

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote"
    ]

    for c in numeric:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return df

# ============================================================
# INDIKATÖRLER
# ============================================================

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def calculate_adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where(
        (up > down) & (up > 0),
        up,
        0.0
    )

    minus_dm = np.where(
        (down > up) & (down > 0),
        down,
        0.0
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    ).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    ).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = 100 * plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm / atr.replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return adx, plus_di, minus_di


def features(df):
    d = df.copy()

    d["ema9"] = d["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["ema21"] = d["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    d["ema200"] = d["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    d["rsi"] = calculate_rsi(d["close"])

    ema12 = d["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = d["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = ema12 - ema26

    d["macd_signal"] = d["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["atr"] = calculate_atr(d)

    adx, plus_di, minus_di = calculate_adx(d)

    d["adx"] = adx
    d["plus_di"] = plus_di
    d["minus_di"] = minus_di

    vol_avg = d["volume"].rolling(20).mean()

    d["vol_ratio"] = (
        d["volume"]
        / vol_avg.replace(0, np.nan)
    )

    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()

    upper = mid + 2 * std
    lower = mid - 2 * std

    d["bb_pos"] = (
        (d["close"] - lower)
        / (upper - lower).replace(0, np.nan)
    )

    d["taker_ratio"] = (
        d["taker_buy_base"]
        / d["volume"].replace(0, np.nan)
    )

    return d

# ============================================================
# BREAKOUT / RETEST
# ============================================================

def breakout_retest(df, direction):
    if len(df) < 30:
        return False, False

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Son kapanan mum
    current = float(close.iloc[-2])

    # Önceki 20 mum
    prev_high = float(
        high.iloc[-22:-2].max()
    )

    prev_low = float(
        low.iloc[-22:-2].min()
    )

    breakout = False
    retest = False

    if direction == "LONG":
        breakout = current > prev_high

        # Son birkaç mumda direnç üstüne kırıp
        # tekrar seviyeyi test etmiş mi?
        recent_high = close.iloc[-6:-2].max()

        if recent_high > prev_high:
            recent_low = low.iloc[-4:-1].min()

            tolerance = prev_high * 0.004

            if (
                recent_low <= prev_high + tolerance
                and current >= prev_high
            ):
                retest = True

    else:
        breakout = current < prev_low

        recent_low_close = close.iloc[-6:-2].min()

        if recent_low_close < prev_low:
            recent_high = high.iloc[-4:-1].max()

            tolerance = prev_low * 0.004

            if (
                recent_high >= prev_low - tolerance
                and current <= prev_low
            ):
                retest = True

    return breakout, retest

# ============================================================
# DESTEK / DİRENÇ
# ============================================================

def swing_levels(df, window=3):
    supports = []
    resistances = []

    highs = df["high"].values
    lows = df["low"].values

    for i in range(
        window,
        len(df) - window
    ):
        if lows[i] == min(
            lows[i-window:i+window+1]
        ):
            supports.append(
                float(lows[i])
            )

        if highs[i] == max(
            highs[i-window:i+window+1]
        ):
            resistances.append(
                float(highs[i])
            )

    return supports, resistances


def merge_levels(levels, tolerance=0.003):
    if not levels:
        return []

    levels = sorted(levels)

    merged = [levels[0]]

    for level in levels[1:]:
        last = merged[-1]

        if abs(level - last) / last <= tolerance:
            merged[-1] = (
                last + level
            ) / 2
        else:
            merged.append(level)

    return merged


def support_resistance(df15, df1h):
    s15, r15 = swing_levels(
        df15.tail(100)
    )

    s1h, r1h = swing_levels(
        df1h.tail(100)
    )

    supports = merge_levels(
        s15 + s1h
    )

    resistances = merge_levels(
        r15 + r1h
    )

    return supports, resistances

# ============================================================
# TP / SL
# ============================================================

def calculate_trade_levels(
    direction,
    price,
    atr,
    supports,
    resistances
):
    if price <= 0 or atr <= 0:
        return None

    # Seviyenin fiyatın en az %0.15 uzağında olması
    min_distance = price * 0.0015

    if direction == "LONG":

        below = [
            x for x in supports
            if x < price - min_distance
        ]

        above = [
            x for x in resistances
            if x > price + min_distance
        ]

        # Teknik SL
        if below:
            technical_sl = max(below) - atr * 0.20
        else:
            technical_sl = price - atr * 1.25

        # Maksimum %2
        max_sl = price * (
            1 - MAX_SL_PERCENT / 100
        )

        sl = max(
            technical_sl,
            max_sl
        )

        risk = price - sl

        if risk <= 0:
            return None

        # TP1
        tp1_candidates = [
            x for x in above
            if x >= price + risk
        ]

        if tp1_candidates:
            tp1 = min(tp1_candidates)
        else:
            tp1 = price + max(
                risk * 1.2,
                atr * 1.5
            )

        # TP2
        tp2_candidates = [
            x for x in above
            if x > tp1
            and x >= price + risk * 2
        ]

        if tp2_candidates:
            tp2 = min(tp2_candidates)
        else:
            tp2 = price + max(
                risk * 2.2,
                atr * 3
            )

    else:

        above = [
            x for x in resistances
            if x > price + min_distance
        ]

        below = [
            x for x in supports
            if x < price - min_distance
        ]

        if above:
            technical_sl = min(above) + atr * 0.20
        else:
            technical_sl = price + atr * 1.25

        max_sl = price * (
            1 + MAX_SL_PERCENT / 100
        )

        sl = min(
            technical_sl,
            max_sl
        )

        risk = sl - price

        if risk <= 0:
            return None

        tp1_candidates = [
            x for x in below
            if x <= price - risk
        ]

        if tp1_candidates:
            tp1 = max(tp1_candidates)
        else:
            tp1 = price - max(
                risk * 1.2,
                atr * 1.5
            )

        tp2_candidates = [
            x for x in below
            if x < tp1
            and x <= price - risk * 2
        ]

        if tp2_candidates:
            tp2 = max(tp2_candidates)
        else:
            tp2 = price - max(
                risk * 2.2,
                atr * 3
            )

    if sl <= 0 or tp1 <= 0 or tp2 <= 0:
        return None

    rr1 = abs(tp1 - price) / abs(price - sl)
    rr2 = abs(tp2 - price) / abs(price - sl)

    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2
    }

# ============================================================
# HABER RSS
# ============================================================

def parse_date(text):
    if not text:
        return None

    try:
        dt = parsedate_to_datetime(text)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except:
        return None


def clean_text(text):
    if not text:
        return ""

    text = text.replace(
        "<![CDATA[",
        ""
    ).replace(
        "]]>",
        ""
    )

    return " ".join(
        text.split()
    )


def parse_rss(source, xml_text):
    result = []

    try:
        root = ET.fromstring(xml_text)

        # RSS item
        for item in root.findall(".//item"):
            title = clean_text(
                item.findtext("title")
            )

            link = clean_text(
                item.findtext("link")
            )

            pub = (
                item.findtext("pubDate")
                or item.findtext("date")
            )

            dt = parse_date(pub)

            result.append({
                "source": source,
                "title": title,
                "link": link,
                "date": dt
            })

        # Atom fallback
        if not result:
            ns = {
                "a":
                "http://www.w3.org/2005/Atom"
            }

            for entry in root.findall(
                ".//a:entry",
                ns
            ):
                title = clean_text(
                    entry.findtext(
                        "a:title",
                        namespaces=ns
                    )
                )

                link_el = entry.find(
                    "a:link",
                    ns
                )

                link = ""

                if link_el is not None:
                    link = link_el.attrib.get(
                        "href",
                        ""
                    )

                pub = (
                    entry.findtext(
                        "a:published",
                        namespaces=ns
                    )
                    or entry.findtext(
                        "a:updated",
                        namespaces=ns
                    )
                )

                dt = None

                try:
                    if pub:
                        dt = datetime.fromisoformat(
                            pub.replace(
                                "Z",
                                "+00:00"
                            )
                        )
                except:
                    pass

                result.append({
                    "source": source,
                    "title": title,
                    "link": link,
                    "date": dt
                })

    except Exception as e:
        print(
            f"RSS parse hata {source}:",
            e
        )

    return result


def update_news():
    global news_cache
    global news_last_update

    now = time.time()

    if (
        news_cache
        and now - news_last_update
        < NEWS_REFRESH_SECONDS
    ):
        return

    all_news = []

    for source, url in NEWS_FEEDS.items():
        xml = get_text(url)

        if not xml:
            continue

        articles = parse_rss(
            source,
            xml
        )

        all_news.extend(articles)

    # Aynı başlıkları temizle
    unique = {}
    now_dt = datetime.now(
        timezone.utc
    )

    for article in all_news:
        title = article["title"].strip()

        if not title:
            continue

        key = (
            title.lower()
            .replace("bitcoin", "btc")
            .replace("ethereum", "eth")
        )

        dt = article.get("date")

        if dt:
            age = (
                now_dt - dt
            ).total_seconds() / 3600

            # 24 saatten eski haberleri cache'e alma
            if age > 24:
                continue

        if key not in unique:
            unique[key] = article

    with news_lock:
        news_cache = list(
            unique.values()
        )

        news_last_update = now

    print(
        f"Haber havuzu güncellendi: "
        f"{len(news_cache)} haber"
    )

# ============================================================
# HABER - COIN EŞLEŞTİRME
# ============================================================

def get_coin_terms(base):
    base = base.upper()

    terms = COIN_NAMES.get(
        base,
        []
    ).copy()

    # BTC, ETH gibi semboller için
    # kısa sembollerde yanlış eşleşme olabilir.
    if len(base) >= 4:
        terms.append(base.lower())

    return list(set(terms))


def article_matches_coin(article, base):
    title = article["title"].lower()

    terms = get_coin_terms(base)

    for term in terms:
        # Çok kısa sembolleri kelime olarak kontrol et
        if len(term) <= 3:
            words = (
                title.replace("-", " ")
                .replace("/", " ")
                .replace(",", " ")
                .replace(".", " ")
                .split()
            )

            if term in words:
                return True

        elif term in title:
            return True

    return False


def news_importance(title):
    t = title.lower()

    strong = sum(
        1
        for word in STRONG_NEWS_WORDS
        if word in t
    )

    medium = sum(
        1
        for word in MEDIUM_NEWS_WORDS
        if word in t
    )

    if strong >= 2:
        return 10

    if strong == 1:
        return 7

    if medium >= 2:
        return 5

    if medium == 1:
        return 3

    return 2


def news_score(base):
    now = datetime.now(
        timezone.utc
    )

    matches = []

    with news_lock:
        articles = list(news_cache)

    for article in articles:

        if not article_matches_coin(
            article,
            base
        ):
            continue

        dt = article.get("date")

        if dt:
            age = (
                now - dt
            ).total_seconds() / 3600

            if age > NEWS_MAX_AGE_HOURS:
                continue

        else:
            age = None

        importance = news_importance(
            article["title"]
        )

        # Çok yeni habere biraz öncelik
        if age is not None:
            if age <= 2:
                importance = min(
                    10,
                    importance + 2
                )

            elif age <= 6:
                importance = min(
                    10,
                    importance + 1
                )

        matches.append({
            **article,
            "age": age,
            "importance": importance
        })

    if not matches:
        return 0, None

    matches.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    best = matches[0]

    # Haber maksimum +10
    score = min(
        10,
        best["importance"]
    )

    return score, best

# ============================================================
# TEKNİK SKOR
# ============================================================

def technical_score(df15, df1h, direction):
    a = df15.iloc[-2]
    b = df1h.iloc[-2]

    score = 0
    reasons = []

    breakout, retest = breakout_retest(
        df15,
        direction
    )

    if direction == "LONG":

        if (
            a["ema9"] > a["ema21"]
            > a["ema50"]
        ):
            score += 16
            reasons.append(
                "15m EMA trend yukarı"
            )

        if a["close"] > a["ema200"]:
            score += 7
            reasons.append(
                "15m EMA200 üstü"
            )

        if (
            b["ema9"] > b["ema21"]
            > b["ema50"]
        ):
            score += 10
            reasons.append(
                "1h trend yukarı"
            )

        if b["close"] > b["ema200"]:
            score += 5

        if a["macd"] > a["macd_signal"]:
            score += 9
            reasons.append(
                "MACD pozitif"
            )

        if 48 <= a["rsi"] <= 70:
            score += 8
            reasons.append(
                f"RSI {a['rsi']:.1f}"
            )

        if (
            a["adx"] >= 20
            and a["plus_di"] > a["minus_di"]
        ):
            score += 10
            reasons.append(
                f"ADX {a['adx']:.1f}"
            )

        if a["vol_ratio"] >= 1.30:
            score += 10
            reasons.append(
                f"Hacim {a['vol_ratio']:.2f}x"
            )

        if a["taker_ratio"] >= 0.52:
            score += 5
            reasons.append(
                "Alıcı baskısı"
            )

        if breakout:
            score += 10
            reasons.append(
                "Breakout"
            )

        if retest:
            score += 10
            reasons.append(
                "Breakout + retest"
            )

    else:

        if (
            a["ema9"] < a["ema21"]
            < a["ema50"]
        ):
            score += 16
            reasons.append(
                "15m EMA trend aşağı"
            )

        if a["close"] < a["ema200"]:
            score += 7
            reasons.append(
                "15m EMA200 altı"
            )

        if (
            b["ema9"] < b["ema21"]
            < b["ema50"]
        ):
            score += 10
            reasons.append(
                "1h trend aşağı"
            )

        if b["close"] < b["ema200"]:
            score += 5

        if a["macd"] < a["macd_signal"]:
            score += 9
            reasons.append(
                "MACD negatif"
            )

        if 30 <= a["rsi"] <= 52:
            score += 8
            reasons.append(
                f"RSI {a['rsi']:.1f}"
            )

        if (
            a["adx"] >= 20
            and a["minus_di"] > a["plus_di"]
        ):
            score += 10
            reasons.append(
                f"ADX {a['adx']:.1f}"
            )

        if a["vol_ratio"] >= 1.30:
            score += 10
            reasons.append(
                f"Hacim {a['vol_ratio']:.2f}x"
            )

        if a["taker_ratio"] <= 0.48:
            score += 5
            reasons.append(
                "Satıcı baskısı"
            )

        if breakout:
            score += 10
            reasons.append(
                "Breakdown"
            )

        if retest:
            score += 10
            reasons.append(
                "Breakdown + retest"
            )

    # Teknik skor max 90'a normalize
    score = min(
        score,
        90
    )

    return score, reasons

# ============================================================
# BTC FİLTRESİ
# ============================================================

def btc_state():
    df = klines(
        "BTCUSDT",
        "15m",
        100
    )

    if df is None:
        return "NEUTRAL"

    df = features(df)

    r = df.iloc[-2]

    if (
        r["ema9"] > r["ema21"]
        and r["macd"] > r["macd_signal"]
    ):
        return "LONG"

    if (
        r["ema9"] < r["ema21"]
        and r["macd"] < r["macd_signal"]
    ):
        return "SHORT"

    return "NEUTRAL"

# Son taramadaki sinyal eşiğine yaklaşan adaylar
candidate_lock = threading.Lock()
last_candidates = []

# ============================================================
# ANALİZ
# ============================================================

def analyze(item, btc, diagnostic=False):
    symbol = item["symbol"]
    base = item["base"]

    df15 = klines(symbol, "15m", 220)
    if df15 is None:
        return None

    time.sleep(0.08)

    df1h = klines(symbol, "1h", 220)
    if df1h is None:
        return None

    try:
        f15 = features(df15)
        f1h = features(df1h)
        row = f15.iloc[-2]

        price = float(row["close"])
        atr = float(row["atr"])

        if math.isnan(price) or math.isnan(atr) or atr <= 0:
            return None

        long_score, long_reasons = technical_score(f15, f1h, "LONG")
        short_score, short_reasons = technical_score(f15, f1h, "SHORT")

        if long_score >= short_score:
            direction = "LONG"
            tech_score = long_score
            reasons = list(long_reasons)
        else:
            direction = "SHORT"
            tech_score = short_score
            reasons = list(short_reasons)

        btc_bonus = 0
        if btc == direction:
            btc_bonus = 3
            reasons.append(f"BTC {direction} uyumlu")
        elif btc != "NEUTRAL":
            reasons.append(f"BTC ters yönde ({btc})")

        nscore, news = news_score(base)
        total = min(100, tech_score + btc_bonus + nscore)

        supports, resistances = support_resistance(f15, f1h)
        levels = calculate_trade_levels(
            direction, price, atr, supports, resistances
        )

        blocks = []
        if tech_score < MIN_TECH_SCORE:
            blocks.append(
                f"Teknik skor düşük: {tech_score:.0f} < {MIN_TECH_SCORE:.0f}"
            )
        if total < MIN_SCORE:
            blocks.append(
                f"Toplam skor düşük: {total:.0f} < {MIN_SCORE:.0f}"
            )
        if not levels:
            blocks.append("SL/TP seviyeleri üretilemedi")

        # Teşhis modunda eşik altındaki adayları da döndür.
        if diagnostic:
            result = {
                "symbol": symbol,
                "base": base,
                "direction": direction,
                "score": total,
                "tech_score": tech_score,
                "btc_bonus": btc_bonus,
                "news_score": nscore,
                "price": price,
                "rsi": float(row["rsi"]),
                "adx": float(row["adx"]),
                "vol_ratio": float(row["vol_ratio"]),
                "reasons": reasons,
                "news": news,
                "volume24h": item["volume"],
                "eligible": not blocks,
                "blocks": blocks,
            }
            if levels:
                result.update({
                    "sl": levels["sl"],
                    "tp1": levels["tp1"],
                    "tp2": levels["tp2"],
                    "rr1": levels["rr1"],
                    "rr2": levels["rr2"],
                })
            return result

        if blocks:
            return None

        return {
            "symbol": symbol,
            "base": base,
            "direction": direction,
            "score": total,
            "tech_score": tech_score,
            "btc_bonus": btc_bonus,
            "news_score": nscore,
            "price": price,
            "sl": levels["sl"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            "rr1": levels["rr1"],
            "rr2": levels["rr2"],
            "rsi": float(row["rsi"]),
            "adx": float(row["adx"]),
            "vol_ratio": float(row["vol_ratio"]),
            "reasons": reasons,
            "news": news,
            "volume24h": item["volume"],
        }

    except Exception as e:
        print(f"{symbol} analiz hata:", e)
        return None


def candidates_text(limit=10):
    with candidate_lock:
        items = list(last_candidates[:limit])

    if not items:
        return (
            "🔎 Henüz aday listesi yok.\n"
            "Önce /tara komutunu çalıştır."
        )

    text = "🔎 EN YAKIN ADAYLAR\n\n"
    for i, x in enumerate(items, 1):
        state = "✅ SİNYAL" if x.get("eligible") else "⏳ YAKIN"
        text += (
            f"{i}. {x['symbol']} {x['direction']} — "
            f"{x['score']:.0f}/100 "
            f"(Teknik {x['tech_score']:.0f}) {state}\n"
        )
        if x.get("blocks"):
            text += "   ❌ " + " | ".join(x["blocks"][:2]) + "\n"
        text += (
            f"   RSI {x['rsi']:.1f} | ADX {x['adx']:.1f} | "
            f"Hacim {x['vol_ratio']:.2f}x\n"
        )

    text += (
        "\nℹ️ Aday listesi işlem sinyali değildir; "
        "neden elendiğini görmek içindir."
    )
    return text


# ============================================================
# FİYAT FORMAT
# ============================================================

def fmt_price(p):
    if p >= 1000:
        return f"{p:.2f}"

    if p >= 100:
        return f"{p:.3f}"

    if p >= 1:
        return f"{p:.4f}"

    if p >= 0.1:
        return f"{p:.5f}"

    if p >= 0.01:
        return f"{p:.6f}"

    return f"{p:.8f}"

# ============================================================
# SİNYAL MESAJI
# ============================================================

def signal_message(x):
    direction_icon = (
        "🟢"
        if x["direction"] == "LONG"
        else "🔴"
    )

    risk_pct = (
        abs(x["price"] - x["sl"])
        / x["price"]
        * 100
    )

    tp1_pct = (
        abs(x["tp1"] - x["price"])
        / x["price"]
        * 100
    )

    tp2_pct = (
        abs(x["tp2"] - x["price"])
        / x["price"]
        * 100
    )

    reasons = "\n".join(
        f"• {r}"
        for r in x["reasons"][:7]
    )

    msg = (
        f"🚨 RADAR V5 SİNYALİ\n\n"
        f"{direction_icon} "
        f"{x['symbol']} "
        f"{x['direction']}\n\n"

        f"⭐ TOPLAM: "
        f"{x['score']:.0f}/100\n"

        f"📊 Teknik: "
        f"{x['tech_score']:.0f}/90\n"

        f"📰 Haber: "
        f"+{x['news_score']:.0f}\n\n"

        f"💰 Giriş: "
        f"{fmt_price(x['price'])}\n"

        f"🛑 SL: "
        f"{fmt_price(x['sl'])} "
        f"({risk_pct:.2f}%)\n"

        f"🎯 TP1: "
        f"{fmt_price(x['tp1'])} "
        f"({tp1_pct:.2f}%) "
        f"R:R {x['rr1']:.2f}\n"

        f"🎯 TP2: "
        f"{fmt_price(x['tp2'])} "
        f"({tp2_pct:.2f}%) "
        f"R:R {x['rr2']:.2f}\n\n"

        f"📈 RSI: "
        f"{x['rsi']:.1f}\n"

        f"⚡ Hacim: "
        f"{x['vol_ratio']:.2f}x\n"

        f"💪 ADX: "
        f"{x['adx']:.1f}\n"

        f"💵 24s Futures hacim: "
        f"${x['volume24h']/1_000_000:.1f}M\n\n"

        f"🔎 Neden:\n"
        f"{reasons}"
    )

    news = x.get("news")

    if news:
        age_text = "zaman bilinmiyor"

        if news["age"] is not None:
            if news["age"] < 1:
                age_text = (
                    f"{int(news['age'] * 60)} dk önce"
                )
            else:
                age_text = (
                    f"{news['age']:.1f} saat önce"
                )

        msg += (
            f"\n\n📰 KATALİZÖR\n"
            f"{news['source']} | "
            f"{age_text}\n"
            f"{news['title']}"
        )

        if news.get("link"):
            msg += (
                f"\n{news['link']}"
            )

    else:
        msg += (
            "\n\n📰 Coin için güçlü yeni "
            "haber katalizörü bulunmadı."
        )

    msg += (
        "\n\n⚠️ Radar skoru gerçekleşmiş "
        "kazanma oranı değildir."
    )

    return msg


# ============================================================
# SANAL İŞLEM MOTORU
# ============================================================

def default_paper_state():
    return {
        "start_balance": PAPER_START_BALANCE,
        "balance": PAPER_START_BALANCE,
        "peak_balance": PAPER_START_BALANCE,
        "max_drawdown_pct": 0.0,
        "open": [],
        "history": []
    }


def load_paper_state():
    try:
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = default_paper_state()
            base.update(data)
            base.setdefault("open", [])
            base.setdefault("history", [])
            return base
    except Exception as e:
        print("Paper state okuma hata:", e)

    return default_paper_state()


paper_state = load_paper_state()


def save_paper_state():
    try:
        tmp = PAPER_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(paper_state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PAPER_FILE)
    except Exception as e:
        print("Paper state kaydetme hata:", e)


def update_drawdown():
    bal = float(paper_state["balance"])
    paper_state["peak_balance"] = max(
        float(paper_state.get("peak_balance", bal)),
        bal
    )
    peak = float(paper_state["peak_balance"])
    if peak > 0:
        dd = (peak - bal) / peak * 100
        paper_state["max_drawdown_pct"] = max(
            float(paper_state.get("max_drawdown_pct", 0)),
            dd
        )


def paper_has_symbol(symbol):
    return any(
        p.get("symbol") == symbol
        for p in paper_state["open"]
    )


def paper_open_trade(signal):
    with paper_lock:
        if paper_has_symbol(signal["symbol"]):
            return False

        if len(paper_state["open"]) >= PAPER_MAX_OPEN:
            return False

        entry = float(signal["price"])
        sl = float(signal["sl"])
        risk_per_unit = abs(entry - sl)

        if entry <= 0 or risk_per_unit <= 0:
            return False

        balance = float(paper_state["balance"])
        risk_usdt = balance * PAPER_RISK_PERCENT / 100.0
        qty = risk_usdt / risk_per_unit
        notional = qty * entry

        # Açılış taker komisyonu
        entry_fee = notional * PAPER_TAKER_FEE
        paper_state["balance"] -= entry_fee

        trade = {
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "score": float(signal["score"]),
            "entry": entry,
            "sl": sl,
            "original_sl": sl,
            "tp1": float(signal["tp1"]),
            "tp2": float(signal["tp2"]),
            "qty": qty,
            "remaining_qty": qty,
            "risk_usdt": risk_usdt,
            "entry_fee": entry_fee,
            "realized_pnl": -entry_fee,
            "tp1_hit": False,
            "opened_at": int(time.time()),
            "last_kline_open": 0
        }

        paper_state["open"].append(trade)
        update_drawdown()
        save_paper_state()

    telegram(
        "🧪 SANAL İŞLEM AÇILDI\n\n"
        f"{trade['symbol']} {trade['direction']}\n"
        f"Giriş: {fmt_price(trade['entry'])}\n"
        f"SL: {fmt_price(trade['sl'])}\n"
        f"TP1: {fmt_price(trade['tp1'])}\n"
        f"TP2: {fmt_price(trade['tp2'])}\n"
        f"Risk: {risk_usdt:.2f} USDT (%{PAPER_RISK_PERCENT:.1f})\n"
        f"Sanal bakiye: {paper_state['balance']:.2f} USDT"
    )
    return True


def paper_realize(trade, exit_price, qty, label):
    if qty <= 0:
        return 0.0

    if trade["direction"] == "LONG":
        gross = (exit_price - trade["entry"]) * qty
    else:
        gross = (trade["entry"] - exit_price) * qty

    fee = abs(exit_price * qty) * PAPER_TAKER_FEE
    net = gross - fee

    paper_state["balance"] += net
    trade["realized_pnl"] += net
    trade["remaining_qty"] = max(0.0, trade["remaining_qty"] - qty)

    return net


def close_paper_trade(trade, exit_price, label):
    qty = float(trade["remaining_qty"])
    paper_realize(trade, exit_price, qty, label)

    trade["closed_at"] = int(time.time())
    trade["exit"] = float(exit_price)
    trade["result"] = label

    paper_state["history"].append(dict(trade))
    paper_state["open"].remove(trade)

    update_drawdown()
    save_paper_state()

    icon = "✅" if trade["realized_pnl"] > 0 else "❌"
    telegram(
        f"{icon} SANAL İŞLEM KAPANDI\n\n"
        f"{trade['symbol']} {trade['direction']}\n"
        f"Sonuç: {label}\n"
        f"Net PNL: {trade['realized_pnl']:+.2f} USDT\n"
        f"Bakiye: {paper_state['balance']:.2f} USDT"
    )


def paper_process_candle(trade, high, low):
    """
    Aynı 1 dakikalık mum içinde hem SL hem hedef görülürse
    performansı şişirmemek için kötü senaryo (SL önce) uygulanır.
    """
    direction = trade["direction"]
    sl = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])

    if direction == "LONG":
        sl_hit = low <= sl
        tp1_hit_now = (not trade["tp1_hit"]) and high >= tp1
        tp2_hit = high >= tp2
    else:
        sl_hit = high >= sl
        tp1_hit_now = (not trade["tp1_hit"]) and low <= tp1
        tp2_hit = low <= tp2

    # Kötümser sıra: aynı mumda stop da görülmüşse stop önce.
    if sl_hit:
        label = "BE" if trade["tp1_hit"] and abs(sl - trade["entry"]) < 1e-12 else "SL"
        close_paper_trade(trade, sl, label)
        return True

    # TP1 görülmeden aynı mumda TP2 de görülürse:
    # önce yarısı TP1, sonra kalan yarısı TP2.
    if tp1_hit_now:
        half = float(trade["remaining_qty"]) * 0.5
        paper_realize(trade, tp1, half, "TP1")
        trade["tp1_hit"] = True
        trade["sl"] = float(trade["entry"])  # kalan yarı breakeven
        save_paper_state()

        telegram(
            "🧪 SANAL TP1\n\n"
            f"{trade['symbol']} {trade['direction']}\n"
            "Pozisyonun %50'si kapandı.\n"
            f"SL girişe taşındı: {fmt_price(trade['entry'])}\n"
            f"Gerçekleşen PNL: {trade['realized_pnl']:+.2f} USDT"
        )

        if tp2_hit:
            close_paper_trade(trade, tp2, "TP2")
            return True

    elif trade["tp1_hit"] and tp2_hit:
        close_paper_trade(trade, tp2, "TP2")
        return True

    return False


def update_paper_positions():
    with paper_lock:
        positions = list(paper_state["open"])

    for trade in positions:
        try:
            data = get_json(
                BASE + "/fapi/v1/klines",
                {
                    "symbol": trade["symbol"],
                    "interval": "1m",
                    "limit": 20
                }
            )

            if not data:
                continue

            opened_ms = int(trade["opened_at"] * 1000)
            last_seen = int(trade.get("last_kline_open", 0))

            for k in data:
                open_ms = int(k[0])

                if open_ms < opened_ms or open_ms <= last_seen:
                    continue

                high = float(k[2])
                low = float(k[3])

                with paper_lock:
                    # Pozisyon önceki mumda kapanmış olabilir
                    if trade not in paper_state["open"]:
                        break

                    closed = paper_process_candle(trade, high, low)
                    trade["last_kline_open"] = open_ms
                    save_paper_state()

                if closed:
                    break

            time.sleep(0.05)

        except Exception as e:
            print(trade.get("symbol"), "paper takip hata:", e)


def paper_performance():
    with paper_lock:
        hist = list(paper_state["history"])
        opened = list(paper_state["open"])
        bal = float(paper_state["balance"])
        start = float(paper_state["start_balance"])
        max_dd = float(paper_state.get("max_drawdown_pct", 0))

    wins = [x for x in hist if float(x.get("realized_pnl", 0)) > 0]
    losses = [x for x in hist if float(x.get("realized_pnl", 0)) < 0]

    total = len(hist)
    win_rate = (len(wins) / total * 100) if total else 0.0
    gross_profit = sum(float(x.get("realized_pnl", 0)) for x in wins)
    gross_loss = abs(sum(float(x.get("realized_pnl", 0)) for x in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    net = bal - start

    pf_text = "∞" if pf >= 999 else f"{pf:.2f}"

    return (
        "📊 SANAL RADAR PERFORMANSI\n\n"
        f"Başlangıç: {start:.2f} USDT\n"
        f"Bakiye: {bal:.2f} USDT\n"
        f"Net PNL: {net:+.2f} USDT\n\n"
        f"Toplam kapanan: {total}\n"
        f"Kazanan: {len(wins)}\n"
        f"Kaybeden: {len(losses)}\n"
        f"Win Rate: %{win_rate:.1f}\n"
        f"Profit Factor: {pf_text}\n"
        f"Max Drawdown: %{max_dd:.2f}\n\n"
        f"Açık sanal işlem: {len(opened)}\n"
        f"İşlem başı risk: %{PAPER_RISK_PERCENT:.1f}"
    )


def paper_positions_text():
    with paper_lock:
        opened = list(paper_state["open"])

    if not opened:
        return "🧪 Açık sanal işlem yok."

    lines = ["🧪 AÇIK SANAL İŞLEMLER\n"]

    for p in opened:
        state = "TP1 ✅ / SL=BE" if p.get("tp1_hit") else "TP1 bekliyor"
        lines.append(
            f"{p['symbol']} {p['direction']} | "
            f"Giriş {fmt_price(p['entry'])} | "
            f"SL {fmt_price(p['sl'])} | "
            f"TP1 {fmt_price(p['tp1'])} | "
            f"TP2 {fmt_price(p['tp2'])} | {state}"
        )

    return "\n".join(lines)


def paper_history_text(limit=10):
    with paper_lock:
        hist = list(paper_state["history"])[-limit:]

    if not hist:
        return "📚 Henüz kapanmış sanal işlem yok."

    lines = [f"📚 SON {len(hist)} SANAL İŞLEM\n"]

    for p in reversed(hist):
        pnl = float(p.get("realized_pnl", 0))
        lines.append(
            f"{p['symbol']} {p['direction']} | "
            f"{p.get('result', '-')} | {pnl:+.2f} USDT"
        )

    return "\n".join(lines)


# ============================================================
# COOLDOWN
# ============================================================

def can_alert(symbol, direction):
    key = (
        symbol,
        direction
    )

    last = alerts.get(
        key,
        0
    )

    if (
        time.time() - last
        < ALERT_COOLDOWN
    ):
        return False

    alerts[key] = time.time()

    return True

# ============================================================
# MARKET TARAMA
# ============================================================

def scan_market(manual=False):
    global last_scan_time
    global last_scan_duration
    global last_scan_count
    global last_signal_count
    global last_candidates

    if not scan_lock.acquire(
        blocking=False
    ):
        if manual:
            telegram(
                "⏳ Tarama zaten devam ediyor."
            )
        return

    started = time.time()

    try:
        # Önce açık sanal işlemlerin TP/SL durumunu kontrol et
        update_paper_positions()

        update_news()

        coins = get_symbols()

        last_scan_count = len(coins)

        print(
            f"Taranacak coin: {len(coins)}"
        )

        if not coins:
            print("Coin listesi alınamadı.")
            return

        btc = btc_state()

        print(
            f"BTC durum: {btc}"
        )

        signals = []
        candidates = []

        for index, item in enumerate(
            coins,
            start=1
        ):
            symbol = item["symbol"]

            print(
                f"[{index}/{len(coins)}] "
                f"{symbol}"
            )

            try:
                result = analyze(
                    item,
                    btc,
                    diagnostic=True
                )

                if result:
                    candidates.append(result)

                if result and result.get("eligible"):
                    signals.append(result)

                    if can_alert(
                        result["symbol"],
                        result["direction"]
                    ):
                        telegram(
                            signal_message(result)
                        )

                        # Gerçek emir YOK. Aynı sinyali sanal işlem olarak aç.
                        paper_open_trade(result)

                        print(
                            "SİNYAL:",
                            result["symbol"],
                            result["direction"],
                            result["score"]
                        )

            except Exception as e:
                print(
                    symbol,
                    "tarama hata:",
                    e
                )

            # Binance'e yük bindirmemek için
            time.sleep(0.12)

        # En yüksek skorlular
        signals.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["tech_score"],
                x["volume24h"]
            ),
            reverse=True
        )

        with candidate_lock:
            last_candidates = candidates[:20]

        last_signal_count = len(
            signals
        )

        if manual:
            if signals:
                top = signals[:5]

                text = (
                    "🔍 MANUEL TARAMA BİTTİ\n\n"
                )

                for s in top:
                    text += (
                        f"{s['symbol']} "
                        f"{s['direction']} "
                        f"{s['score']:.0f}/100\n"
                    )

                telegram(text)
                telegram(candidates_text(5))

            else:
                telegram(
                    "🔍 Tarama bitti.\n"
                    "Şu anda 80+ uygun sinyal bulunamadı.\n\n"
                    + candidates_text(5)
                )

    finally:
        last_scan_time = time.time()

        last_scan_duration = (
            time.time() - started
        )

        scan_lock.release()

        print(
            f"Tarama tamamlandı: "
            f"{last_scan_duration:.1f} sn"
        )

# ============================================================
# DURUM
# ============================================================

def status():
    if last_scan_time:
        last_scan = datetime.fromtimestamp(
            last_scan_time
        ).strftime(
            "%H:%M:%S"
        )
    else:
        last_scan = "Henüz yok"

    with news_lock:
        news_count = len(
            news_cache
        )

    return (
        "🤖 COIN RADAR V5.2 DIAGNOSTIC\n\n"

        f"🟢 Durum: Çalışıyor\n"
        f"⏱ Uptime: {uptime()}\n\n"

        f"⭐ Minimum skor: {MIN_SCORE:.0f}\n"
        f"📊 Min teknik: {MIN_TECH_SCORE:.0f}\n"

        f"💵 Min hacim: "
        f"${MIN_VOLUME/1_000_000:.0f}M\n"

        f"🔄 Tarama aralığı: "
        f"{SCAN_INTERVAL} sn\n\n"

        f"🕒 Son tarama: {last_scan}\n"

        f"⏳ Tarama süresi: "
        f"{last_scan_duration:.1f} sn\n"

        f"🪙 Taranan: "
        f"{last_scan_count}\n"

        f"🚨 Uygun sinyal: "
        f"{last_signal_count}\n"

        f"📰 Haber havuzu: "
        f"{news_count}\n\n"

        "📈 Timeframe: 15m + 1h\n"
        "🎯 TP: Destek/direnç + ATR\n"
        "🛑 SL: Teknik + ATR, max %2\n"
        "📰 Haber bonusu: max +10\n"
        f"🧪 Sanal bakiye: {paper_state['balance']:.2f} USDT\n"
        f"🧪 Açık sanal işlem: {len(paper_state['open'])}"
    )

# ============================================================
# TELEGRAM KOMUTLARI
# ============================================================

def telegram_commands():
    if not TOKEN:
        print(
            "Telegram token yok."
        )
        return

    offset = None

    while True:
        try:
            params = {
                "timeout": 25
            }

            if offset is not None:
                params["offset"] = offset

            url = (
                f"https://api.telegram.org/"
                f"bot{TOKEN}/getUpdates"
            )

            r = session.get(
                url,
                params=params,
                timeout=35
            )

            if r.status_code == 409:
                print(
                    "Telegram 409: "
                    "Bu bot tokenini başka "
                    "bir süreç de kullanıyor."
                )

                time.sleep(10)
                continue

            data = r.json()

            if not data.get("ok"):
                time.sleep(3)
                continue

            for update in data.get(
                "result",
                []
            ):
                offset = (
                    update["update_id"]
                    + 1
                )

                message = update.get(
                    "message",
                    {}
                )

                chat_id = str(
                    message.get(
                        "chat",
                        {}
                    ).get(
                        "id",
                        ""
                    )
                )

                # Sadece kendi chat ID
                if CHAT and chat_id != str(CHAT):
                    continue

                text = (
                    message.get(
                        "text",
                        ""
                    )
                    .strip()
                    .lower()
                )

                if text == "/durum":
                    telegram(
                        status()
                    )

                elif text == "/performans":
                    telegram(
                        paper_performance()
                    )

                elif text == "/pozisyonlar":
                    telegram(
                        paper_positions_text()
                    )

                elif text == "/gecmis":
                    telegram(
                        paper_history_text()
                    )

                elif text == "/adaylar":
                    telegram(
                        candidates_text(10)
                    )

                elif text == "/tara":
                    telegram(
                        "🔍 Manuel tarama başlatıldı..."
                    )

                    threading.Thread(
                        target=scan_market,
                        kwargs={
                            "manual": True
                        },
                        daemon=True
                    ).start()

                elif text in [
                    "/start",
                    "/yardim"
                ]:
                    telegram(
                        "🤖 COIN RADAR V5.2 DIAGNOSTIC\n\n"
                        "/durum - Bot durumu\n"
                        "/tara - Manuel tarama\n"
                        "/performans - Sanal performans\n"
                        "/pozisyonlar - Açık sanal işlemler\n"
                        "/gecmis - Son 10 sanal işlem\n"
                        "/yardim - Komutlar\n\n"
                        "Minimum sinyal: 80/100\n"
                        "Teknik + Haber + BTC filtresi"
                    )

        except Exception as e:
            print(
                "Telegram polling hata:",
                e
            )

            time.sleep(5)

# ============================================================
# RADAR LOOP
# ============================================================

def radar_loop():
    while True:
        try:
            scan_market()

        except Exception as e:
            print(
                "Radar loop hata:",
                e
            )

        time.sleep(
            SCAN_INTERVAL
        )

# ============================================================
# RUN
# ============================================================

def run():
    print(
        "Coin Radar V5.2 Diagnostic başlıyor..."
    )

    update_news()

    telegram(
        "🟢 Coin Radar V5.2 DIAGNOSTIC BAŞLADI\n\n"
        "⭐ Minimum skor: 80\n"
        "📊 15m + 1h teknik analiz\n"
        "📰 Çoklu haber takibi\n"
        "⚡ Breakout / Retest\n"
        "🎯 Dinamik TP / SL\n"
        "💵 Min Futures hacim: $10M\n\n"
        "🧪 Sanal işlem modu AÇIK\n"
        "💰 Başlangıç sanal bakiye: 200 USDT\n"
        "⚠️ Gerçek Binance emri KAPALI."
    )

    radar_thread = threading.Thread(
        target=radar_loop,
        daemon=True
    )

    radar_thread.start()

    telegram_commands()


if __name__ == "__main__":
    run()
