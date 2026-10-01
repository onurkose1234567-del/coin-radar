# Coin Radar v1
Binance USDT perpetual piyasasında 15m + 1h teknik radar. LONG/SHORT skorlar; RSI, EMA, MACD, hacim anomalisi, ATR, ADX/DI, Bollinger konumu, taker buy/sell oranı, breakout/breakdown ve BTC filtresi kullanır.

## Kurulum
1. Python 3.10+ kurun.
2. `pip install -r requirements.txt`
3. `.env.example` dosyasını `.env` adıyla kopyalayın.
4. Telegram'da @BotFather ile bot oluşturup tokenı `TELEGRAM_BOT_TOKEN` alanına, kendi chat ID'nizi `TELEGRAM_CHAT_ID` alanına yerel olarak yazın. Tokenı kimseyle paylaşmayın.
5. `python radar.py`

## Önemli
`MIN_SCORE=80`, %80 kazanma oranı anlamına gelmez; teknik koşulların 100 üzerinden radar skorudur. Gerçek hit-rate için ayrı backtest/walk-forward modülü gerekir. Bu v1 otomatik emir göndermez.
