# 🚀 Enhanced Crypto Trading Bot with Volume Validation

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Binance](https://img.shields.io/badge/Exchange-Binance-yellow.svg)](https://binance.com)
[![Telegram](https://img.shields.io/badge/Control-Telegram-blue.svg)](https://telegram.org)

> **A sophisticated cryptocurrency trading bot that prevents "falling knife" entries through advanced volume analysis**

## ⚠️ Important Disclaimer

**This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. You are solely responsible for your trading decisions and any losses incurred. Always test on testnet first.**

---

## 🌟 What Makes This Bot Special

This isn't just another trading bot. 

## 🚀 Key Features

### 🧠 Smart Trading Strategies
- **Trend Following** - Catches pullbacks in established uptrends
- **Mean Reversion** - Capitalizes on oversold bounces
- **Breakout Detection** - Rides momentum with proper volume confirmation

### 🛡️ Advanced Risk Management
- **ATR-Based Dynamic Stop Losses** - Adapts to market volatility
- **Trailing Stop Loss** - Locks in profits automatically
- **Dual Circuit Breakers** - Prevents catastrophic losses
  - **Portfolio Circuit Breaker** - Stops trading if portfolio drops below 80% of initial value
  - **Drawdown Circuit Breaker** - Triggers if drawdown exceeds 20% from peak
- **Position Sizing** - Risk-adjusted allocation per trade
- **Maximum Concurrent Positions** - Prevents over-exposure

### 📱 Full Telegram Control
- Real-time monitoring and control
- Manual trading capabilities
- Live chart generation
- Portfolio tracking
- Strategy performance analysis

### 🔧 Production-Ready Features
- Comprehensive error handling
- Detailed logging system
- Testnet and live trading support
- Concurrent position management
- API rate limiting

## 📊 Strategy Performance Highlights

The enhanced volume validation prevents entries during:
- Market crashes with high selling volume
- False breakouts with panic selling
- Institutional dumping disguised as "high activity"

### 🔥 The "Falling Knife" Problem
Most bots use simple `volume > average` validation, which can't distinguish between:
- ✅ **Buying pressure** (institutional accumulation, breakout confirmation)
- ❌ **Selling pressure** (panic selling, institutional dumping)

### 🎯 Our Solution: Advanced Volume Validation
- **Volume-Price Direction Analysis** - Distinguishes buying vs selling pressure
- **On-Balance Volume (OBV) Confirmation** - Ensures institutional money flow alignment
- **Anti-Panic Protection** - Filters out extremely high volume spikes that indicate panic
- **Multi-Period Validation** - Confirms trends across multiple timeframes

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Binance account with API keys
- Telegram bot token

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/hellcatxrp/hybrid-trading-bot.git
cd hybrid-trading-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Configuration
Create a .env file:

BINANCE_ENV=TESTNET
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_SECRET_KEY=your_testnet_secret
TELEGRAM_BOT_TOKEN_2=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

4. Run #Within your activated virtual environment
python trading_bot.py  // you can rename the .py file to whatever you desire. Keep it secret,keep it safe.

## 📋 Telegram Commands

### 🔍 Monitoring & Information
| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Show welcome message and all available commands | `/start` |
| `/status` | Current positions, bot status, drawdown, watchlist | `/status` |
| `/balance` | Portfolio breakdown and P&L vs baseline | `/balance` |
| `/criteria` | Check entry conditions for all monitored symbols | `/criteria` |
| `/chart SYMBOL` | Generate 15M chart with indicators (24h) | `/chart HBARUSDC` |

### 💰 Trading Commands
| Command | Description | Example |
|---------|-------------|---------|
| `/buy SYMBOL AMOUNT [STRATEGY]` | Manual market buy order | `/buy SUIUSDC 100 breakout` |
| `/forcesell SYMBOL` | Immediately sell specific position | `/forcesell XRPUSDC` |

### 📝 Watchlist Management
| Command | Description | Example |
|---------|-------------|---------|
| `/include SYMBOL` | Add symbol to active watchlist | `/include VIRTUALUSDC` |
| `/exclude SYMBOL` | Remove symbol from monitoring | `/exclude LINKUSDC` |

### ⚙️ Bot Control
| Command | Description | ⚠️ Warning |
|---------|-------------|------------|
| `/pause` | Pause new entries (keep managing positions) | Safe - preserves positions |
| `/resume` | Resume automatic trading | Safe - restarts scanning |
| `/stop` | **LIQUIDATE ALL + SHUTDOWN** | ⚠️ **DANGER**: Sells everything! |

### 📊 Strategy Options for /buy Command
- `trend` - Uses Trend strategy TP/SL (2.0%/2.7%)
- `meanrevert` - Uses Mean Reversion TP/SL (2.0%/2.7%) 
- `breakout` - Uses Breakout strategy TP/SL (3.0%/4.5%)
- No strategy specified - Defaults to Manual (uses Breakout settings)

### 💡 Watchlist Management Examples

#### Adding New Symbols

/include DOGEUSDC     # Adds DOGE to monitoring
/include ADAUSDC      # Adds ADA to monitoring  
/include MATICUSDC    # Adds MATIC to monitoring

🎯 Trading Strategies Explained
Trend Strategy
Entry: Price > 200 EMA, RSI ≤ 30, Price < Lower BB, Enhanced Volume
Goal: Catch pullbacks in strong uptrends
Risk: Lower (conservative entries)
Mean Reversion Strategy
Entry: RSI < 26, Price < 50 EMA, Enhanced Volume
Goal: Bounce trades from oversold levels
Risk: Medium (counter-trend)
Breakout Strategy
Entry: Price > 12H high, ADX > 25, RSI > 55, Enhanced Volume
Goal: Ride momentum breakouts
Risk: Higher (early trend detection)
🛠️ Advanced Configuration
Position Sizing
TREND_POSITION_SIZE = Decimal('0.10')      # 10% of balance
MEAN_REVERT_POSITION_SIZE = Decimal('0.10') # 10% of balance  
BREAKOUT_POSITION_SIZE = Decimal('0.15')    # 15% of balance
Risk Management
USE_ATR_SIZING = True                       # Dynamic stop losses
TRAILING_STOP_ACTIVATION_PERCENT = 1.5      # TSL activates at 1.5% profit
LOSS_CIRCUIT_BREAKER_THRESHOLD = 0.8       # Stop at 20% portfolio loss