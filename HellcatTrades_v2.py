import os
import time
import datetime
import asyncio
import collections
import logging
import sys
from decimal import Decimal, ROUND_DOWN, DecimalException, InvalidOperation
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import mplfinance as mpf
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
import io
from telegram import InputFile
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance import AsyncClient
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from typing import Dict, List, DefaultDict, Any, Optional, Tuple

# --- Configuration & Constants ---
load_dotenv()
BINANCE_ENV = os.environ.get('BINANCE_ENV', 'TESTNET').upper()
IS_TESTNET = (BINANCE_ENV == 'TESTNET')

# Binance API Keys
if IS_TESTNET:
    BINANCE_API_KEY = os.environ.get('BINANCE_TESTNET_API_KEY')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_TESTNET_SECRET_KEY')
    print("INFO: Using TESTNET Binance API Keys.")
else:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY')
    print("INFO: Using LIVE Binance API Keys.")

# Telegram
TELEGRAM_BOT_TOKEN_2 = os.environ.get('TELEGRAM_BOT_TOKEN_2')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Trading Parameters
QUOTE_ASSET = 'USDC'
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_PERIOD = 250
CHART_LOOKBACK_PERIOD = 96  # 24 hours of 15-minute candles (4 candles/hour * 24 hours)
SLEEP_INTERVAL_SECONDS = int(os.environ.get('SLEEP_INTERVAL_SECONDS', '60'))
MAX_CONCURRENT_POSITIONS = 7  
PRICE_DISCREPANCY_ALERT_THRESHOLD_PERCENT = Decimal('0.5') # Log if live ticker and kline close differ by this %
WATCHLIST = ['HBARUSDC', 'LINKUSDC', 'AVAXUSDC', 'SUIUSDC', 'XRPUSDC', 'MUBARAKUSDC', 'VIRTUALUSDC']  # Default watchlist - configure to your preference

# Enhanced Risk Management
LOSS_CIRCUIT_BREAKER_THRESHOLD = Decimal('0.8')  # Pause if balance drops below 80% of initial
DRAWDOWN_CIRCUIT_BREAKER = Decimal('20.0')  # NEW: Pause if drawdown exceeds 20%
MAX_EQUITY_DRAWDOWN_TRACKING = 100  # NEW: Number of recent equity points to track for drawdown
BREAKOUT_COOLDOWN_HOURS = 12 # Cooldown period in hours before another breakout trade on the same symbol

# Position Sizing & Risk
USE_ATR_SIZING = True  # NEW: Set to True to use ATR based sizing

# NEW: Strategy-Specific Position Sizing
TREND_POSITION_SIZE = Decimal('0.10')  # 10% - set these all according to your own preference
MEAN_REVERT_POSITION_SIZE = Decimal('0.10')  # 10% 
BREAKOUT_POSITION_SIZE = Decimal('0.15')  # 15% 

# NEW: Trailing Stop Settings
TRAILING_STOP_ACTIVATION_PERCENT = Decimal('1.5')  # Activate trailing stop after 1% profit
TRAILING_STOP_LOCKIN_PROFIT_PERCENT = Decimal('1.0') # NEW: Set TSL to lock in this % profit from entry when activated *can not prevent slippage

# Strategy Specific TP/SL Percentages (from entry price)
TREND_TP = Decimal('2.0')  # OPTIMIZED
TREND_SL = Decimal('2.7')  # OPTIMIZED
MEAN_REVERT_TP = Decimal('2.0')  # OPTIMIZED
MEAN_REVERT_SL = Decimal('2.7')  # OPTIMIZED
BREAKOUT_TP = Decimal('3.0')  # OPTIMIZED
BREAKOUT_SL = Decimal('4.5')  # OPTIMIZED

# NEW: Enhanced Volume Validation Parameters
VOLUME_PRICE_CORRELATION_THRESHOLD = Decimal('0.002')  # 0.2% price change threshold
VOLUME_RATIO_THRESHOLD = Decimal('1.2')  # Volume must be 20% above SMA
OBV_CONFIRMATION_PERIODS = 3  # OBV must be rising for this many periods

# API and Order Handling
API_REQUESTS_PER_MINUTE = 2400
API_REQUEST_INTERVAL = 60 / API_REQUESTS_PER_MINUTE
ORDER_RETRY_DELAY_SECONDS = 2
MAX_ORDER_RETRIES = 2
ORDER_STATUS_POLL_INTERVAL = 1
ORDER_STATUS_TIMEOUT_SECONDS = 15
CACHE_EXPIRY_SECONDS = 3600
KLINE_CACHE_DURATION = 55  

# Strategy Parameters
EMA_SHORT_PERIOD = 20
EMA_MEDIUM_PERIOD = 50
EMA_LONG_PERIOD = 200
RSI_PERIOD = 14
RSI_OVERSOLD = Decimal('26.0')  # Configure at your discretion
RSI_EXIT = Decimal('55.0')  
ADX_PERIOD = 14
ADX_BREAKOUT_THRESHOLD = Decimal('25.0')  # Configure at your discretion
VOLUME_SMA_PERIOD = 20
ATR_PERIOD = 14
BBANDS_PERIOD = 20  # For Bollinger Bands
BBANDS_STD = 2  # Standard deviations for Bollinger Bands

# Emojis
EMOJI_START = "🚀"; EMOJI_STOP = "🛑"; EMOJI_PAUSE = "⏸️"; EMOJI_RESUME = "▶️"
EMOJI_STATUS = "📊"; EMOJI_CRITERIA = "🔍"; EMOJI_EXCLUDE = "🚫"; EMOJI_INCLUDE = "➕"
EMOJI_BALANCE = "🏦"; EMOJI_BUY = "🟢"; EMOJI_SELL_TP = "📈"; EMOJI_SELL_SL = "📉"
EMOJI_ERROR = "⚠️"; EMOJI_WARNING = "🔔"; EMOJI_INFO = "ℹ️"; EMOJI_CHECK = "✅"
EMOJI_CROSS = "❌"; EMOJI_WAIT = "⏳"; EMOJI_PROFIT = "💰"; EMOJI_LOSS = "💸"
EMOJI_CIRCUIT = "🔌"; EMOJI_CHART = "📊"; EMOJI_ATR = "📏"; EMOJI_TRAILING = "🔄" 
EMOJI_VOLUME = "📊"  # NEW: Volume validation emoji

# Global State
state: Dict[str, Any] = {
    "is_running": True,
    "is_paused": False,
    "last_error": None,
    "open_position_count": 0,
    "positions": collections.defaultdict(dict),
    "excluded_symbols": [],
    "watchlist_symbols": WATCHLIST.copy(),
    "initial_usdc_balance": Decimal('0'),
    "initial_portfolio_value_baseline": None,
    "equity_history": [],  
    "max_equity_seen": Decimal('0'),  
    "dca_history": collections.defaultdict(dict),  
    "last_breakout_times_by_symbol": collections.defaultdict(float) 
}
state_lock = asyncio.Lock()
symbol_info_cache: Dict[str, Dict] = {}
symbol_info_lock = asyncio.Lock()
last_api_request_time = 0
api_request_lock = asyncio.Lock()

# --- Logging Setup ---
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, "trading_bot.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("binance").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# --- Helper Functions ---
async def rate_limit_api():
    """Ensures requests don't exceed API limits."""
    global last_api_request_time
    async with api_request_lock:
        now = time.time()
        elapsed = now - last_api_request_time
        if elapsed < API_REQUEST_INTERVAL:
            await asyncio.sleep(API_REQUEST_INTERVAL - elapsed)
        last_api_request_time = time.time()

async def initialize_binance_client(max_retries: int = 3) -> Optional[AsyncClient]:
    """Initializes the Binance AsyncClient with retries."""
    logging.info(f"Initializing Binance AsyncClient for {BINANCE_ENV}...")
    for attempt in range(1, max_retries + 1):
        try:
            await rate_limit_api()
            client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_SECRET_KEY, testnet=IS_TESTNET)
            await client.ping()
            logging.info("Binance AsyncClient initialized successfully.")
            return client
        except (BinanceAPIException, BinanceRequestException, Exception) as e:
            logging.error(f"Attempt {attempt}/{max_retries} failed to initialize Binance client: {e}")
            if attempt < max_retries:
                await asyncio.sleep(5 * attempt)
    logging.critical("Failed to initialize Binance client after all retries.")
    return None

async def update_symbol_info(client: AsyncClient, symbol: str) -> Dict[str, Any]:
    """Fetches and processes symbol trading rules from Binance."""
    await rate_limit_api()
    try:
        logging.debug(f"[{symbol}] Fetching fresh symbol info...")
        info = await client.get_symbol_info(symbol=symbol)
        if not info:
            logging.warning(f"[{symbol}] No symbol info returned from API.")
            return {}
        rules = {
            'minQty': Decimal('0'),
            'maxQty': Decimal('inf'),
            'stepSize': Decimal('1'),
            'minNotional': Decimal('0'),
            'baseAsset': info.get('baseAsset'),
            'quoteAsset': info.get('quoteAsset')
        }
        found_step_size = False
        for f in info.get('filters', []):
            ft = f['filterType']
            try:
                if ft in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                     rules['minQty'] = max(rules['minQty'], Decimal(f['minQty']))
                     rules['maxQty'] = min(rules['maxQty'], Decimal(f['maxQty']))
                     rules['stepSize'] = Decimal(f['stepSize'])
                     found_step_size = True
                elif ft in ['MIN_NOTIONAL', 'NOTIONAL']:
                    rules['minNotional'] = max(rules['minNotional'], Decimal(f.get('minNotional', f.get('notional', '0'))))
            except (KeyError, InvalidOperation, TypeError) as filter_e:
                 logging.warning(f"[{symbol}] Error processing filter {ft}: {filter_e} - Filter: {f}")
                 continue

        if not found_step_size or rules['stepSize'] <= 0:
             logging.warning(f"[{symbol}] Invalid or missing stepSize. Using default of 1.")
             rules['stepSize'] = Decimal('1')

        return rules
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logging.error(f"[{symbol}] Error fetching/processing symbol info: {e}")
        return {}

async def get_symbol_rules(client: AsyncClient, symbol: str) -> Dict[str, Any]:
    """Gets symbol trading rules, using cache if available."""
    cache_key = f"{symbol}_rules"
    async with symbol_info_lock:
        cached_data = symbol_info_cache.get(cache_key)
        if cached_data and (time.time() - cached_data['timestamp']) < CACHE_EXPIRY_SECONDS:
            logging.debug(f"[{symbol}] Using cached symbol rules.")
            return cached_data['rules']
        
        logging.debug(f"[{symbol}] Cache miss/expired for rules. Fetching...")
        rules = await update_symbol_info(client, symbol)
        if rules:
            symbol_info_cache[cache_key] = {'rules': rules, 'timestamp': time.time()}
            logging.info(f"[{symbol}] Updated symbol rules cache.")
        else:
            logging.warning(f"[{symbol}] Failed to fetch rules, returning empty dict.")
        return rules

def adjust_quantity_to_precision(quantity: Decimal, step_size: Decimal) -> Optional[Decimal]:
    """Adjusts quantity based on the symbol's stepSize filter."""
    try:
        quantity = Decimal(quantity)
        step_size = Decimal(step_size)
        if step_size <= 0:
            logging.error(f"Invalid step_size ({step_size}) in adjust_quantity_to_precision.")
            return None

        adjusted_qty = (quantity // step_size) * step_size
        decimal_places = abs(step_size.normalize().as_tuple().exponent)
        precision_exp = Decimal(f'1e-{decimal_places}')
        formatted_qty = adjusted_qty.quantize(precision_exp, rounding=ROUND_DOWN)

        if formatted_qty == Decimal('0') and quantity > Decimal('0'):
             logging.warning(f"Adjusted quantity became zero. Original: {quantity}, Step: {step_size}")
             return Decimal('0')

        return formatted_qty
    except (InvalidOperation, TypeError, Exception) as e:
        logging.error(f"Error adjusting quantity precision: Q={quantity}, Step={step_size}. Error: {e}", exc_info=True)
        return None

async def get_balance(client: AsyncClient, asset: str) -> Decimal:
    """Fetches the free balance for a given asset."""
    await rate_limit_api()
    try:
        balance_info = await client.get_asset_balance(asset=asset)
        return Decimal(balance_info['free']) if balance_info else Decimal('0')
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logging.error(f"API Error fetching balance {asset}: {e}")
        return Decimal('0')

async def get_latest_data(client: AsyncClient, symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """Fetches latest kline data, using cache."""
    cache_key = f"{symbol}_{interval}_{lookback}"
    async with symbol_info_lock:
        cached_data = symbol_info_cache.get(cache_key)
        if cached_data and (time.time() - cached_data['timestamp']) < KLINE_CACHE_DURATION:
            logging.debug(f"[{symbol}] Using cached kline data for {interval}.")
            return cached_data['data'].copy()

    logging.debug(f"[{symbol}] Cache miss/expired for kline data ({interval}). Fetching...")
    await rate_limit_api()
    try:
        indicator_lookback_needed = max(EMA_LONG_PERIOD, ADX_PERIOD, RSI_PERIOD, VOLUME_SMA_PERIOD, ATR_PERIOD, BBANDS_PERIOD)
        fetch_limit = lookback + indicator_lookback_needed
        logging.debug(f"[{symbol}] Fetching {fetch_limit} klines for {interval}...")

        klines = await client.get_klines(symbol=symbol, interval=interval, limit=fetch_limit)
        if not klines:
             logging.warning(f"[{symbol}] No klines returned from API for {interval}.")
             return pd.DataFrame()

        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[['Open time'] + cols_to_convert]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=cols_to_convert, inplace=True)

        if len(df) < lookback:
            logging.warning(f"[{symbol}] Insufficient valid klines after cleaning: {len(df)} < {lookback}")
            return pd.DataFrame()
        if len(df) < indicator_lookback_needed + 1:
             logging.warning(f"[{symbol}] Insufficient valid klines for indicators: {len(df)} < {indicator_lookback_needed+1}")
             return pd.DataFrame()

        df.rename(columns={'Open time': 'Timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
        df.set_index('Timestamp', inplace=True)

        df_final = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume']]

        async with symbol_info_lock:
             symbol_info_cache[cache_key] = {'data': df_final.copy(), 'timestamp': time.time()}
        logging.debug(f"[{symbol}] Updated kline data cache for {interval}.")
        return df_final

    except (BinanceAPIException, BinanceRequestException) as e:
        if e.code == -1121:
            logging.error(f"API Error fetching klines for {symbol}/{interval}: Invalid Symbol.")
        else:
            logging.error(f"API Error fetching klines for {symbol}/{interval}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching klines: {e}", exc_info=True)
        return pd.DataFrame()

def safe_decimal(value: Any) -> Optional[Decimal]:
    """Safely converts a value to Decimal, handling None, NaN, and errors."""
    if value is None or (isinstance(value, (float, int, str, Decimal)) and pd.isna(value)):
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, DecimalException, InvalidOperation) as e:
        logging.warning(f"safe_decimal failed for value '{value}' type {type(value)}: {e}")
        return None

# --- NEW: Enhanced Volume Analysis Functions ---
def analyze_volume_direction(df: pd.DataFrame, current_idx: int = -1) -> str:
    """
    Analyzes if high volume is associated with buying or selling pressure.
    
    Args:
        df: DataFrame with OHLCV data and Volume_SMA
        current_idx: Index to analyze (default -1 for latest)
        
    Returns:
        "BUYING_VOLUME", "SELLING_VOLUME", "NEUTRAL_VOLUME", or "LOW_VOLUME"
    """
    if len(df) < 2:
        return "INSUFFICIENT_DATA"
        
    try:
        current = df.iloc[current_idx]
        prev = df.iloc[current_idx - 1]
        
        current_price = safe_decimal(current['close'])
        prev_price = safe_decimal(prev['close'])
        current_volume = safe_decimal(current['volume'])
        volume_sma = safe_decimal(current['Volume_SMA'])
        
        if any(v is None for v in [current_price, prev_price, current_volume, volume_sma]):
            return "INSUFFICIENT_DATA"
            
        # Calculate price change percentage
        price_change = (current_price - prev_price) / prev_price
        volume_ratio = current_volume / volume_sma
        
        # Check if volume is significantly elevated
        if volume_ratio > VOLUME_RATIO_THRESHOLD:
            if price_change > VOLUME_PRICE_CORRELATION_THRESHOLD:
                return "BUYING_VOLUME"
            elif price_change < -VOLUME_PRICE_CORRELATION_THRESHOLD:
                return "SELLING_VOLUME"
            else:
                return "NEUTRAL_VOLUME"
        else:
            return "LOW_VOLUME"
            
    except Exception as e:
        logging.warning(f"Error in analyze_volume_direction: {e}")
        return "ERROR"

def validate_obv_trend(df: pd.DataFrame, periods: int = OBV_CONFIRMATION_PERIODS) -> bool:
    """
    Validates that On-Balance Volume is trending upward.
    
    Args:
        df: DataFrame with OBV data
        periods: Number of periods to check for upward trend
        
    Returns:
        True if OBV is trending up, False otherwise
    """
    if len(df) < periods + 1:
        return False
        
    try:
        # Get the last 'periods + 1' OBV values
        obv_values = []
        for i in range(periods + 1):
            obv_val = safe_decimal(df.iloc[-(i+1)]['OBV'])
            if obv_val is None:
                return False
            obv_values.append(obv_val)
        
        # Reverse to get chronological order (oldest to newest)
        obv_values.reverse()
        
        # Check if OBV is generally trending upward
        rising_count = 0
        for i in range(1, len(obv_values)):
            if obv_values[i] > obv_values[i-1]:
                rising_count += 1
                
        # OBV should be rising for at least 60% of the periods
        return rising_count >= (periods * 0.6)
        
    except Exception as e:
        logging.warning(f"Error in validate_obv_trend: {e}")
        return False

def enhanced_volume_validation(df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
    """
    Comprehensive volume validation combining multiple techniques.
    
    Args:
        df: DataFrame with all indicators including OBV
        symbol: Symbol name for logging
        
    Returns:
        (is_valid, reason) tuple
    """
    try:
        # 1. Volume-Price Direction Analysis
        volume_direction = analyze_volume_direction(df)
        
        if volume_direction not in ["BUYING_VOLUME", "NEUTRAL_VOLUME"]:
            return False, f"Volume direction: {volume_direction}"
        
        # 2. OBV Trend Validation
        obv_rising = validate_obv_trend(df)
        
        if not obv_rising:
            return False, "OBV not trending upward"
        
        # 3. Additional Volume Quality Checks
        latest = df.iloc[-1]
        volume = safe_decimal(latest['volume'])
        volume_sma = safe_decimal(latest['Volume_SMA'])
        
        if volume is None or volume_sma is None:
            return False, "Missing volume data"
            
        # Volume should be elevated but not extremely so (avoid panic scenarios)
        volume_ratio = volume / volume_sma
        if volume_ratio > Decimal('5.0'):  # Extremely high volume - could be panic
            return False, f"Extremely high volume ratio: {volume_ratio:.2f}"
        
        logging.debug(f"[{symbol}] {EMOJI_VOLUME} Volume validation PASSED: "
                     f"Direction={volume_direction}, OBV_Rising={obv_rising}, "
                     f"Volume_Ratio={volume_ratio:.2f}")
        
        return True, f"Valid: {volume_direction}, OBV rising, ratio {volume_ratio:.2f}"
        
    except Exception as e:
        logging.error(f"[{symbol}] Error in enhanced_volume_validation: {e}")
        return False, f"Validation error: {e}"

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators required by the strategies."""
    min_len_required = max(EMA_LONG_PERIOD, ADX_PERIOD, RSI_PERIOD, VOLUME_SMA_PERIOD, ATR_PERIOD, BBANDS_PERIOD) + 1
    if df.empty or len(df) < min_len_required:
        logging.debug(f"calculate_indicators: DataFrame empty or too short ({len(df)} < {min_len_required})")
        return pd.DataFrame()

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"calculate_indicators: Missing required columns: need {required_cols}, have {df.columns.tolist()}")
        return pd.DataFrame()

    if df[required_cols].isna().any().any():
        logging.warning("calculate_indicators: Input DataFrame contains NaN values in essential columns.")
        return pd.DataFrame()

    df_copy = df.copy()
    try:
        # Calculate Indicators using pandas_ta
        df_copy.ta.ema(close='close', length=EMA_SHORT_PERIOD, append=True, col_names=('EMA_short',))
        df_copy.ta.ema(close='close', length=EMA_MEDIUM_PERIOD, append=True, col_names=('EMA_medium',))
        df_copy.ta.ema(close='close', length=EMA_LONG_PERIOD, append=True, col_names=('EMA_long',))
        df_copy.ta.rsi(close='close', length=RSI_PERIOD, append=True, col_names=('RSI',))
        df_copy.ta.adx(high='high', low='low', close='close', length=ADX_PERIOD, append=True, col_names=('ADX', 'DMP', 'DMN'))
        df_copy.ta.sma(close='volume', length=VOLUME_SMA_PERIOD, append=True, col_names=('Volume_SMA',))
        df_copy.ta.atr(high='high', low='low', close='close', length=ATR_PERIOD, append=True, col_names=('ATR',))
        
        # Always calculate Bollinger Bands - providing 5 names as pandas_ta.bbands returns 5 series
        df_copy.ta.bbands(close='close', length=BBANDS_PERIOD, std=BBANDS_STD, append=True,
                          col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'))

        # NEW: Calculate On-Balance Volume (OBV)
        df_copy.ta.obv(close='close', volume='volume', append=True, col_names=('OBV',))

        # Calculate 48-period rolling high for breakout detection
        df_copy['recent_high'] = df_copy['close'].rolling(window=48).max().shift(1)

        expected_cols = ['EMA_short', 'EMA_medium', 'EMA_long', 'RSI', 'ADX', 'DMP', 'DMN',
                         'Volume_SMA', 'ATR', 'BBL', 'BBM', 'BBU', 'OBV', 'recent_high']

        # Check if all expected columns were successfully added by pandas_ta
        if not all(col in df_copy.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df_copy.columns]
            logging.warning(f"calculate_indicators: Missing expected columns after computation: {missing_cols}. DataFrame columns: {df_copy.columns.tolist()}")
            return pd.DataFrame()

        # Define the final set of columns to be included in the processed DataFrame.
        final_cols = ['open', 'high', 'low', 'close', 'volume'] + expected_cols
        
        # Create df_processed as an explicit copy to avoid SettingWithCopyWarning
        df_processed = df_copy[final_cols].copy()

        # Drop rows with NaN indicators from df_processed
        # (except recent_high which can be NaN at the start of some calculations)
        df_processed.dropna(subset=[col for col in expected_cols if col != 'recent_high'], inplace=True)

        # Check if enough valid rows remain after dropping NaNs
        min_rows_after_na = 2
        if len(df_processed) < min_rows_after_na:
            logging.warning(f"calculate_indicators: Not enough valid indicator rows after dropna ({len(df_processed)} < {min_rows_after_na})")
            return pd.DataFrame()

        # Final check for NaNs in the core indicator columns of the processed DataFrame
        if df_processed[[col for col in expected_cols if col != 'recent_high']].isna().any().any():
            logging.warning("calculate_indicators: Indicators in df_processed contain NaN values even after dropna.")
            return pd.DataFrame()

        return df_processed

    except Exception as e:
        logging.error(f"Error calculating indicators: {e}", exc_info=True)
        return pd.DataFrame()
    
async def get_latest_price(client: AsyncClient, symbol: str) -> Decimal:
    """Fetches the latest market price for a symbol."""
    await rate_limit_api()
    try:
        ticker = await client.get_symbol_ticker(symbol=symbol)
        price_str = ticker.get('price')
        if price_str:
            if not price_str.replace('.', '', 1).isdigit():
                 logging.warning(f"[{symbol}] Invalid price string received: '{price_str}'")
                 return Decimal('0')
            return Decimal(price_str)
        else:
            logging.warning(f"[{symbol}] Latest price not found in ticker response: {ticker}")
            return Decimal('0')
    except (BinanceAPIException, BinanceRequestException, InvalidOperation, Exception) as e:
         logging.error(f"[{symbol}] Could not get latest price: {e}")
         return Decimal('0')

# --- MULTI-TIMEFRAME DATA FETCHING ---
async def get_multi_timeframe_data(client: AsyncClient, symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetches data for a symbol across multiple timeframes required for Trend strategy."""
    timeframes = {
        '1d': Client.KLINE_INTERVAL_1DAY,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '15m': INTERVAL  # Use the existing 15min interval constant
    }
    
    lookbacks = {
        '1d': 250,  # Enough for 200 EMA
        '4h': 30,   # Enough for RSI 14
        '1h': 30,   # Enough for Bollinger Bands (20,2)
        '15m': LOOKBACK_PERIOD  # Use existing constant
    }
    
    results = {}
    tasks = {}
    
    # Create tasks for each timeframe
    for tf_name, tf_interval in timeframes.items():
        tasks[tf_name] = asyncio.create_task(
            get_latest_data(client, symbol, tf_interval, lookbacks[tf_name])
        )
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # Process results
    for tf_name, task in tasks.items():
        try:
            if task.exception():
                logging.error(f"[{symbol}] Error fetching {tf_name} data: {task.exception()}")
                results[tf_name] = pd.DataFrame()
            else:
                results[tf_name] = task.result()
        except Exception as e:
            logging.error(f"[{symbol}] Error processing {tf_name} result: {e}")
            results[tf_name] = pd.DataFrame()
    
    return results

# --- UPDATED: Enhanced Trend Signal Check with Volume Validation ---
async def check_trend_signal(symbol: str, indicators: pd.DataFrame) -> bool:
    """
    Implements the enhanced TREND strategy conditions on the 15M timeframe:
    1. Price > 200 EMA
    2. RSI(14) ≤ 30
    3. Price closed below Lower Bollinger Band
    4. ENHANCED: Volume validation (buying pressure + OBV confirmation)
    """
    if indicators.empty or len(indicators) < 2:
        logging.warning(f"[{symbol}] Not enough data for trend signal check")
        return False
    
    latest = indicators.iloc[-1]
    
    # Extract all needed indicators using safe_decimal
    price = safe_decimal(latest['close'])
    rsi = safe_decimal(latest['RSI'])
    ema_long = safe_decimal(latest['EMA_long'])  # 200 EMA
    bbl = safe_decimal(latest['BBL'])  # Lower Bollinger Band
    
    # Check all conditions except volume
    if any(v is None for v in [price, rsi, ema_long, bbl]):
        logging.warning(f"[{symbol}] Missing indicators for trend signal check")
        return False
    
    # 1. Price > 200 EMA
    trend_condition_1 = price > ema_long
    
    # 2. RSI(14) ≤ 30
    trend_condition_2 = rsi <= Decimal('30')
    
    # 3. Price closed below Lower Bollinger Band
    trend_condition_3 = price < bbl
    
    # 4. ENHANCED: Volume validation with buying pressure check
    volume_valid, volume_reason = enhanced_volume_validation(indicators, symbol)
    
    # Log individual conditions
    logging.debug(f"[{symbol}] TREND Conditions: "
                 f"Price > 200 EMA: {trend_condition_1}, "
                 f"RSI ≤ 30: {trend_condition_2}, "
                 f"Price < Lower BB: {trend_condition_3}, "
                 f"Volume Valid: {volume_valid} ({volume_reason})")
    
    # All conditions must be met
    trend_signal = trend_condition_1 and trend_condition_2 and trend_condition_3 and volume_valid
    
    if trend_signal:
        logging.info(f"[{symbol}] {EMOJI_VOLUME} TREND SIGNAL: All conditions met! Volume: {volume_reason}")
    elif not volume_valid:
        logging.info(f"[{symbol}] TREND signal blocked by volume validation: {volume_reason}")
    
    return trend_signal

# --- NEW: Calculate ATR-based stop loss ---
def calculate_atr_stop_loss(price: Decimal, atr: Decimal, strategy: str) -> Decimal:
    """
    Calculate a dynamic stop loss based on ATR and strategy type.
    
    Args:
        price: Current price/entry price
        atr: Current ATR value
        strategy: Strategy type ("Trend", "MeanReversion", or "Breakout")
        
    Returns:
        Stop loss price
    """
    # Define ATR multipliers for each strategy
    if strategy == "Trend":
        multiplier = Decimal('2.0')  # More conservative for trend following
    elif strategy == "MeanReversion":
        multiplier = Decimal('1.5')  # Tighter for mean reversion
    else:  # Breakout
        multiplier = Decimal('2.5')  # Wider for breakouts
        
    # Calculate stop distance
    stop_distance = atr * multiplier
    
    # Set a minimum stop percentage based on original SL percentages as floor
    min_stop_percent = Decimal('0')
    if strategy == "Trend":
        min_stop_percent = TREND_SL
    elif strategy == "MeanReversion":
        min_stop_percent = MEAN_REVERT_SL
    else:  # Breakout
        min_stop_percent = BREAKOUT_SL
    
    min_stop_distance = price * min_stop_percent / Decimal('100')
    
    # Use maximum of ATR-based or percentage-based stop distance
    final_stop_distance = max(stop_distance, min_stop_distance)
    
    # Calculate stop loss price
    stop_loss = price - final_stop_distance
    
    logging.info(f"Dynamic ATR stop loss calculated: Entry={price}, ATR={atr}, "
                 f"Stop distance={final_stop_distance} ({final_stop_distance/price*100:.2f}%), "
                 f"Stop price={stop_loss}")
    
    return stop_loss

async def _calculate_current_relevant_value(client: AsyncClient, current_watchlist: List[str]) -> Tuple[Decimal, List[str]]:
    """
    Calculates the current total value of relevant assets (USDC + watchlist base assets)
    and returns the total value and a list of formatted holding strings.
    Requires the current watchlist to be passed in.
    """
    func_name = "_calculate_current_relevant_value"
    logging.info(f"[{func_name}] Starting value calculation (Watchlist: {current_watchlist})...")

    current_total_usdc_value = Decimal('0')
    holdings_list = []
    relevant_assets_set = {QUOTE_ASSET}
    relevant_assets_set.add('USDC')

    try:
        # Use the passed-in watchlist
        for symbol_wl in current_watchlist:
            if symbol_wl.endswith(QUOTE_ASSET):
                base_asset = symbol_wl[:-len(QUOTE_ASSET)]
                relevant_assets_set.add(base_asset)
            else:
                logging.warning(f"[{func_name}] Watchlist symbol {symbol_wl} doesn't end with {QUOTE_ASSET}")
        logging.debug(f"[{func_name}] Relevant assets for value calc: {relevant_assets_set}")

        # Fetch all account balances
        logging.info(f"[{func_name}] Calling rate_limit_api before get_account...")
        await rate_limit_api()
        logging.info(f"[{func_name}] Calling client.get_account()...")
        account = await client.get_account()
        logging.info(f"[{func_name}] client.get_account() returned.")

        balances = account.get('balances', [])
        if not balances:
             logging.warning(f"[{func_name}] Could not retrieve account balances.")
             return Decimal('0'), ["Error: Could not retrieve balances."]

        assets_to_price = {}
        logging.debug(f"[{func_name}] Iterating through {len(balances)} fetched balances...")
        for balance in balances:
            asset = balance['asset']
            # Calculate qty *inside* loop before filtering/use
            qty = safe_decimal(balance['free']) + safe_decimal(balance['locked'])
            if qty is None or qty <= Decimal('0.00000001'):
                continue

            # Filter *after* getting qty
            if asset not in relevant_assets_set:
                continue

            # Process relevant assets
            if asset == QUOTE_ASSET:
                current_total_usdc_value += qty
                holdings_list.append(f"<b>{asset}</b>: {qty:.4f}")
            elif asset == 'USDC' and QUOTE_ASSET != 'USDC':
                price = Decimal('1.0')
                value = qty * price
                if value > Decimal('0.01'):
                    current_total_usdc_value += value
                    holdings_list.append(f"{asset}: {qty:.4f} (Value: ~{value:.2f} {QUOTE_ASSET})")
            else: # Other relevant base assets
                symbol = f"{asset}{QUOTE_ASSET}"
                assets_to_price[symbol] = qty
                logging.debug(f"[{func_name}] Added {symbol} (Qty: {qty}) to assets_to_price.")

        # Fetch prices concurrently
        if assets_to_price:
            logging.info(f"[{func_name}] Preparing to gather prices for: {list(assets_to_price.keys())}")
            price_tasks = {symbol: asyncio.create_task(get_latest_price(client, symbol)) for symbol in assets_to_price}
            logging.info(f"[{func_name}] Calling asyncio.gather() for {len(price_tasks)} price tasks...")
            price_results = await asyncio.gather(*price_tasks.values(), return_exceptions=True)
            logging.info(f"[{func_name}] asyncio.gather() returned.")

            logging.debug(f"[{func_name}] Processing gathered price results...")
            idx = 0
            # Use assets_to_price.keys() to ensure order matches results
            for symbol in assets_to_price.keys():
                task_result = price_results[idx]
                asset = symbol.replace(QUOTE_ASSET, '')
                qty = assets_to_price[symbol] # Get qty again

                try:
                    if isinstance(task_result, Exception):
                         raise task_result # Raise if the task failed
                    price = task_result # Result is the price Decimal
                    if price is None or price <= Decimal('0'):
                        holdings_list.append(f"{asset}: {qty:.8f} (Price Error)")
                    else:
                         value = qty * price
                         if value > Decimal('0.01'):
                             current_total_usdc_value += value
                             holdings_list.append(f"{asset}: {qty:.8f} (Value: {value:.2f} {QUOTE_ASSET} @ {price:.4f})")
                except Exception as price_e:
                    logging.warning(f"[{func_name}] Error processing price for {symbol}: {price_e}")
                    holdings_list.append(f"{asset}: {qty:.8f} (Price N/A)")
                idx += 1  # Increment index AFTER processing result for this symbol

        logging.info(f"[{func_name}] Finished calculation. Total Value: {current_total_usdc_value}")
        return current_total_usdc_value, sorted(holdings_list)

    except Exception as e:
        logging.error(f"Error in {func_name}: {e}", exc_info=True)
        return Decimal('0'), [f"Error calculating value: {e}"]

async def generate_chart(client: AsyncClient, symbol: str, interval: str, lookback: int) -> Optional[str]:
    """Generates a 15-minute timeframe chart with OHLC, EMAs, and volume using mplfinance, returning the filepath."""
    logging.info(f"[{symbol}] Starting chart generation with lookback {lookback}...")
    start_time = time.time()
    chart_filepath = None

    try:
        # Fetch the latest data
        logging.debug(f"[{symbol}] Fetching data for chart...")
        fetch_start = time.time()
        indicator_lookback_needed = max(EMA_LONG_PERIOD, ADX_PERIOD, RSI_PERIOD, VOLUME_SMA_PERIOD, ATR_PERIOD)
        fetch_limit = lookback + indicator_lookback_needed
        df = await get_latest_data(client, symbol, interval, fetch_limit)
        fetch_duration = time.time() - fetch_start
        logging.debug(f"[{symbol}] Data fetching took {fetch_duration:.2f}s. Rows: {len(df) if not df.empty else 0}")

        if df.empty:
            logging.warning(f"[{symbol}] Cannot generate chart: No valid data fetched.")
            return None

        # Calculate indicators
        logging.debug(f"[{symbol}] Calculating indicators for chart...")
        calc_start = time.time()
        indicators = calculate_indicators(df)
        calc_duration = time.time() - calc_start
        logging.debug(f"[{symbol}] Indicator calculation took {calc_duration:.2f}s. Rows: {len(indicators) if not indicators.empty else 0}")

        if indicators.empty:
            logging.warning(f"[{symbol}] Cannot generate chart: No valid indicators calculated.")
            return None

        # Trim to the requested lookback period
        logging.debug(f"[{symbol}] Trimming data to lookback period...")
        trim_start = time.time()
        indicators = indicators.tail(lookback)
        trim_duration = time.time() - trim_start
        logging.debug(f"[{symbol}] Data trimming took {trim_duration:.2f}s. Rows: {len(indicators)}")

        # Prepare DataFrame for mplfinance
        logging.debug(f"[{symbol}] Preparing DataFrame for mplfinance...")
        plot_start = time.time()
        # Ensure the index is timezone-aware (mplfinance requires this)
        if indicators.index.tz is None:
            indicators.index = indicators.index.tz_localize('UTC')
        # Rename columns to match mplfinance expectations
        plot_df = indicators.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Prepare EMAs as additional plots
        ema_plots = [
            mpf.make_addplot(indicators['EMA_short'], color='blue', label=f'EMA {EMA_SHORT_PERIOD}', width=1),
            mpf.make_addplot(indicators['EMA_medium'], color='orange', label=f'EMA {EMA_MEDIUM_PERIOD}', width=1),
            mpf.make_addplot(indicators['EMA_long'], color='purple', label=f'EMA {EMA_LONG_PERIOD}', width=1),
        ]

        # Create a temporary file to save the chart
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            chart_filepath = tmpfile.name
            # Generate the chart using mplfinance
            mpf.plot(
                plot_df,
                type='candle',
                style='charles',
                title=f"{symbol} 15M Chart (24h)",
                ylabel='Price',
                volume=True,
                ylabel_lower='Volume',
                figsize=(12, 7),
                addplot=ema_plots,
                savefig=dict(fname=chart_filepath, dpi=100)
            )
            plot_duration = time.time() - plot_start
            logging.debug(f"[{symbol}] Chart generation took {plot_duration:.2f}s.")

        # Verify the file was created
        if not os.path.exists(chart_filepath):
            logging.error(f"[{symbol}] Chart file was not created at {chart_filepath}.")
            return None

        # Check file size (Telegram photo limit is 10MB)
        file_size_mb = os.path.getsize(chart_filepath) / (1024 * 1024)
        logging.debug(f"[{symbol}] Generated chart file size: {file_size_mb:.2f} MB")
        if file_size_mb > 9:
            logging.warning(f"[{symbol}] Chart file size ({file_size_mb:.2f} MB) may exceed Telegram limit. Attempting compression...")
            from PIL import Image
            img = Image.open(chart_filepath)
            img.save(chart_filepath, format="PNG", optimize=True, quality=85)
            new_size_mb = os.path.getsize(chart_filepath) / (1024 * 1024)
            logging.debug(f"[{symbol}] Compressed chart file size: {new_size_mb:.2f} MB")

        total_duration = time.time() - start_time
        logging.info(f"[{symbol}] Chart generation completed successfully in {total_duration:.2f}s. Saved to {chart_filepath}")
        return chart_filepath

    except Exception as e:
        total_duration = time.time() - start_time
        logging.error(f"[{symbol}] Chart generation failed after {total_duration:.2f}s: {e}", exc_info=True)
        if chart_filepath and os.path.exists(chart_filepath):
            try:
                os.remove(chart_filepath)
                logging.debug(f"[{symbol}] Cleaned up temporary chart file: {chart_filepath}")
            except OSError as rm_err:
                logging.error(f"[{symbol}] Error removing temporary chart file {chart_filepath}: {rm_err}")
        return None

# --- NEW: Validate Symbol Helper ---
async def validate_symbol(client: AsyncClient, symbol: str) -> Tuple[bool, str]:
    """
    Validates if a symbol exists and is tradable on Binance.
    
    Args:
        client: AsyncClient instance
        symbol: Symbol to validate
        
    Returns:
        (is_valid, message) tuple
    """
    try:
        await rate_limit_api()
        info = await client.get_symbol_info(symbol=symbol)
        
        if not info:
            return False, "Symbol not found on exchange"
            
        if info.get('status') != 'TRADING':
            return False, f"Symbol not tradable (Status: {info.get('status', 'N/A')})"
            
        if info.get('quoteAsset') != QUOTE_ASSET:
            return False, f"Symbol does not trade against {QUOTE_ASSET}"
            
        return True, "Symbol is valid"
        
    except BinanceAPIException as e:
        if e.code == -1121:  # Invalid symbol
            return False, "Invalid symbol format"
        return False, f"API error: {e.message}"
        
    except Exception as e:
        logging.error(f"Unexpected error validating symbol {symbol}: {e}")
        return False, f"Validation error: {str(e)}"

# --- MODIFIED: Enhanced Place Market Order with ATR-based SL calculation ---
async def place_market_order(
    client: AsyncClient,
    symbol: str,
    side: str,
    quantity: Decimal,
    strategy: str,
    current_price_for_check: Optional[Decimal] = None,
    max_attempts: int = MAX_ORDER_RETRIES,
    atr_value: Optional[Decimal] = None  # NEW: For dynamic SL calculation
) -> Tuple[Optional[Dict], Optional[Decimal], Optional[Decimal]]:
    """Places a market order with pre-checks, polling, and retries."""
    rules = await get_symbol_rules(client, symbol)
    if not rules:
        logging.error(f"[{symbol}] Cannot place order: Failed to get symbol rules.")
        return None, None, None

    # 1. Adjust Quantity Precision
    adjusted_qty = adjust_quantity_to_precision(quantity, rules['stepSize'])
    if adjusted_qty is None or adjusted_qty <= Decimal('0'):
        logging.error(f"[{symbol}] Cannot place order: Invalid adjusted quantity after precision: {adjusted_qty} from {quantity}")
        return None, None, None

    # 2. Check Min Quantity
    if adjusted_qty < rules['minQty']:
        logging.warning(f"[{symbol}] Adjusted quantity {adjusted_qty} < min quantity {rules['minQty']}. Skipping order.")
        return None, None, None

    # 3. Check Min Notional (More efficient pre-check)
    price_to_use = current_price_for_check
    if price_to_use is None:
         logging.debug(f"[{symbol}] Fetching price for minNotional check as it wasn't provided.")
         price_to_use = await get_latest_price(client, symbol)

    if price_to_use is None or price_to_use <= Decimal('0'):
         logging.error(f"[{symbol}] Cannot perform minNotional check: Invalid price ({price_to_use}). Skipping order.")
         return None, None, None

    notional_value = adjusted_qty * price_to_use
    if notional_value < rules['minNotional']:
        logging.warning(f"[{symbol}] Estimated notional value {notional_value:.4f} < min notional {rules['minNotional']}. Skipping order. (Qty: {adjusted_qty}, Price: {price_to_use})")
        return None, None, None

    # --- Proceed with Order Placement ---
    decimal_places = abs(rules['stepSize'].normalize().as_tuple().exponent)
    quantity_str = "{:0.0{}f}".format(adjusted_qty, decimal_places)

    # NEW: Log ATR value if available (for reference)
    if atr_value is not None and side == Client.SIDE_BUY:
        logging.info(f"[{symbol}] ATR value: {atr_value} for position sizing calculation")

    logging.info(f"[{symbol}] Attempting {side} market order for {quantity_str} (Strategy: {strategy})...")

    attempt = 1
    order = None
    while attempt <= max_attempts:
        try:
            await rate_limit_api()
            order = await client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity_str
            )
            logging.info(f"[{symbol}] Order placed (Attempt {attempt}): ID {order.get('orderId')}, Status {order.get('status')}")

            # --- Order Status Polling ---
            order_id = order.get('orderId')
            if not order_id:
                logging.error(f"[{symbol}] Order response missing orderId: {order}. Retrying if possible.")
                raise ValueError("Order ID missing")

            start_time = time.time()
            final_order_status = None
            while time.time() - start_time < ORDER_STATUS_TIMEOUT_SECONDS:
                await rate_limit_api()
                try:
                    order_status = await client.get_order(symbol=symbol, orderId=order_id)
                    final_order_status = order_status
                    status = order_status.get('status')
                    logging.debug(f"[{symbol}] Polling order {order_id}: Status {status}")

                    if status == 'FILLED':
                        executed_qty_str = order_status.get('executedQty', '0')
                        cummulative_quote_qty_str = order_status.get('cummulativeQuoteQty', '0')

                        executed_qty = safe_decimal(executed_qty_str)
                        cummulative_quote_qty = safe_decimal(cummulative_quote_qty_str)

                        if executed_qty is None or cummulative_quote_qty is None:
                             logging.error(f"[{symbol}] Order {order_id} FILLED but error converting quantities: ExecQty='{executed_qty_str}', CumQuoteQty='{cummulative_quote_qty_str}'")
                             return None, None, None

                        if executed_qty > 0:
                            filled_price = cummulative_quote_qty / executed_qty
                            logging.info(f"[{symbol}] Order {order_id} FILLED: {executed_qty} @ {filled_price:.8f}")
                            if abs(executed_qty - adjusted_qty) > rules['stepSize'] * Decimal('0.9'):
                                logging.warning(f"[{symbol}] Partial fill detected: Executed {executed_qty} / Requested {adjusted_qty}")
                            return order_status, filled_price, executed_qty
                        else:
                            logging.warning(f"[{symbol}] Order {order_id} status FILLED but executedQty is {executed_qty}. Treating as 0 fill.")
                            return order_status, Decimal('0'), Decimal('0')

                    if status in ['REJECTED', 'EXPIRED', 'CANCELED', 'PENDING_CANCEL']:
                        logging.warning(f"[{symbol}] Order {order_id} ended with status {status}.")
                        return None, None, None

                    await asyncio.sleep(ORDER_STATUS_POLL_INTERVAL)

                except (BinanceAPIException, BinanceRequestException) as poll_e:
                     logging.warning(f"[{symbol}] API error polling order {order_id} (will retry polling if timeout not reached): {poll_e}")
                     await asyncio.sleep(ORDER_STATUS_POLL_INTERVAL * 2)

            # --- Polling Timeout ---
            logging.warning(f"[{symbol}] Order {order_id} status polling timed out after {ORDER_STATUS_TIMEOUT_SECONDS}s. Last status: {final_order_status.get('status', 'Unknown') if final_order_status else 'Unknown'}")
            # Attempt to cancel the order if it's possibly stuck
            if final_order_status and final_order_status.get('status') in ['NEW', 'PARTIALLY_FILLED']:
                try:
                    await rate_limit_api()
                    cancel_res = await client.cancel_order(symbol=symbol, orderId=order_id)
                    logging.info(f"[{symbol}] Attempted cancel for timed-out order {order_id}. Response Status: {cancel_res.get('status')}")
                except (BinanceAPIException, BinanceRequestException) as cancel_e:
                     logging.error(f"[{symbol}] Failed to cancel timed-out order {order_id}: {cancel_e}")
            return None, None, None

        except (BinanceAPIException, BinanceRequestException) as e:
            logging.warning(f"[{symbol}] API Error placing order (Attempt {attempt}/{max_attempts}): {e.code} - {e.message}")
            if e.code == -2010 and "insufficient balance" in e.message.lower():
                logging.error(f"[{symbol}] Order failed due to insufficient balance. Check available funds.")
                return None, None, None
            if e.code == -1013:
                 logging.error(f"[{symbol}] Order failed due to exchange filters (Price/Qty/Notional): {e.message}")
                 return None, None, None
            # Other potentially retryable API errors
            if attempt < max_attempts:
                delay = ORDER_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logging.info(f"Retrying order placement in {delay:.1f}s...")
                await asyncio.sleep(delay)
                attempt += 1
                continue
            else:
                 logging.error(f"[{symbol}] Order failed after {max_attempts} attempts due to API error: {e}")
                 return None, None, None

        except Exception as e:
            logging.error(f"[{symbol}] Unexpected error placing order (Attempt {attempt}/{max_attempts}): {e}", exc_info=True)
            if attempt < max_attempts:
                delay = ORDER_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logging.info(f"Retrying order placement in {delay:.1f}s...")
                await asyncio.sleep(delay)
                attempt += 1
                continue
            else:
                logging.error(f"[{symbol}] Order placement failed permanently after {max_attempts} attempts due to unexpected error.")
                return None, None, None

    logging.error(f"[{symbol}] Order placement failed after all attempts and loops.")
    return None, None, None

# --- Calculate current drawdown ---
def calculate_current_drawdown(current_equity: Decimal, max_equity: Decimal) -> Decimal:
    """
    Calculate current drawdown as a percentage.
    
    Args:
        current_equity: Current portfolio value
        max_equity: Maximum portfolio value observed
        
    Returns:
        Drawdown as a percentage (negative number)
    """
    if max_equity <= 0:
        return Decimal('0')
    
    drawdown = ((current_equity / max_equity) - Decimal('1')) * Decimal('100')
    return drawdown  # Will be negative for a drawdown

# --- Telegram Functions ---
async def send_telegram_message(message: str) -> None:
    """Sends a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN_2 or not TELEGRAM_CHAT_ID:
        logging.debug("Telegram token or chat ID not set, skipping message.")
        return
    try:
        async with Bot(token=TELEGRAM_BOT_TOKEN_2) as bot:
            await bot.send_message(chat_id=int(TELEGRAM_CHAT_ID), text=message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        logging.debug(f"Sent TG message: {message[:50]}...")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    
    help_text = (
        f"{EMOJI_START} Trading Bot Initialized.\n\n"
        f"<b>Available Commands:</b>\n"
        f"/status - Show current status, mode, and positions.\n"
        f"/balance - Show portfolio balance (relevant assets).\n"
        f"/criteria - Check current criteria for monitored symbols.\n"
        f"/chart SYMBOL - Generate 15M chart for a symbol.\n\n"
        f"<b>Manual Trading:</b>\n"
        f"/buy SYMBOL AMOUNT_USDC [STRATEGY] - Place manual market buy order.\n"
        f"/forcesell SYMBOL - Immediately sell specified position.\n\n"
        f"<b>Watchlist:</b>\n"
        f"/include SYMBOL - Add symbol to watchlist.\n"
        f"/exclude SYMBOL - Remove symbol from monitoring.\n\n"
        f"<b>Operation:</b>\n"
        f"/pause - Pause automatic strategy entries.\n"
        f"/resume - Resume automatic strategy entries.\n"
        f"/stop - Stop bot and liquidate all positions."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /status command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    async with state_lock:
        s_is_running = state['is_running']
        s_is_paused = state['is_paused']
        s_open_count = state['open_position_count']
        s_watchlist = state['watchlist_symbols'][:]
        s_excluded = state['excluded_symbols'][:]
        s_last_error = state['last_error']
        s_positions = state['positions'].copy()
        
        # Calculate current drawdown
        current_equity = state.get('equity_history', [])
        max_equity = state.get('max_equity_seen', Decimal('0'))
        if current_equity and max_equity > 0:
            current_equity_value = current_equity[-1] if isinstance(current_equity[-1], Decimal) else Decimal('0')
            current_drawdown = calculate_current_drawdown(current_equity_value, max_equity)
        else:
            current_drawdown = None

    status_msg = (
        f"{EMOJI_STATUS} <b>Bot Status ({datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')})</b>\n"
        f"Mode: {BINANCE_ENV}\n"
        f"Running: {s_is_running}\nPaused: {s_is_paused}\n"
        f"Open Positions: {s_open_count}/{MAX_CONCURRENT_POSITIONS}\n"
    )
    # Add drawdown information if available
    if current_drawdown is not None:
        drawdown_emoji = EMOJI_PROFIT if current_drawdown >= 0 else EMOJI_LOSS
        status_msg += f"Current Drawdown: {drawdown_emoji} {current_drawdown:.2f}%\n"
    
    status_msg += (
        f"Watchlist ({len(s_watchlist)}): {', '.join(s_watchlist)}\n"
        f"Excluded ({len(s_excluded)}): {', '.join(s_excluded) if s_excluded else 'None'}\n"
        f"Last Error: {s_last_error if s_last_error else 'None'}\n"
        "--------------------\n"
    )

    if s_positions:
        status_msg += "<b>Open Positions:</b>\n"
        for symbol in sorted(s_positions.keys()):
            pos = s_positions[symbol]
            pnl = pos.get('current_pnl_percent', Decimal('0'))
            strategy = pos.get('strategy', 'N/A')
            entry_p = pos.get('entry_price', Decimal('0'))
            held_q = pos.get('held_qty', Decimal('0'))
            tp = pos.get('tp_price', Decimal('0'))
            sl = pos.get('sl_price', Decimal('0'))
            trailing_active = pos.get('trailing_stop_active', False)
            trailing_stop = pos.get('trailing_stop_price', sl)
            pnl_emoji = EMOJI_PROFIT if pnl >= 0 else EMOJI_LOSS

            # Display trailing stop if active
            if trailing_active:
                stop_display = f"TSL: {trailing_stop:.4f} ({((entry_p-trailing_stop)/entry_p)*100:.2f}%)"
            else:
                stop_display = f"SL: {sl:.4f} ({((entry_p-sl)/entry_p)*100:.2f}%)"

            status_msg += (
                f"<b>{symbol}</b> ({strategy}) {pnl_emoji}{pnl:.2f}%\n"
                f"  Qty: {held_q:.8f} | Entry: {entry_p:.4f}\n"
                f"  TP: {tp:.4f} | {stop_display}\n"
                "-\n"
                )
    else:
        status_msg += "No open positions managed by the bot.\n"

    MAX_MSG_LEN = 4096
    if len(status_msg) > MAX_MSG_LEN:
         status_msg = status_msg[:4050] + "\n... (message truncated)"

    await update.message.reply_text(status_msg, parse_mode=ParseMode.HTML)

# --- UPDATED: Enhanced Criteria Command with Volume Validation Details ---
async def criteria_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /criteria command with HTML-safe formatting and enhanced volume validation details."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    await update.message.reply_text(f"{EMOJI_WAIT} Fetching latest criteria data...")
    async with state_lock:
        watchlist = [s for s in state['watchlist_symbols'] if s not in state['excluded_symbols']]
        if not watchlist:
            await update.message.reply_text(f"{EMOJI_INFO} No symbols currently being monitored.")
            return

    results_text = f"{EMOJI_CRITERIA} <b>Criteria Check ({datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')})</b>\n"
    results_text += "--------------------\n"

    for symbol in watchlist:
        results_text_symbol = f"<b>{symbol}</b>:\n"
        try:
            df = await get_latest_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
            
            if df.empty:
                results_text_symbol += "  No data available for this symbol.\n"
                results_text_symbol += "--------------------\n"
                results_text += results_text_symbol
                continue
            
            indicators = calculate_indicators(df)
            if indicators.empty:
                results_text_symbol += "  Could not calculate indicators for this symbol.\n"
                results_text_symbol += "--------------------\n"
                results_text += results_text_symbol
                continue
            
            latest = indicators.iloc[-1]

            price = safe_decimal(latest['close'])
            rsi = safe_decimal(latest['RSI'])
            adx = safe_decimal(latest['ADX'])
            volume = safe_decimal(latest['volume'])
            volume_sma = safe_decimal(latest['Volume_SMA'])
            ema_medium = safe_decimal(latest['EMA_medium'])
            ema_long = safe_decimal(latest['EMA_long'])
            bbl = safe_decimal(latest['BBL'])
            recent_high = safe_decimal(latest['recent_high'])

            def fmt(value, precision_format_str):
                if value is None:
                    return "N/A"
                try:
                    if not isinstance(value, (Decimal, float, int)):
                        return "NotNum"
                    return f"{value:{precision_format_str}}"
                except (TypeError, ValueError):
                    logging.warning(f"Formatting error for value '{value}' type {type(value)} with format '{precision_format_str}'")
                    return "ErrFmt"

            if price is None:
                logging.warning(f"[{symbol}] Price is None in criteria_command, cannot display full details.")
                results_text_symbol += "  Core price data unavailable for detailed criteria display.\n"

            # --- Enhanced Volume Analysis ---
            volume_valid, volume_reason = enhanced_volume_validation(indicators, symbol)
            volume_direction = analyze_volume_direction(indicators)
            obv_rising = validate_obv_trend(indicators)

            # --- TREND Strategy Conditions (with enhanced volume validation) ---
            trend_condition_1 = price is not None and ema_long is not None and price > ema_long
            trend_condition_2 = rsi is not None and rsi <= Decimal('30')
            trend_condition_3 = price is not None and bbl is not None and price < bbl
            trend_condition_4 = volume_valid  # Enhanced volume validation
            trend_signal_active = trend_condition_1 and trend_condition_2 and trend_condition_3 and trend_condition_4

            # --- Mean Reversion Strategy Conditions ---
            mr_rsi_ok = rsi is not None and RSI_OVERSOLD is not None and rsi < RSI_OVERSOLD
            mr_price_ema_ok = price is not None and ema_medium is not None and price < ema_medium
            mr_vol_ok = volume_valid  # Enhanced volume validation
            mean_revert_signal_active = mr_rsi_ok and mr_price_ema_ok and mr_vol_ok
            
            # --- Breakout Strategy Conditions ---
            bo_price_high_ok = price is not None and recent_high is not None and price > recent_high
            bo_vol_ok = volume_valid  # Enhanced volume validation
            bo_adx_ok = adx is not None and ADX_BREAKOUT_THRESHOLD is not None and adx > ADX_BREAKOUT_THRESHOLD
            bo_rsi_ok = rsi is not None and RSI_EXIT is not None and rsi > RSI_EXIT
            breakout_signal_active = bo_price_high_ok and bo_vol_ok and bo_adx_ok and bo_rsi_ok
            
            # --- Format output for this symbol using HTML entities ---
            results_text_symbol += f"<b>Price:</b> {fmt(price, '.4f')}\n"
            results_text_symbol += f"<b>RSI({RSI_PERIOD}):</b> {fmt(rsi, '.2f')}\n"
            results_text_symbol += f"<b>ADX({ADX_PERIOD}):</b> {fmt(adx, '.2f')}\n"
            results_text_symbol += f"<b>Volume:</b> {fmt(volume, '.2f')} (SMA({VOLUME_SMA_PERIOD}): {fmt(volume_sma, '.2f')})\n"
            results_text_symbol += f"<b>EMAs M({EMA_MEDIUM_PERIOD}):</b> {fmt(ema_medium, '.4f')} <b>L({EMA_LONG_PERIOD}):</b> {fmt(ema_long, '.4f')}\n"
            results_text_symbol += f"<b>Recent High(12H):</b> {fmt(recent_high, '.4f')}\n"
            
            # --- Enhanced Volume Analysis Display ---
            volume_emoji = EMOJI_CHECK if volume_valid else EMOJI_CROSS
            results_text_symbol += f"\n<b>{EMOJI_VOLUME} Volume Analysis:</b> {volume_emoji}\n"
            results_text_symbol += f"  Direction: {volume_direction}\n"
            results_text_symbol += f"  OBV Rising: {EMOJI_CHECK if obv_rising else EMOJI_CROSS}\n"
            results_text_symbol += f"  Reason: {volume_reason}\n"

            # --- Strategy Results ---
            results_text_symbol += f"\n<b>TREND Strategy:</b> {EMOJI_CHECK if trend_signal_active else EMOJI_CROSS}\n"
            results_text_symbol += f"  1. P &gt; EMA({EMA_LONG_PERIOD}): [{EMOJI_CHECK if trend_condition_1 else EMOJI_CROSS}]\n"
            results_text_symbol += f"  2. RSI &#8804; 30: [{EMOJI_CHECK if trend_condition_2 else EMOJI_CROSS}]\n"
            results_text_symbol += f"  3. P &lt; LowerBB: [{EMOJI_CHECK if trend_condition_3 else EMOJI_CROSS}]\n"
            results_text_symbol += f"  4. Volume Valid: [{EMOJI_CHECK if trend_condition_4 else EMOJI_CROSS}]\n"

            results_text_symbol += f"\n<b>MEAN REVERSION Strategy:</b> {EMOJI_CHECK if mean_revert_signal_active else EMOJI_CROSS}\n"
            results_text_symbol += f"  1. RSI &lt; {RSI_OVERSOLD}: [{EMOJI_CHECK if mr_rsi_ok else EMOJI_CROSS}]\n"
            results_text_symbol += f"  2. P &lt; EMA({EMA_MEDIUM_PERIOD}): [{EMOJI_CHECK if mr_price_ema_ok else EMOJI_CROSS}]\n"
            results_text_symbol += f"  3. Volume Valid: [{EMOJI_CHECK if mr_vol_ok else EMOJI_CROSS}]\n"

            results_text_symbol += f"\n<b>BREAKOUT Strategy:</b> {EMOJI_CHECK if breakout_signal_active else EMOJI_CROSS}\n"
            results_text_symbol += f"  1. P &gt; Recent High: [{EMOJI_CHECK if bo_price_high_ok else EMOJI_CROSS}]\n"
            results_text_symbol += f"  2. Volume Valid: [{EMOJI_CHECK if bo_vol_ok else EMOJI_CROSS}]\n"
            results_text_symbol += f"  3. ADX &gt; {ADX_BREAKOUT_THRESHOLD}: [{EMOJI_CHECK if bo_adx_ok else EMOJI_CROSS}]\n"
            results_text_symbol += f"  4. RSI &gt; {RSI_EXIT}: [{EMOJI_CHECK if bo_rsi_ok else EMOJI_CROSS}]\n"
            
            # Breakout Cooldown Status
            last_bo_time = state.get('last_breakout_times_by_symbol', {}).get(symbol, 0.0)
            cooldown_seconds = BREAKOUT_COOLDOWN_HOURS * 60 * 60
            if last_bo_time > 0:
                time_since_last_bo = time.time() - last_bo_time
                if time_since_last_bo <= cooldown_seconds:
                    remaining_hours = (cooldown_seconds - time_since_last_bo) / 3600
                    results_text_symbol += f"  Cooldown: {EMOJI_WAIT} Active ({fmt(remaining_hours, '.1f')}h left)\n"
                else:
                    results_text_symbol += f"  Cooldown: {EMOJI_CHECK} Inactive (Expired)\n"
            else:
                results_text_symbol += f"  Cooldown: {EMOJI_CHECK} Inactive (No prior BO)\n"
            
        except Exception as symbol_e:
            logging.error(f"Error processing criteria for {symbol}: {symbol_e}", exc_info=True)
            results_text_symbol += f"  Error processing data: {str(symbol_e)}\n"

        results_text_symbol += "--------------------\n"
        results_text += results_text_symbol

    # Handle potential message length issues (send in chunks if needed)
    MAX_MSG_LEN = 4096
    if len(results_text) > MAX_MSG_LEN:
        logging.info(f"Criteria message length ({len(results_text)}) exceeds limit. Sending in chunks.")
        start_index = 0
        while start_index < len(results_text):
            separator = '\n--------------------\n'
            chunk_limit = start_index + MAX_MSG_LEN
            
            split_pos = results_text.rfind(separator, start_index, chunk_limit)

            if split_pos != -1 and split_pos > start_index:
                chunk_end = split_pos + len(separator)
            elif chunk_limit < len(results_text):
                split_pos = results_text.rfind('\n', start_index, chunk_limit)
                if split_pos != -1 and split_pos > start_index:
                    chunk_end = split_pos + 1
                else:
                    chunk_end = chunk_limit - 50
            else:
                chunk_end = len(results_text)

            chunk = results_text[start_index:chunk_end]
            
            if not chunk.strip(): break

            if chunk_end < len(results_text) and (split_pos == -1 or split_pos < start_index):
                 if len(chunk) > MAX_MSG_LEN - 20:
                     chunk = chunk[:MAX_MSG_LEN - 20]
                 chunk += "\n...(truncated chunk)..."

            await send_telegram_message(chunk)
            start_index = chunk_end
            if start_index < len(results_text): await asyncio.sleep(0.5)
    else:
        await send_telegram_message(results_text)

async def exclude_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /exclude command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(f"{EMOJI_WARNING} Usage: /exclude SYMBOL")
        return
    symbol = context.args[0].upper()

    async with state_lock:
        if symbol not in state['watchlist_symbols']:
            await update.message.reply_text(f"{EMOJI_WARNING} {symbol} is not currently in the watchlist.")
            return
        if symbol not in state['excluded_symbols']:
            state['excluded_symbols'].append(symbol)
            state['excluded_symbols'].sort()
            await update.message.reply_text(f"{EMOJI_EXCLUDE} Excluded {symbol} from monitoring.")
            logging.info(f"Excluded {symbol} via Telegram command.")
        else:
             await update.message.reply_text(f"{EMOJI_INFO} {symbol} is already excluded.")

async def include_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /include command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(f"{EMOJI_WARNING} Usage: /include SYMBOL")
        return
    symbol = context.args[0].upper()

    # Validate symbol before adding
    await update.message.reply_text(f"{EMOJI_WAIT} Validating {symbol}...")
    is_valid, msg = await validate_symbol(client, symbol)
    if not is_valid:
        await update.message.reply_text(f"{EMOJI_CROSS} {msg}")
        return

    async with state_lock:
        # Remove from excluded if it's there
        if symbol in state['excluded_symbols']:
            state['excluded_symbols'].remove(symbol)
            logging.info(f"Removed {symbol} from exclusion list via include command.")
        # Add to watchlist if not already there
        if symbol not in state['watchlist_symbols']:
            state['watchlist_symbols'].append(symbol)
            state['watchlist_symbols'].sort()
            await update.message.reply_text(f"{EMOJI_INCLUDE} Added {symbol} to watchlist.")
            logging.info(f"Included {symbol} via Telegram command.")
        else:
            await update.message.reply_text(f"{EMOJI_INFO} {symbol} was already in the watchlist (and removed from excluded if present).")

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /balance command, showing relevant assets and PnL vs baseline."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    await update.message.reply_text(f"{EMOJI_WAIT} Calculating relevant portfolio balance...")

    try:
        # Get current watchlist first (needs lock)
        async with state_lock:
             current_watchlist_for_cmd = state['watchlist_symbols'][:]
             current_equity = state.get('equity_history', [])
             max_equity = state.get('max_equity_seen', Decimal('0'))
        
        logging.debug(f"[/balance] Watchlist for command: {current_watchlist_for_cmd}")

        # Calculate current value and get holdings
        current_total_usdc_value, holdings = await _calculate_current_relevant_value(client, current_watchlist_for_cmd)
        logging.debug(f"[/balance] Calculated value: {current_total_usdc_value}, Holdings: {holdings}")

        # Retrieve initial baseline value from state
        async with state_lock:
            baseline = state.get('initial_portfolio_value_baseline', None)
        logging.debug(f"[/balance] Retrieved baseline: {baseline}")

        # Calculate current drawdown
        current_drawdown = None
        if current_equity and max_equity > 0 and current_equity[-1] is not None:
            current_equity_value = current_equity[-1] if isinstance(current_equity[-1], Decimal) else Decimal('0')
            current_drawdown = calculate_current_drawdown(current_equity_value, max_equity)

        # Construct message
        msg_header = f"{EMOJI_BALANCE} <b>Portfolio Balance (Relevant Assets)</b>\n--------------------\n"
        msg_body = "\n".join(holdings) if holdings else "No relevant holdings found or error during calculation."

        # Footer section with Total Value
        msg_footer = f"--------------------\n<b>Current Value (Relevant): {current_total_usdc_value:.2f} {QUOTE_ASSET}</b>"

        # Add drawdown information if available
        if current_drawdown is not None:
            drawdown_emoji = EMOJI_PROFIT if current_drawdown >= 0 else EMOJI_LOSS
            msg_footer += f"\nCurrent Drawdown: {drawdown_emoji} {current_drawdown:.2f}%"

        # PnL Calculation vs Baseline
        if baseline is not None and baseline > Decimal('0'):
            pnl = current_total_usdc_value - baseline
            pnl_percent = (pnl / baseline * 100) if baseline != 0 else Decimal('0')
            pnl_emoji = EMOJI_PROFIT if pnl >= 0 else EMOJI_LOSS
            msg_footer += f"\nInitial Baseline Value: {baseline:.2f}"
            msg_footer += f"\nPnL (vs Baseline): {pnl_emoji} {pnl:.2f} ({pnl_percent:.2f}%)"
        elif baseline is None:
             msg_footer += "\nInitial Baseline: Not set yet (try again shortly)"
        else:
             msg_footer += f"\nInitial Baseline: {baseline:.2f} (Cannot calculate PnL %)"

        # Handle message length for Telegram
        full_msg = msg_header + msg_body + msg_footer
        MAX_MSG_LEN = 4096
        if len(full_msg) > MAX_MSG_LEN:
            cutoff = len(msg_header) + len(msg_body) - (len(full_msg) - MAX_MSG_LEN + 50)
            msg_body_truncated = msg_body[:max(0, cutoff)] + "\n... (holdings truncated)"
            full_msg = msg_header + msg_body_truncated + msg_footer
            if len(full_msg) > MAX_MSG_LEN:
                full_msg = full_msg[:4050] + "\n...(message truncated)"

        await update.message.reply_text(full_msg, parse_mode=ParseMode.HTML)

    except Exception as e:
        logging.error(f"Error handling /balance: {e}", exc_info=True)
        await update.message.reply_text(f"{EMOJI_ERROR} An error occurred during /balance.")

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /pause command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    async with state_lock:
        if state['is_paused']:
            await update.message.reply_text(f"{EMOJI_INFO} Bot is already paused.")
            return
        state['is_paused'] = True
        state['last_error'] = "Paused by user command."
    logging.info("Bot paused via Telegram command.")
    await update.message.reply_text(f"{EMOJI_PAUSE} Bot paused. No new entries will be made.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /resume command."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    async with state_lock:
        if not state['is_paused']:
            await update.message.reply_text(f"{EMOJI_INFO} Bot is already running.")
            return
        state['is_paused'] = False
        if state['last_error'] == "Paused by user command.":
             state['last_error'] = None
    logging.info("Bot resumed via Telegram command.")
    await update.message.reply_text(f"{EMOJI_RESUME} Bot resumed.")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /stop command - signals the bot to stop gracefully."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    logging.info("Received stop command via Telegram. Signaling shutdown...")
    async with state_lock:
        state['is_running'] = False
    await update.message.reply_text(f"{EMOJI_STOP} Stop signal received. Bot will shut down and liquidate after current cycle finishes...")

async def forcesell_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /forcesell command to immediately sell a specific position."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(f"{EMOJI_WARNING} Usage: /forcesell SYMBOL")
        return
    symbol = context.args[0].upper()

    if not symbol.endswith(QUOTE_ASSET) or len(symbol) <= len(QUOTE_ASSET):
         await update.message.reply_text(f"{EMOJI_CROSS} Invalid symbol format: {symbol}")
         return

    pos_to_sell = None
    async with state_lock:
        pos = state['positions'].get(symbol)
        if not pos:
            await update.message.reply_text(f"{EMOJI_WARNING} No active position found for {symbol} in bot state.")
            return
        pos_to_sell = pos.copy()

    qty = pos_to_sell.get('held_qty', Decimal('0'))
    entry_price = pos_to_sell.get('entry_price', Decimal('0'))
    strategy = pos_to_sell.get('strategy', 'ForceSell')

    if qty <= 0:
        await update.message.reply_text(f"{EMOJI_INFO} Position {symbol} has zero or negative quantity in state.")
        async with state_lock:
            if symbol in state['positions']:
                 del state['positions'][symbol]
                 state['open_position_count'] = max(0, state['open_position_count'] - 1)
        return

    await update.message.reply_text(f"{EMOJI_WAIT} Attempting to force sell {qty:.8f} {symbol}...")

    order, filled_price, executed_qty = await place_market_order(client, symbol, Client.SIDE_SELL, qty, strategy, current_price_for_check=None)

    if executed_qty is not None and executed_qty > 0:
        profit = (filled_price - entry_price) * executed_qty
        pnl_emoji = EMOJI_PROFIT if profit >= 0 else EMOJI_LOSS
        await update.message.reply_text(
            f"{EMOJI_SELL_SL} [{symbol}] Force-Sold: {executed_qty:.8f} @ {filled_price:.4f}\n"
            f"PnL: {pnl_emoji} {profit:.2f} USDC"
            )
        logging.info(f"Force sold {symbol}: Qty={executed_qty}, Price={filled_price}, Entry={entry_price}, PnL={profit}")
        async with state_lock:
            if symbol in state['positions']:
                 del state['positions'][symbol]
                 state['open_position_count'] = max(0, state['open_position_count'] - 1)
            else:
                 logging.warning(f"[{symbol}] Position not found in state after successful force sell?")

    elif executed_qty == Decimal('0'):
         await update.message.reply_text(f"{EMOJI_WARNING} [{symbol}] Force-sell order filled with 0 quantity. Check exchange.")
         logging.warning(f"Force sell {symbol} filled with 0 qty.")
         async with state_lock:
            if symbol in state['positions']:
                 del state['positions'][symbol]
                 state['open_position_count'] = max(0, state['open_position_count'] - 1)
    else:
        await update.message.reply_text(f"{EMOJI_ERROR} [{symbol}] Force-sell order failed. Check logs and exchange.")
        logging.error(f"Force sell order failed for {symbol}.")

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /chart SYMBOL command to send a 15M chart snapshot."""
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(f"{EMOJI_WARNING} Usage: /chart SYMBOL")
        return
    symbol = context.args[0].upper()

    # Validate symbol
    if not symbol.endswith(QUOTE_ASSET) or len(symbol) <= len(QUOTE_ASSET):
        await update.message.reply_text(f"{EMOJI_CROSS} Invalid symbol format: {symbol}")
        return

    # Check if symbol is in watchlist or positions
    async with state_lock:
        valid_symbols = state['watchlist_symbols'] + list(state['positions'].keys())
    if symbol not in valid_symbols:
        await update.message.reply_text(f"{EMOJI_WARNING} {symbol} is not in watchlist or positions.")
        return

    # Validate symbol exists on Binance
    await update.message.reply_text(f"{EMOJI_WAIT} Validating {symbol} for chart...")
    is_valid, msg = await validate_symbol(client, symbol)
    if not is_valid:
        await update.message.reply_text(f"{EMOJI_CROSS} {msg}")
        return

    # Generate the chart
    await update.message.reply_text(f"{EMOJI_WAIT} Generating chart for {symbol} (15M, 24h lookback)...")
    chart_timeout = 25.0
    chart_filepath = None
    try:
        chart_filepath = await asyncio.wait_for(
            generate_chart(client, symbol, INTERVAL, CHART_LOOKBACK_PERIOD),
            timeout=chart_timeout
        )
        if chart_filepath is None:
            await update.message.reply_text(f"{EMOJI_ERROR} Failed to generate chart for {symbol}. Check logs for details.")
            return
        # Send the chart as a photo via Telegram
        with open(chart_filepath, 'rb') as chart_file:
            await update.message.reply_photo(
                photo=chart_file,
                caption=f"{EMOJI_CHART} {symbol} 15M Chart (24h)\nGenerated at {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
        logging.info(f"Sent chart for {symbol} via Telegram.")

    except asyncio.TimeoutError:
        logging.error(f"[{symbol}] Chart generation timed out after {chart_timeout}s.")
        await update.message.reply_text(f"{EMOJI_ERROR} Chart generation for {symbol} timed out after {chart_timeout}s.")
    except Exception as e:
        logging.error(f"[{symbol}] Error generating/sending chart: {e}", exc_info=True)
        await update.message.reply_text(f"{EMOJI_ERROR} Error generating/sending chart for {symbol}: {e}")
    finally:
        if chart_filepath and os.path.exists(chart_filepath):
            try:
                os.remove(chart_filepath)
                logging.debug(f"[{symbol}] Cleaned up temporary chart file: {chart_filepath}")
            except OSError as rm_err:
                logging.error(f"[{symbol}] Error removing temporary chart file {chart_filepath}: {rm_err}")

# --- ENHANCED buy_market_usdc_command with Strategy Selection ---
async def buy_market_usdc_command(update: Update, context: ContextTypes.DEFAULT_TYPE, client: AsyncClient) -> None:
    """Handles the /buy SYMBOL AMOUNT_USDC [STRATEGY] command for manual market orders."""
    func_name = "buy_market_usdc_command"
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        logging.warning(f"[{func_name}] Received command from unauthorized chat ID: {update.effective_chat.id}")
        return

    # --- 1. Parse Arguments ---
    if not context.args or len(context.args) < 2 or len(context.args) > 3:
        await update.message.reply_text(
            f"{EMOJI_WARNING} Usage: /buy SYMBOL AMOUNT_USDC [STRATEGY]\n"
            f"Example: /buy HBARUSDC 150\n"
            f"Example with strategy: /buy HBARUSDC 150 breakout\n\n"
            f"Valid strategies: trend, meanrevert, breakout (default: manual)"
        )
        return

    symbol = context.args[0].upper()
    amount_usdc_str = context.args[1]
    
    # Optional strategy parameter
    strategy_type = "Manual"
    if len(context.args) == 3:
        input_strategy = context.args[2].lower()
        if input_strategy in ["trend", "meanrevert", "breakout"]:
            strategy_type = input_strategy.capitalize()
            if strategy_type == "Meanrevert":
                strategy_type = "MeanReversion"  # Normalize name
    
    amount_usdc = None
    try:
        amount_usdc = Decimal(amount_usdc_str)
        if amount_usdc <= 0:
            raise ValueError("Amount must be positive")
    except (InvalidOperation, ValueError) as e:
        logging.warning(f"[{func_name}:{symbol}] Invalid amount provided: {amount_usdc_str}. Error: {e}")
        await update.message.reply_text(f"{EMOJI_ERROR} Invalid AMOUNT_USDC: '{amount_usdc_str}'. Must be a positive number.")
        return

    # --- 2. Validate Symbol Format & Quote Asset ---
    if not symbol.endswith(QUOTE_ASSET):
        await update.message.reply_text(f"{EMOJI_ERROR} Invalid symbol: {symbol}. Must end with {QUOTE_ASSET}.")
        return

    await update.message.reply_text(f"{EMOJI_WAIT} Processing manual MARKET buy for {amount_usdc} {QUOTE_ASSET} of {symbol} ({strategy_type} strategy)...")
    logging.info(f"[{func_name}:{symbol}] Received manual buy request for {amount_usdc} {QUOTE_ASSET} with strategy {strategy_type}.")

    # --- 3. Check State & Validate Symbol ---
    try:
        is_valid, msg = await validate_symbol(client, symbol)
        if not is_valid:
            await update.message.reply_text(f"{EMOJI_CROSS} Cannot buy {symbol}: {msg}")
            return
        logging.debug(f"[{func_name}:{symbol}] Symbol validation successful.")

        # Check state (max positions, existing position) under lock
        async with state_lock:
            position_exists = symbol in state['positions']
            current_pos_count = state.get('open_position_count', 0)
            max_pos_reached = current_pos_count >= MAX_CONCURRENT_POSITIONS

        if position_exists:
            await update.message.reply_text(f"{EMOJI_WARNING} Position already open for {symbol} in bot state. Cannot place duplicate manual buy.")
            return
        if max_pos_reached:
            await update.message.reply_text(f"{EMOJI_WARNING} Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached. Cannot open new position.")
            return

        # --- 4. Check USDC Balance ---
        logging.info(f"[{func_name}:{symbol}] Checking {QUOTE_ASSET} balance...")
        usdc_balance = await get_balance(client, QUOTE_ASSET)
        if usdc_balance < amount_usdc:
            await update.message.reply_text(
                f"{EMOJI_ERROR} Insufficient balance. Have {usdc_balance:.2f} {QUOTE_ASSET}, need {amount_usdc:.2f} {QUOTE_ASSET}."
            )
            return
        logging.info(f"[{func_name}:{symbol}] Balance sufficient ({usdc_balance:.2f} {QUOTE_ASSET}).")

        # --- 5. Get Price & Technical Indicators for Advanced Position Sizing ---
        logging.info(f"[{func_name}:{symbol}] Fetching latest data for trade setup...")
        df = await get_latest_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
        if df.empty:
            logging.warning(f"[{func_name}:{symbol}] Failed to get latest data for ATR calculation.")
            await update.message.reply_text(f"{EMOJI_ERROR} Failed to fetch market data for {symbol}.")
            return
            
        indicators = calculate_indicators(df)
        if indicators.empty:
            logging.warning(f"[{func_name}:{symbol}] Failed to calculate indicators.")
            await update.message.reply_text(f"{EMOJI_ERROR} Failed to calculate technical indicators for {symbol}.")
            return
            
        # Extract latest row
        latest = indicators.iloc[-1]
        current_price = safe_decimal(latest['close'])
        atr_value = safe_decimal(latest['ATR'])
        
        if current_price is None or current_price <= 0:
            logging.warning(f"[{func_name}:{symbol}] Invalid current price from indicators: {current_price}")
            # Fallback to direct API query
            current_price = await get_latest_price(client, symbol)
            if current_price <= 0:
                await update.message.reply_text(f"{EMOJI_ERROR} Could not fetch valid current price for {symbol}.")
                return
        
        logging.info(f"[{func_name}:{symbol}] Latest price: {current_price:.8f}, ATR: {atr_value if atr_value else 'N/A'}")

        # --- 6. Calculate Position Size ---
        # Calculate quantity from amount_usdc
        quantity = amount_usdc / current_price
        logging.info(f"[{func_name}:{symbol}] Calculated target quantity: {quantity:.8f}")

        # --- 7. Place Market Order ---
        base_asset_name = symbol.replace(QUOTE_ASSET, '')
        await update.message.reply_text(f"{EMOJI_BUY} Attempting MARKET buy for ~{quantity:.8f} {base_asset_name} ({amount_usdc} {QUOTE_ASSET})...")
        
        # Use the place_market_order function with ATR value for reference
        order, filled_price, executed_qty = await place_market_order(
            client, symbol, Client.SIDE_BUY, quantity, f"{strategy_type}Buy", 
            current_price_for_check=current_price, atr_value=atr_value
        )

        # --- 8. Handle Result & Update State ---
        if executed_qty is not None and executed_qty > 0:
            # Get actual cost from order details
            actual_cost_str = order.get('cummulativeQuoteQty') if order else None
            actual_cost = safe_decimal(actual_cost_str) if actual_cost_str else (filled_price * executed_qty)
            
            # Calculate take profit and stop loss prices based on strategy
            if strategy_type == "Trend":
                tp_percent = TREND_TP
                sl_percent = TREND_SL
            elif strategy_type == "MeanReversion":
                tp_percent = MEAN_REVERT_TP
                sl_percent = MEAN_REVERT_SL
            elif strategy_type == "Breakout":
                tp_percent = BREAKOUT_TP
                sl_percent = BREAKOUT_SL
            else:  # Manual
                tp_percent = BREAKOUT_TP  # Use breakout as default for manual
                sl_percent = BREAKOUT_SL
                
            # Calculate standard percentage-based TP/SL
            tp_price = filled_price * (1 + tp_percent / 100)
            
            # Use ATR-based stop loss if available, otherwise use percentage
            if atr_value is not None and atr_value > 0 and USE_ATR_SIZING:
                sl_price = calculate_atr_stop_loss(filled_price, atr_value, strategy_type)
                # Calculate effective SL percentage for display
                sl_percent_effective = ((filled_price - sl_price) / filled_price) * 100
            else:
                sl_price = filled_price * (1 - sl_percent / 100)
                sl_percent_effective = sl_percent

            await update.message.reply_text(
                f"{EMOJI_CHECK} [{strategy_type.upper()} BUY FILLED]\n"
                f"<b>Symbol:</b> {symbol}\n"
                f"<b>Qty:</b> {executed_qty:.8f}\n"
                f"<b>Avg Price:</b> {filled_price:.8f}\n"
                f"<b>Total Cost:</b> {actual_cost:.2f} {QUOTE_ASSET}\n"
                f"<b>TP:</b> {tp_price:.8f} (+{tp_percent}%)\n"
                f"<b>SL:</b> {sl_price:.8f} (-{sl_percent_effective:.2f}%)"
            )
            logging.info(f"[{func_name}:{symbol}] Manual market buy filled: Qty={executed_qty}, Price={filled_price}, Cost={actual_cost:.2f}")

            # --- Update state with the position ---
            async with state_lock:
                if state['open_position_count'] < MAX_CONCURRENT_POSITIONS:
                    # Add position details to state with trailing stop info
                    state['positions'][symbol] = {
                        'entry_price': filled_price,
                        'held_qty': executed_qty,
                        'strategy': strategy_type,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'highest_since_entry': filled_price,  # For trailing stop
                        'trailing_stop_active': False,  # Trailing stop not active initially
                        'trailing_stop_price': sl_price,  # Initialize trailing stop
                        'entry_time': time.time(),
                        'current_pnl_percent': Decimal('0.0'),
                        'initial_atr': atr_value,  # Store ATR at entry time
                        'fixed_tsl_offset': None
                    }
                    state['open_position_count'] += 1

                    # NEW: Record timestamp if this was a Breakout trade
                    if strategy_type == "Breakout":
                        state['last_breakout_times_by_symbol'][symbol] = time.time()
                        logging.info(f"[{symbol}] Breakout trade timestamp recorded for cooldown.")
                    
                    # Initialize DCA history
                    if 'dca_history' not in state:
                        state['dca_history'] = collections.defaultdict(dict)
                    state['dca_history'][symbol] = {'dca_count': 0, 'last_dca_time': 0}
                    
                    logging.info(f"[{func_name}:{symbol}] Added position to state. New count: {state['open_position_count']}")
                else:
                    logging.error(f"[{func_name}:{symbol}] Buy filled but max positions reached ({state['open_position_count']}). Position NOT added to state tracking!")
                    await update.message.reply_text(f"{EMOJI_ERROR} [{symbol}] Buy filled but max positions reached. Position not tracked internally by bot!")

        elif executed_qty == Decimal('0'):
            await update.message.reply_text(f"{EMOJI_WARNING} [{symbol}] Buy order possibly filled with 0 quantity (check exchange). Order details: {order}")
            logging.warning(f"[{func_name}:{symbol}] Buy filled with 0 qty. Order: {order}")
        else:
            await update.message.reply_text(f"{EMOJI_ERROR} [{symbol}] Failed to place market buy order. Check logs.")
            logging.error(f"[{func_name}:{symbol}] Market buy order failed (place_market_order returned None).")

    except BinanceAPIException as e:
        logging.error(f"[{func_name}:{symbol}] Binance API Error: {e}")
        await update.message.reply_text(f"{EMOJI_ERROR} API Error during /buy ({symbol}): {e.message}")
    except BinanceRequestException as e:
         logging.error(f"[{func_name}:{symbol}] Binance Request Error: {e}")
         await update.message.reply_text(f"{EMOJI_ERROR} Network/Request Error during /buy ({symbol}): {e}")
    except Exception as e:
        logging.error(f"[{func_name}:{symbol}] Unexpected Error: {e}", exc_info=True)
        await update.message.reply_text(f"{EMOJI_ERROR} An unexpected error occurred during /buy: {e}")

# --- MODIFIED: Position Management Function ---
async def check_and_manage_position(client: AsyncClient, symbol: str, indicators: pd.DataFrame, position: Dict[str, Any]) -> None:
    """
    Checks current position against TP/SL levels and manages exits.
    Prioritizes live ticker price for TP/SL checks.
    """
    price_source_log = "UNKNOWN"

    # Attempt to get live ticker price first
    live_ticker_price = await get_latest_price(client, symbol)

    # Fallback to kline close if live price failed or is invalid
    current_price = Decimal('0')
    if live_ticker_price is not None and live_ticker_price > Decimal('0'):
        current_price = live_ticker_price
        price_source_log = "LIVE_TICKER"
    elif not indicators.empty:
        kline_close_price = safe_decimal(indicators.iloc[-1]['close'])
        if kline_close_price is not None and kline_close_price > Decimal('0'):
            current_price = kline_close_price
            price_source_log = "KLINE_FALLBACK"
            logging.warning(f"[{symbol}] Used kline close price {current_price:.8f} as fallback for position management (live ticker failed).")
        else:
            logging.error(f"[{symbol}] Both live ticker and kline close price are invalid. Cannot manage position this cycle.")
            return
    else:
        logging.error(f"[{symbol}] Live ticker price failed and indicators are empty. Cannot manage position this cycle.")
        return

    # Log price discrepancy if kline data is available
    if price_source_log == "LIVE_TICKER" and not indicators.empty:
        kline_close_price_for_comparison = safe_decimal(indicators.iloc[-1]['close'])
        if kline_close_price_for_comparison is not None and kline_close_price_for_comparison > Decimal('0'):
            price_diff_percent = (abs(live_ticker_price - kline_close_price_for_comparison) / live_ticker_price) * Decimal('100')
            if price_diff_percent > PRICE_DISCREPANCY_ALERT_THRESHOLD_PERCENT:
                logging.warning(
                    f"[{symbol}] Price Discrepancy Alert: LiveTicker={live_ticker_price:.8f}, "
                    f"CachedKlineClose={kline_close_price_for_comparison:.8f} (Diff: {price_diff_percent:.2f}%)"
                )

    # Extract position details from the 'position' argument initially
    entry_price = position.get('entry_price', Decimal('0'))
    tp_price = position.get('tp_price', Decimal('0'))
    initial_sl_price = position.get('sl_price', Decimal('0'))
    qty = position.get('held_qty', Decimal('0'))
    strategy = position.get('strategy', 'Unknown')
    pos_highest_since_entry = position.get('highest_since_entry', entry_price)
    pos_trailing_stop_active = position.get('trailing_stop_active', False)
    pos_trailing_stop_price = position.get('trailing_stop_price', initial_sl_price)

    if qty <= Decimal('0'):
        logging.warning(f"[{symbol}] Position has zero or negative quantity in argument. Skipping management.")
        # Optionally, try to clean up from state if it's inconsistent
        async with state_lock:
            if symbol in state['positions'] and state['positions'][symbol].get('held_qty', Decimal('0')) <= Decimal('0'):
                logging.warning(f"[{symbol}] Removing position with zero/negative qty from state.")
                del state['positions'][symbol]
                state['open_position_count'] = max(0, state['open_position_count'] -1)
        return
    
    if entry_price <= Decimal('0'):
        logging.error(f"[{symbol}] Position has invalid entry price {entry_price}. Cannot calculate PnL or manage.")
        return

    pnl_percent = ((current_price / entry_price) - Decimal('1')) * Decimal('100')

    # --- State Update and Trailing Stop Logic ---
    async with state_lock:
        if symbol not in state['positions']:
            logging.warning(f"[{symbol}] Position not found in global state during management. Might have been closed by another process/command.")
            return

        # Update PnL in state
        state['positions'][symbol]['current_pnl_percent'] = pnl_percent

        # Get current state values for trailing stop logic
        current_pos_state = state['positions'][symbol]
        state_highest_since_entry = current_pos_state.get('highest_since_entry', entry_price)
        state_trailing_stop_active = current_pos_state.get('trailing_stop_active', False)
        state_trailing_stop_price = current_pos_state.get('trailing_stop_price', initial_sl_price)

        # Update highest price seen since entry
        if current_price > state_highest_since_entry:
            state['positions'][symbol]['highest_since_entry'] = current_price
            state_highest_since_entry = current_price
            logging.debug(f"[{symbol}] New highest_since_entry: {state_highest_since_entry:.8f}")

        # Trailing Stop Activation & Adjustment
        if TRAILING_STOP_ACTIVATION_PERCENT > 0 and TRAILING_STOP_LOCKIN_PROFIT_PERCENT > 0:
            activation_target_price = entry_price * (Decimal('1') + TRAILING_STOP_ACTIVATION_PERCENT / Decimal('100'))

            if not state_trailing_stop_active and state_highest_since_entry >= activation_target_price:
                # --- TSL ACTIVATION & INITIAL SETTING ---
                state['positions'][symbol]['trailing_stop_active'] = True
                state_trailing_stop_active = True
                logging.info(f"[{EMOJI_TRAILING}{symbol}] Trailing Stop Feature ACTIVATED. Price {state_highest_since_entry:.8f} >= Activation Target {activation_target_price:.8f}")

                # 1. Calculate the TSL price to lock in the specified profit from entry
                new_initial_tsl_price = entry_price * (Decimal('1') + TRAILING_STOP_LOCKIN_PROFIT_PERCENT / Decimal('100'))
                
                # 2. Calculate the fixed dollar offset from the current peak
                fixed_dollar_offset_to_store = state_highest_since_entry - new_initial_tsl_price
                
                # Ensure the calculated offset is not negative
                if fixed_dollar_offset_to_store < Decimal('0'):
                    logging.warning(f"[{symbol}] Calculated fixed_tsl_offset is negative ({fixed_dollar_offset_to_store:.8f}). Clamping to 0.")
                    fixed_dollar_offset_to_store = Decimal('0')
                    new_initial_tsl_price = state_highest_since_entry

                # 3. Update the position state with the new TSL price and the fixed offset
                state['positions'][symbol]['trailing_stop_price'] = new_initial_tsl_price
                state['positions'][symbol]['fixed_tsl_offset'] = fixed_dollar_offset_to_store
                state_trailing_stop_price = new_initial_tsl_price

                logging.info(f"[{EMOJI_TRAILING}{symbol}] Initial TSL SET to {new_initial_tsl_price:.8f} (Locks in {TRAILING_STOP_LOCKIN_PROFIT_PERCENT}% from entry).")
                logging.info(f"[{EMOJI_TRAILING}{symbol}] Fixed offset from activation peak ({state_highest_since_entry:.8f}) is {fixed_dollar_offset_to_store:.8f}. This offset will trail.")

            elif state_trailing_stop_active:
                # --- TSL ADJUSTMENT (if TSL is already active and a fixed offset is stored) ---
                fixed_dollar_offset = state['positions'][symbol].get('fixed_tsl_offset')

                if fixed_dollar_offset is not None and fixed_dollar_offset >= Decimal('0'):
                    # Calculate new TSL based on the most recent highest_since_entry and the stored fixed offset
                    potential_new_tsl = state_highest_since_entry - fixed_dollar_offset
                    
                    # Only move the TSL up (to a higher price)
                    if potential_new_tsl > state_trailing_stop_price:
                        state['positions'][symbol]['trailing_stop_price'] = potential_new_tsl
                        state_trailing_stop_price = potential_new_tsl
                        logging.info(f"[{EMOJI_TRAILING}{symbol}] Trailing Stop (Fixed Offset) adjusted UP to {potential_new_tsl:.8f}")
                else:
                    logging.warning(f"[{symbol}] Trailing stop is active, but 'fixed_tsl_offset' is invalid or missing ({fixed_dollar_offset}). TSL cannot be adjusted with new fixed offset logic this cycle.")

        # For exit checks below, use the potentially updated state values
        final_trailing_stop_active = state_trailing_stop_active
        final_trailing_stop_price = state_trailing_stop_price

    # --- Log position status using the definitive current_price and updated TSL details ---
    log_msg = (f"[{symbol}] ({price_source_log}) Pos Check: Price={current_price:.8f}, Entry={entry_price:.8f}, "
               f"PnL={pnl_percent:.2f}%, TP={tp_price:.8f}, ")
    if final_trailing_stop_active:
        log_msg += f"TSL={final_trailing_stop_price:.8f} (Active)"
    else:
        log_msg += f"SL={initial_sl_price:.8f} (Initial)"
    logging.info(log_msg)

    # --- Check for Exit Conditions using live current_price ---
    exit_reason_details = None

    if current_price >= tp_price:
        exit_reason_details = ("tp", f"TAKE PROFIT triggered at {current_price:.8f} >= Target {tp_price:.8f}")
    elif final_trailing_stop_active and current_price <= final_trailing_stop_price:
        exit_reason_details = ("trailing_sl", f"TRAILING STOP LOSS triggered at {current_price:.8f} <= Trail {final_trailing_stop_price:.8f}")
    elif not final_trailing_stop_active and current_price <= initial_sl_price:
        exit_reason_details = ("sl", f"INITIAL STOP LOSS triggered at {current_price:.8f} <= Stop {initial_sl_price:.8f}")

    if exit_reason_details:
        exit_type, log_message = exit_reason_details
        logging.info(f"[{symbol}] {log_message}")
        await execute_position_exit(client, symbol, qty, current_price, exit_type, entry_price, strategy)

# --- NEW: Position Exit Function ---
async def execute_position_exit(
    client: AsyncClient, 
    symbol: str, 
    qty: Decimal, 
    current_price: Decimal, 
    exit_type: str, 
    entry_price: Decimal, 
    strategy: str
) -> None:
    """
    Executes a position exit based on TP/SL and manages state.
    
    Args:
        client: AsyncClient instance
        symbol: Symbol to exit
        qty: Quantity to sell
        current_price: Current market price
        exit_type: Exit type ("tp", "sl", or "trailing_sl")
        entry_price: Original entry price
        strategy: Strategy that opened the position
    """
    if exit_type == "tp":
        exit_reason = "Take Profit"
        exit_emoji = EMOJI_SELL_TP
    elif exit_type == "trailing_sl":
        exit_reason = "Trailing Stop Loss"
        exit_emoji = f"{EMOJI_SELL_SL}{EMOJI_TRAILING}"
    else:
        exit_reason = "Stop Loss"
        exit_emoji = EMOJI_SELL_SL
        
    logging.info(f"[{symbol}] Executing {exit_reason} exit: {qty:.8f} at ~{current_price:.8f}")
    
    # Execute market sell order
    order, filled_price, executed_qty = await place_market_order(
        client, symbol, Client.SIDE_SELL, qty, strategy, current_price_for_check=current_price
    )
    
    if executed_qty is not None and executed_qty > 0:
        # Calculate profit
        profit = (filled_price - entry_price) * executed_qty
        profit_percent = ((filled_price / entry_price) - Decimal('1')) * Decimal('100')
        pnl_emoji = EMOJI_PROFIT if profit >= 0 else EMOJI_LOSS
        
        # Log and notify
        logging.info(f"[{symbol}] {exit_reason} exit executed: {executed_qty:.8f} @ {filled_price:.8f}, PnL: {profit:.2f} {QUOTE_ASSET} ({profit_percent:.2f}%)")
        
        await send_telegram_message(
            f"{exit_emoji} [{symbol}] {exit_reason} Exit ({strategy}):\n"
            f"  Sold: {executed_qty:.8f} @ {filled_price:.4f}\n"
            f"  Entry: {entry_price:.4f}\n"
            f"  PnL: {pnl_emoji} {profit:.2f} {QUOTE_ASSET} ({profit_percent:.2f}%)"
        )
        
        # Update state
        async with state_lock:
            if symbol in state['positions']:
                del state['positions'][symbol]
                state['open_position_count'] = max(0, state['open_position_count'] - 1)
                logging.info(f"[{symbol}] Position removed from state. New count: {state['open_position_count']}")
    
    elif executed_qty == Decimal('0'):
        logging.warning(f"[{symbol}] {exit_reason} order filled with 0 quantity.")
        await send_telegram_message(f"{EMOJI_WARNING} [{symbol}] {exit_reason} order filled with 0 quantity. Check exchange.")
        
        # Clean up state anyway
        async with state_lock:
            if symbol in state['positions']:
                del state['positions'][symbol]
                state['open_position_count'] = max(0, state['open_position_count'] - 1)
    
    else:
        logging.error(f"[{symbol}] Failed to execute {exit_reason} exit order.")
        await send_telegram_message(f"{EMOJI_ERROR} [{symbol}] Failed to execute {exit_reason} exit. Check logs.")

# --- UPDATED: Check Entry Signals Function with Enhanced Volume Validation ---
async def check_entry_signals(client: AsyncClient, symbol: str, indicators: pd.DataFrame) -> None:
    """
    Checks entry signals for a symbol and executes trades if signals are triggered.
    Now includes enhanced volume validation across all strategies.
    """
    if indicators.empty or len(indicators) < 2:
        logging.warning(f"[{symbol}] Not enough data to check entry signals.")
        return
    
    # Extract the latest data from 15M timeframe
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2]
    
    # Extract all needed indicators using safe_decimal
    price = safe_decimal(latest['close'])
    rsi = safe_decimal(latest['RSI'])
    adx = safe_decimal(latest['ADX'])
    ema_medium = safe_decimal(latest['EMA_medium'])
    recent_high = safe_decimal(latest['recent_high'])
    
    # Check for any missing critical indicators
    required_indicators = [price]
    if any(v is None for v in required_indicators):
        logging.warning(f"[{symbol}] Missing critical indicators for entry signal check.")
        return
    
    # ----- Enhanced Volume Validation for All Strategies -----
    volume_valid, volume_reason = enhanced_volume_validation(indicators, symbol)
    
    if not volume_valid:
        logging.debug(f"[{symbol}] Entry signals blocked by volume validation: {volume_reason}")
        return
    
    # ----- Check Strategy Signals (All now use enhanced volume validation) -----
    
    # 1. Trend Strategy - Enhanced with new volume validation
    trend_signal = await check_trend_signal(symbol, indicators)
    
    # 2. Mean Reversion Strategy - Enhanced volume validation
    mean_revert_signal = (
        rsi is not None and rsi < RSI_OVERSOLD and 
        ema_medium is not None and price < ema_medium and 
        volume_valid  # Enhanced volume validation
    )
    
    # 3. Breakout Strategy - Enhanced volume validation
    breakout_signal = False
    if recent_high is not None and rsi is not None and adx is not None:
        breakout_signal = (
            price > recent_high and
            volume_valid and  # Enhanced volume validation
            adx > ADX_BREAKOUT_THRESHOLD and
            rsi > RSI_EXIT
        )
    
    # Execute trade if any signal is triggered, with priority order:
    # 1. Breakout (highest potential), 2. Mean Reversion, 3. Trend
    usdc_balance = await get_balance(client, QUOTE_ASSET)
    
    if breakout_signal:
        last_breakout_time = state['last_breakout_times_by_symbol'].get(symbol, 0.0)
        cooldown_seconds = BREAKOUT_COOLDOWN_HOURS * 60 * 60

        if (time.time() - last_breakout_time) > cooldown_seconds:
            logging.info(f"[{symbol}] {EMOJI_VOLUME} Breakout signal confirmed for {symbol}. Volume: {volume_reason}")
            await execute_entry_signal(client, symbol, "Breakout", price, usdc_balance, safe_decimal(latest.get('ATR')))
        else:
            remaining_cooldown_seconds = cooldown_seconds - (time.time() - last_breakout_time)
            remaining_cooldown_hours = remaining_cooldown_seconds / 3600
            logging.info(f"[{symbol}] Breakout signal for {symbol} is TRUE, but SKIPPED due to active cooldown. "
                         f"Last breakout: {datetime.datetime.fromtimestamp(last_breakout_time).strftime('%Y-%m-%d %H:%M:%S')}, "
                         f"Remaining cooldown: {remaining_cooldown_hours:.2f} hours.")
    elif mean_revert_signal:
        logging.info(f"[{symbol}] {EMOJI_VOLUME} Mean Reversion signal confirmed for {symbol}. Volume: {volume_reason}")
        await execute_entry_signal(client, symbol, "MeanReversion", price, usdc_balance, safe_decimal(latest.get('ATR')))       
    elif trend_signal:
        logging.info(f"[{symbol}] {EMOJI_VOLUME} Trend signal confirmed for {symbol}. Volume: {volume_reason}")
        await execute_entry_signal(client, symbol, "Trend", price, usdc_balance, safe_decimal(latest.get('ATR')))

# --- NEW: Entry Signal Execution Function ---
async def execute_entry_signal(
    client: AsyncClient, 
    symbol: str, 
    strategy: str, 
    price: Decimal, 
    usdc_balance: Decimal,
    atr: Optional[Decimal]
) -> None:
    """
    Executes an entry based on a strategy signal with dynamic ATR-based stops.
    
    Args:
        client: AsyncClient instance
        symbol: Symbol to trade
        strategy: Strategy type ("Trend", "MeanReversion", or "Breakout")
        price: Current market price
        usdc_balance: Available USDC balance
        atr: Current ATR value if available
    """
    logging.info(f"[{symbol}] {strategy} entry signal triggered at price {price:.8f}")
    
    # Set parameters based on strategy
    if strategy == "Trend":
        position_size_percent = TREND_POSITION_SIZE
        tp_percent = TREND_TP
    elif strategy == "MeanReversion":
        position_size_percent = MEAN_REVERT_POSITION_SIZE
        tp_percent = MEAN_REVERT_TP
    else:  # Breakout
        position_size_percent = BREAKOUT_POSITION_SIZE
        tp_percent = BREAKOUT_TP
    
    # Calculate position size
    position_size_usdc = usdc_balance * position_size_percent
    
    # Standard position sizing as fallback
    qty = position_size_usdc / price
    
    # Use ATR-based sizing if enabled and ATR is available
    if USE_ATR_SIZING and atr is not None and atr > 0:
        logging.info(f"[{symbol}] Using ATR-based position sizing (ATR={atr:.8f})")
        
        # Calculate ATR-based stop loss
        sl_price = calculate_atr_stop_loss(price, atr, strategy)
        
        # Calculate stop loss distance
        sl_distance = price - sl_price
        
        # Calculate position size based on risk amount
        risk_amount = usdc_balance * position_size_percent 
        
        if sl_distance > Decimal('0'):
            # Size position based on risk per trade
            atr_qty = risk_amount / sl_distance
            
            # Use smaller of standard or ATR-based quantity for safety
            qty = min(qty, atr_qty)
            
            logging.info(f"[{symbol}] ATR-based position size: {qty:.8f} (risk amount: {risk_amount:.2f}, stop distance: {sl_distance:.8f})")
    else:
        # Calculate stop loss the traditional way based on percentage
        if strategy == "Trend":
            sl_percent = TREND_SL
        elif strategy == "MeanReversion":
            sl_percent = MEAN_REVERT_SL
        else:  # Breakout
            sl_percent = BREAKOUT_SL
            
        sl_price = price * (Decimal('1') - sl_percent / Decimal('100'))
    
    # Calculate take profit price
    tp_price = price * (Decimal('1') + tp_percent / Decimal('100'))
    
    # Execute the market buy
    await send_telegram_message(f"{EMOJI_BUY} [{strategy.upper()} SIGNAL] {symbol}: Opening position at {price:.8f}...")
    
    order, filled_price, executed_qty = await place_market_order(
        client, symbol, Client.SIDE_BUY, qty, strategy, current_price_for_check=price, atr_value=atr
    )
    
    if executed_qty is not None and executed_qty > 0:
        # Recalculate stop loss based on actual filled price
        if atr is not None and atr > 0:
            sl_price = calculate_atr_stop_loss(filled_price, atr, strategy)
        else:
            # Fallback to percentage-based if ATR not available
            if strategy == "Trend":
                sl_percent = TREND_SL
            elif strategy == "MeanReversion":
                sl_percent = MEAN_REVERT_SL
            else:  # Breakout
                sl_percent = BREAKOUT_SL
                
            sl_price = filled_price * (Decimal('1') - sl_percent / Decimal('100'))
        
        # Recalculate TP based on filled price
        tp_price = filled_price * (Decimal('1') + tp_percent / Decimal('100'))
        
        # Get actual cost
        actual_cost_str = order.get('cummulativeQuoteQty') if order else None
        actual_cost = safe_decimal(actual_cost_str) if actual_cost_str else (filled_price * executed_qty)
        
        # Calculate effective SL percentage for display
        sl_percent_effective = ((filled_price - sl_price) / filled_price) * Decimal('100')
        
        # Log and notify
        logging.info(f"[{symbol}] {strategy} entry executed: {executed_qty:.8f} @ {filled_price:.8f}, Cost: {actual_cost:.2f}")
        
        await send_telegram_message(
            f"{EMOJI_CHECK} [{strategy.upper()} ENTRY FILLED]\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Qty:</b> {executed_qty:.8f}\n"
            f"<b>Avg Price:</b> {filled_price:.8f}\n"
            f"<b>Total Cost:</b> {actual_cost:.2f} {QUOTE_ASSET}\n"
            f"<b>TP:</b> {tp_price:.8f} (+{tp_percent}%)\n"
            f"<b>SL:</b> {sl_price:.8f} (-{sl_percent_effective:.2f}%)"
        )
        
        # Update state
        async with state_lock:
            if state['open_position_count'] < MAX_CONCURRENT_POSITIONS:
                state['positions'][symbol] = {
                    'entry_price': filled_price,
                    'held_qty': executed_qty,
                    'strategy': strategy,
                    'tp_price': tp_price,
                    'sl_price': sl_price,  # Initial stop loss
                    'highest_since_entry': filled_price,  # For trailing stop
                    'trailing_stop_active': False,  # Trailing stop not active initially
                    'trailing_stop_price': sl_price,  # Initialize trailing stop
                    'entry_time': time.time(),
                    'current_pnl_percent': Decimal('0.0'),
                    'initial_atr': atr  # Store ATR at entry time for reference
                }
                state['open_position_count'] += 1
                
                # Record breakout timestamp for cooldown
                if strategy == "Breakout":
                    state['last_breakout_times_by_symbol'][symbol] = time.time()
                    logging.info(f"[{symbol}] Breakout trade timestamp recorded for cooldown.")
                
                # Initialize DCA history
                if 'dca_history' not in state:
                    state['dca_history'] = collections.defaultdict(dict)
                state['dca_history'][symbol] = {'dca_count': 0, 'last_dca_time': 0}
                
                logging.info(f"[{symbol}] Position added to state. New count: {state['open_position_count']}")
            else:
                logging.error(f"[{symbol}] Trade filled but max positions already reached. Position NOT tracked!")
                await send_telegram_message(f"{EMOJI_ERROR} [{symbol}] Trade filled but max positions already reached!")
    
    elif executed_qty == Decimal('0'):
        logging.warning(f"[{symbol}] Entry order filled with 0 quantity.")
        await send_telegram_message(f"{EMOJI_WARNING} [{symbol}] Entry order filled with 0 quantity. Check exchange.")
    
    else:
        logging.error(f"[{symbol}] Failed to execute entry order.")
        await send_telegram_message(f"{EMOJI_ERROR} [{symbol}] Failed to execute entry. Check logs.")

# --- ENHANCED: Full Portfolio Stable Up Function ---
async def execute_stable_up(client: AsyncClient, reason: str = "Unknown") -> None:
    """
    Liquidates all open bot positions AND sells holdings of watchlist base assets to QUOTE_ASSET.
    Intended for circuit breaker trips or manual stop command.
    """
    logging.info(f"--- Initiating Full Portfolio Stable Up ({reason}) ---")
    await send_telegram_message(f"{EMOJI_CIRCUIT} Initiating full portfolio stable up ({reason}). Selling positions and watchlist assets to {QUOTE_ASSET}...")

    assets_to_liquidate = set()
    symbols_to_liquidate = set()

    async with state_lock:
        # Add assets from open positions
        for symbol, pos in state['positions'].items():
            base_asset = symbol.replace(QUOTE_ASSET, '')
            assets_to_liquidate.add(base_asset)
            symbols_to_liquidate.add(symbol)
            logging.debug(f"[StableUp] Added {base_asset} from active position {symbol}")

        # Add assets from watchlist
        for symbol in state['watchlist_symbols']:
            if symbol.endswith(QUOTE_ASSET):
                base_asset = symbol.replace(QUOTE_ASSET, '')
                assets_to_liquidate.add(base_asset)
                symbols_to_liquidate.add(symbol)
                logging.debug(f"[StableUp] Added {base_asset} from watchlist symbol {symbol}")
            else:
                 logging.warning(f"[StableUp] Skipping non-{QUOTE_ASSET} symbol in watchlist: {symbol}")

    if not assets_to_liquidate:
        logging.info("[StableUp] No assets identified for liquidation (no positions or relevant watchlist items).")
        await send_telegram_message(f"{EMOJI_INFO} Stable Up: No assets needed liquidation.")
        # Still clear positions state in case it was inconsistent
        async with state_lock:
            state['positions'].clear()
            state['open_position_count'] = 0
        return

    logging.info(f"[StableUp] Assets targeted for liquidation: {', '.join(assets_to_liquidate)}")

    # --- Fetch ALL Balances ---
    all_balances = []
    try:
        await rate_limit_api()
        account_info = await client.get_account()
        all_balances = account_info.get('balances', [])
        logging.info(f"[StableUp] Fetched {len(all_balances)} asset balances.")
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logging.error(f"[StableUp] CRITICAL: Failed to fetch account balances: {e}. Liquidation may be incomplete.")
        await send_telegram_message(f"{EMOJI_ERROR} Stable Up: Failed to fetch account balances! Manual check required. Error: {e}")
        # Proceed to clear internal state anyway, but warn user
        async with state_lock:
            state['positions'].clear()
            state['open_position_count'] = 0
        return

    # --- Prepare Sell Tasks ---
    sell_tasks = []
    assets_sold_or_attempted = set()

    for balance in all_balances:
        asset = balance['asset']
        if asset in assets_to_liquidate:
            try:
                # Try to sell TOTAL balance (free + locked)
                total_qty_str = str(Decimal(balance['free']) + Decimal(balance['locked']))
                total_qty = Decimal(total_qty_str)
                assets_sold_or_attempted.add(asset)

                # Basic dust check
                if total_qty <= Decimal('1E-8'):
                    logging.debug(f"[StableUp] Skipping {asset}: Quantity {total_qty} is dust.")
                    continue

                symbol = f"{asset}{QUOTE_ASSET}"

                # Validate symbol is still tradable before attempting sell
                if symbol not in symbols_to_liquidate:
                     logging.warning(f"[StableUp] Asset {asset} balance found, but symbol {symbol} not in original positions/watchlist. Skipping sell.")
                     continue

                logging.info(f"[StableUp] Preparing to sell {total_qty} {asset} via {symbol}...")
                sell_tasks.append(asyncio.create_task(
                    place_market_order(client, symbol, Client.SIDE_SELL, total_qty, f"StableUp-{reason}", current_price_for_check=None),
                    name=f"stableup_sell_{symbol}"
                ))

            except (InvalidOperation, TypeError, KeyError, Exception) as qty_err:
                 logging.error(f"[StableUp] Error processing balance for {asset}: {qty_err}")
                 continue

    # --- Execute Selling Concurrently ---
    logging.info(f"[StableUp] Attempting to liquidate {len(sell_tasks)} asset holdings...")
    results = []
    if sell_tasks:
        results = await asyncio.gather(*sell_tasks, return_exceptions=True)
    logging.info(f"[StableUp] Liquidation attempts finished.")

    # --- Process Results and Notify ---
    success_count = 0
    fail_count = 0

    for i, task in enumerate(sell_tasks):
        asset_symbol = task.get_name().replace("stableup_sell_", "")
        base_asset = asset_symbol.replace(QUOTE_ASSET, '')
        result = results[i]

        if isinstance(result, Exception):
            fail_count += 1
            logging.error(f"[StableUp] Liquidation task for {asset_symbol} failed: {result}", exc_info=result)
            await send_telegram_message(f"{EMOJI_ERROR} [StableUp] FAILED selling {base_asset}: {result}")
        elif isinstance(result, tuple) and len(result) == 3:
            order, filled_price, executed_qty = result
            if executed_qty is not None and executed_qty > 0:
                success_count += 1
                logging.info(f"[StableUp] Successfully sold {executed_qty:.8f} {base_asset} @ {filled_price:.4f}")
                await send_telegram_message(f"{EMOJI_SELL_SL} [StableUp] Sold: {executed_qty:.8f} {base_asset} @ {filled_price:.4f}")
            elif executed_qty == Decimal('0'):
                fail_count += 1
                logging.warning(f"[StableUp] Sell order for {asset_symbol} filled 0 quantity (likely filter issue or dust).")
                await send_telegram_message(f"{EMOJI_WARNING} [StableUp] Sold 0 {base_asset} (check filters/dust).")
            else:
                fail_count += 1
                logging.error(f"[StableUp] Sell order failed for {asset_symbol} (returned None).")
                await send_telegram_message(f"{EMOJI_ERROR} [StableUp] FAILED selling {base_asset} (Order Failed).")
        else:
            fail_count += 1
            logging.error(f"[StableUp] Unexpected result for {asset_symbol}: {result}")
            await send_telegram_message(f"{EMOJI_ERROR} [StableUp] Unexpected error selling {base_asset}.")

    # Check if any targeted assets were missed
    missed_assets = assets_to_liquidate - assets_sold_or_attempted
    if missed_assets:
         logging.warning(f"[StableUp] Some targeted assets were not attempted/sold (check logs): {', '.join(missed_assets)}")
         await send_telegram_message(f"{EMOJI_WARNING} [StableUp] Note: Not all targeted assets were attempted (e.g., dust balance). Check logs.")

    # --- Final State Update and Summary ---
    async with state_lock:
        logging.info("[StableUp] Clearing internal bot position state.")
        state['positions'].clear()
        state['open_position_count'] = 0
        # Keep bot paused, requires manual /resume
        state['is_paused'] = True
        if not state['last_error']:
             state['last_error'] = f"Stable Up executed ({reason}). Manual resume required."

    summary_msg = f"{EMOJI_CIRCUIT} Stable Up ({reason}) Complete. Sold: {success_count}, Failed/Skipped: {fail_count}. Bot paused. Manual resume required."
    logging.info(summary_msg)
    await send_telegram_message(summary_msg)

async def setup_telegram_bot(client: AsyncClient):
    """Initializes and configures the Telegram bot application."""
    if not TELEGRAM_BOT_TOKEN_2 or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram BOT_TOKEN or CHAT_ID missing. Telegram features disabled.")
        return None
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN_2).build()

        # Register Command Handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("criteria", lambda update, context: criteria_command(update, context, client)))
        application.add_handler(CommandHandler("exclude", lambda update, context: exclude_command(update, context, client)))
        application.add_handler(CommandHandler("include", lambda update, context: include_command(update, context, client)))
        application.add_handler(CommandHandler("balance", lambda update, context: balance_command(update, context, client)))
        application.add_handler(CommandHandler("forcesell", lambda update, context: forcesell_command(update, context, client)))
        application.add_handler(CommandHandler("chart", lambda update, context: chart_command(update, context, client)))
        application.add_handler(CommandHandler("buy", lambda update, context: buy_market_usdc_command(update, context, client))) 
        application.add_handler(CommandHandler("pause", pause_command))
        application.add_handler(CommandHandler("resume", resume_command))
        application.add_handler(CommandHandler("stop", stop_command))

        await application.initialize()
        logging.info("Telegram bot initialized.")
        return application
    except Exception as e:
        logging.error(f"Failed to initialize Telegram bot: {e}", exc_info=True)
        return None

# --- ENHANCED: Trading Logic with Advanced Risk Management ---
async def run_trading_logic(client: AsyncClient) -> None:
    """Main loop for fetching data, checking signals, and managing positions with enhanced risk management."""
    initial_setup_done = False
    initial_baseline_value_calculated = None
    initial_usdc_value_calculated = Decimal('0')

    # --- Initial Setup Block ---
    try:
        logging.info("[run_trading_logic] Starting initial setup...")
        async with state_lock:
            logging.debug("[run_trading_logic] Initial setup lock acquired.")
            # 1. Set Initial USDC Balance
            if state['initial_usdc_balance'] == Decimal('0'):
                logging.debug("[run_trading_logic] Fetching initial USDC balance...")
                try:
                    usdc_bal = await get_balance(client, QUOTE_ASSET)
                    initial_usdc_value_calculated = usdc_bal if usdc_bal > 0 else Decimal('1.0')
                    state['initial_usdc_balance'] = initial_usdc_value_calculated
                    logging.info(f"[run_trading_logic] Initial {QUOTE_ASSET} balance recorded: {state['initial_usdc_balance']:.4f}")
                except Exception as e:
                    logging.error(f"[run_trading_logic] Failed to get initial {QUOTE_ASSET} balance: {e}. Using fallback 1.0.")
                    initial_usdc_value_calculated = Decimal('1.0')
                    state['initial_usdc_balance'] = initial_usdc_value_calculated
            else:
                initial_usdc_value_calculated = state['initial_usdc_balance']
                logging.info(f"[run_trading_logic] Using pre-existing initial USDC balance: {initial_usdc_value_calculated:.4f}")

            # 2. Set Initial Portfolio Value Baseline
            if state['initial_portfolio_value_baseline'] is None:
                watchlist_for_baseline = state['watchlist_symbols'][:]
                logging.info("[run_trading_logic] Calculating initial portfolio value baseline...")
                logging.debug(f"[run_trading_logic] Watchlist for baseline: {watchlist_for_baseline}")
            else:
                watchlist_for_baseline = []
                initial_baseline_value_calculated = state['initial_portfolio_value_baseline']
                logging.info(f"[run_trading_logic] Using pre-existing initial portfolio baseline: {initial_baseline_value_calculated:.4f}")
                initial_setup_done = True

        # --- Calculation happens OUTSIDE the initial lock ---
        if not initial_setup_done and state['initial_portfolio_value_baseline'] is None:
             try:
                  initial_value, _ = await _calculate_current_relevant_value(client, watchlist_for_baseline)
                  if initial_value > 0:
                       initial_baseline_value_calculated = initial_value
                       async with state_lock:
                            state['initial_portfolio_value_baseline'] = initial_value
                            # Initialize max equity seen with baseline value
                            state['max_equity_seen'] = initial_value
                            # Initialize equity history with baseline value
                            state['equity_history'] = [initial_value]
                       logging.info(f"[run_trading_logic] Initial portfolio value baseline recorded: {initial_value:.4f} {QUOTE_ASSET}")
                       initial_setup_done = True
                  else:
                       logging.warning("[run_trading_logic] Initial portfolio baseline calculation resulted in zero or negative value. Will retry next cycle.")
             except Exception as e:
                  logging.error(f"[run_trading_logic] Failed to calculate initial portfolio value baseline: {e}. Will retry next cycle.")

    except Exception as setup_exc:
         logging.critical(f"[run_trading_logic] CRITICAL error during initial setup: {setup_exc}", exc_info=True)
         await send_telegram_message(f"{EMOJI_ERROR} CRITICAL: Bot failed initial setup: {setup_exc}")
    
    # Send startup message only AFTER attempting setup
    try:
        async with state_lock:
            initial_usdc_val_msg = state.get('initial_usdc_balance', Decimal('0'))
            initial_baseline_val_msg = state.get('initial_portfolio_value_baseline', None)
            current_watchlist_for_msg = state.get('watchlist_symbols', [])[:]

        # Determine the baseline string representation correctly
        if initial_baseline_val_msg is not None:
            baseline_str = f"{initial_baseline_val_msg:.2f}"
        else:
            baseline_str = 'Calculating...'

        # For display, multiply by 100 to show as whole percentages
        trend_pos_display_percent = TREND_POSITION_SIZE * 100
        mean_rev_pos_display_percent = MEAN_REVERT_POSITION_SIZE * 100
        breakout_pos_display_percent = BREAKOUT_POSITION_SIZE * 100

        sizing_method_info = ""
        if USE_ATR_SIZING:
                sizing_method_info = "(Target risk per trade, size may be reduced by ATR Stop distance)"
        else:
                sizing_method_info = "(Fixed % of balance allocated per trade)"

        startup_msg = (
             f"{EMOJI_START} Enhanced Bot Started ({BINANCE_ENV}) with Volume Validation Fix.\n"
             f"Initial USDC: {initial_usdc_val_msg:.2f}\n"
             f"Initial Baseline Value: {baseline_str}\n"
             f"Monitoring: {', '.join(current_watchlist_for_msg)}\n\n"
             f"<b>✅ VOLUME VALIDATION ENHANCEMENTS:</b>\n"
             f"• Volume-Price Direction Analysis\n"
             f"• On-Balance Volume (OBV) Confirmation\n"
             f"• Anti-Falling Knife Protection\n"
             f"• Buying vs Selling Pressure Detection\n\n"
             f"<b>Optimized Parameters:</b>\n"
             f"Target Position Allocation/Risk: {trend_pos_display_percent:.2f}% (Trend), {mean_rev_pos_display_percent:.2f}% (MR), {breakout_pos_display_percent:.2f}% (BO)\n"
             f"Position Sizing Method: {sizing_method_info}\n"
             f"TP/SL - Trend: {TREND_TP}%/{TREND_SL}%\n"
             f"TP/SL - Mean Rev: {MEAN_REVERT_TP}%/{MEAN_REVERT_SL}%\n"
             f"TP/SL - Breakout: {BREAKOUT_TP}%/{BREAKOUT_SL}%\n"
             f"<b>Features:</b> ATR-Based Initial SL & Fixed Offset TSL\n"
             f"<b>Breakout Cooldown:</b> {BREAKOUT_COOLDOWN_HOURS} hours\n"
             f"<b>All Strategies:</b> 15M Timeframe"
        )
        logging.debug(f"[run_trading_logic] Attempting to send startup message:\n{startup_msg}")
        await send_telegram_message(startup_msg)
        logging.info("[run_trading_logic] Successfully sent enhanced startup message with volume validation details.")

    except Exception as tg_err:
         logging.error(f"[run_trading_logic] Failed to send initial startup message: {tg_err}", exc_info=True)

    # --- Main Trading Loop ---
    while True:
        try:
            start_cycle_time = time.time()

            # --- Re-check if baseline needs setting (if failed on first attempt) ---
            async with state_lock:
                 baseline_is_set = state['initial_portfolio_value_baseline'] is not None
                 watchlist_for_retry = state['watchlist_symbols'][:] if not baseline_is_set else []

            if not baseline_is_set:
                 logging.info("[run_trading_logic] Attempting to set portfolio baseline again (inside loop)...")
                 try:
                      initial_value, _ = await _calculate_current_relevant_value(client, watchlist_for_retry)
                      if initial_value > 0:
                           async with state_lock:
                                state['initial_portfolio_value_baseline'] = initial_value
                                # Initialize max equity seen with baseline value
                                state['max_equity_seen'] = initial_value
                                # Initialize equity history with baseline value
                                state['equity_history'] = [initial_value]
                           logging.info(f"[run_trading_logic] Initial portfolio value baseline recorded (delayed): {initial_value:.4f} {QUOTE_ASSET}")
                           await send_telegram_message(f"{EMOJI_INFO} Initial Baseline Value Recorded: {initial_value:.2f} {QUOTE_ASSET}")
                           baseline_is_set = True
                      else:
                           logging.warning("[run_trading_logic] Delayed portfolio baseline calculation resulted in zero/negative.")
                 except Exception as e:
                      logging.error(f"[run_trading_logic] Delayed portfolio baseline calculation failed: {e}")

            # --- State Check ---
            async with state_lock:
                if not state['is_running']:
                    logging.info("[run_trading_logic] Stop signal received. Exiting trading loop.")
                    break
                if state['is_paused']:
                    logging.debug("[run_trading_logic] Bot is paused. Skipping trading cycle.")
                    await asyncio.sleep(max(5, SLEEP_INTERVAL_SECONDS // 5))
                    continue

                current_watchlist = [s for s in state['watchlist_symbols'] if s not in state['excluded_symbols']]
                current_positions = state['positions'].copy()
                current_open_pos_count = state['open_position_count']
                initial_usdc_balance = state['initial_usdc_balance']
                
                # Get max equity seen for drawdown calculation
                max_equity_seen = state.get('max_equity_seen', Decimal('0'))

            # --- Portfolio Valuation & Drawdown Monitoring ---
            # Calculate current portfolio value
            current_usdc_balance = await get_balance(client, QUOTE_ASSET)
            current_portfolio_value, _ = await _calculate_current_relevant_value(client, current_watchlist)
            
            # Update equity history and max seen
            async with state_lock:
                state['equity_history'].append(current_portfolio_value)
                # Keep only last N points to prevent unlimited growth
                if len(state['equity_history']) > MAX_EQUITY_DRAWDOWN_TRACKING:
                    state['equity_history'] = state['equity_history'][-MAX_EQUITY_DRAWDOWN_TRACKING:]
                
                # Update max equity if we have a new high
                if current_portfolio_value > state['max_equity_seen']:
                    state['max_equity_seen'] = current_portfolio_value
                    logging.info(f"New portfolio value high: {current_portfolio_value:.2f} {QUOTE_ASSET}")
                
                # Get updated max for current calculation
                max_equity_seen = state['max_equity_seen']
            
            # Calculate current drawdown
            current_drawdown = Decimal('0')
            if max_equity_seen > Decimal('0'):
                current_drawdown = calculate_current_drawdown(current_portfolio_value, max_equity_seen)
                logging.info(f"Current drawdown: {current_drawdown:.2f}% (Current: {current_portfolio_value:.2f}, Max: {max_equity_seen:.2f})")

            # --- Circuit Breaker Check (Both Balance and Drawdown) ---
            
            # 1. Balance-based circuit breaker
            initial_baseline_value = state.get('initial_portfolio_value_baseline')
            portfolio_ratio = Decimal('1.0')
            if initial_baseline_value is not None and initial_baseline_value > Decimal('0'):
                portfolio_ratio = current_portfolio_value / initial_baseline_value

            # Check the condition AFTER calculating ratio safely
            if initial_baseline_value is not None and portfolio_ratio < LOSS_CIRCUIT_BREAKER_THRESHOLD:
                logging.critical(f"[run_trading_logic] PORTFOLIO CIRCUIT BREAKER TRIPPED! Current value {current_portfolio_value:.2f} < Threshold ({LOSS_CIRCUIT_BREAKER_THRESHOLD*100}%) of Initial {initial_baseline_value:.2f}")
                error_msg = f"Portfolio Value Circuit Breaker ({portfolio_ratio*100:.2f}% of initial)"
                # Set error message first
                async with state_lock:
                     state['last_error'] = error_msg
                # Call the stable up function to liquidate and pause
                await execute_stable_up(client, reason=error_msg)
                continue
            # 2. Drawdown-based circuit breaker
            if current_drawdown < -DRAWDOWN_CIRCUIT_BREAKER: # Drawdown is negative, threshold is positive
                logging.critical(f"[run_trading_logic] DRAWDOWN CIRCUIT BREAKER TRIPPED! Current drawdown {current_drawdown:.2f}% exceeds threshold {DRAWDOWN_CIRCUIT_BREAKER}%")
                error_msg = f"Drawdown Circuit Breaker ({current_drawdown:.2f}% > {DRAWDOWN_CIRCUIT_BREAKER}%)"
                # Set error message first
                async with state_lock:
                     state['last_error'] = error_msg
                # Call the stable up function to liquidate and pause
                await execute_stable_up(client, reason=error_msg)
                continue

            logging.info(f"--- Cycle Start --- Open: {current_open_pos_count}/{MAX_CONCURRENT_POSITIONS} | USDC: {current_usdc_balance:.2f} | Drawdown: {current_drawdown:.2f}% | Monitoring: {len(current_watchlist)} ---")

            # --- Process All Relevant Symbols: Both Positions and Watchlist ---
            all_symbols_to_process = list(set(current_watchlist + list(current_positions.keys())))
            
            # Early exit if no symbols to process
            if not all_symbols_to_process:
                logging.warning("No symbols to process (empty watchlist and no positions). Skipping cycle.")
                await asyncio.sleep(SLEEP_INTERVAL_SECONDS)
                continue
                
            # Fetch data for all symbols concurrently
            symbol_data_tasks = {}
            for symbol in all_symbols_to_process:
                symbol_data_tasks[symbol] = asyncio.create_task(
                    get_latest_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
                )
            
            # Wait for all data tasks to complete
            await asyncio.gather(*symbol_data_tasks.values(), return_exceptions=True)
            
            # Process each symbol's data
            for symbol in all_symbols_to_process:
                try:
                    # Get result from finished task
                    df_task = symbol_data_tasks[symbol]
                    if df_task.exception():
                        logging.error(f"[{symbol}] Error fetching data: {df_task.exception()}")
                        continue
                        
                    df = df_task.result()
                    if df.empty:
                        logging.warning(f"[{symbol}] Empty dataframe returned. Skipping.")
                        continue
                        
                    # Calculate indicators
                    indicators = calculate_indicators(df)
                    if indicators.empty:
                        logging.warning(f"[{symbol}] Failed to calculate indicators. Skipping.")
                        continue
                    
                    # Check if this is a position or watchlist symbol
                    is_position = symbol in current_positions
                    
                    if is_position:
                        # --- POSITION MANAGEMENT ---
                        position = current_positions[symbol]
                        await check_and_manage_position(client, symbol, indicators, position)
                    else:
                        # --- SIGNAL SCANNING (only if we have room for new positions) ---
                        if current_open_pos_count < MAX_CONCURRENT_POSITIONS and not state['is_paused']:
                            await check_entry_signals(client, symbol, indicators)
                
                except Exception as symbol_e:
                    logging.error(f"[{symbol}] Error processing symbol: {symbol_e}", exc_info=True)

            # --- Cycle End Logging ---
            cycle_duration = time.time() - start_cycle_time
            logging.info(f"--- Cycle End ({cycle_duration:.2f}s) ---")

            # --- Sleep ---
            sleep_needed = SLEEP_INTERVAL_SECONDS - cycle_duration
            if sleep_needed > 0:
                logging.debug(f"Sleeping for {sleep_needed:.2f}s...")
                await asyncio.sleep(sleep_needed)
            else:
                logging.warning("Cycle duration exceeded sleep interval.")

        except Exception as loop_error:
            logging.critical(f"[run_trading_logic] CRITICAL ERROR IN TRADING LOOP: {loop_error} !!!", exc_info=True)
            await send_telegram_message(f"{EMOJI_ERROR} CRITICAL Error in trading loop: {loop_error}. Check logs!")
            async with state_lock:
                 state['last_error'] = f"Loop Error: {loop_error}"
            await asyncio.sleep(SLEEP_INTERVAL_SECONDS * 2)

async def main():
    """Main entry point for the trading bot."""
    logging.info(f"--- Starting Enhanced Trading Bot with Volume Validation Fix --- ({BINANCE_ENV})")
    client = await initialize_binance_client()

    if not client:
        logging.critical("Failed to initialize Binance client. Exiting.")
        if TELEGRAM_BOT_TOKEN_2 and TELEGRAM_CHAT_ID:
            try:
                async with Bot(token=TELEGRAM_BOT_TOKEN_2) as temp_bot:
                    await temp_bot.send_message(chat_id=int(TELEGRAM_CHAT_ID), text=f"{EMOJI_ERROR} CRITICAL: Failed to initialize Binance client. Bot cannot start.")
            except Exception as tg_init_e:
                logging.error(f"Failed to send initial Telegram error message: {tg_init_e}")
        sys.exit(1)

    telegram_app = await setup_telegram_bot(client)
    if not telegram_app:
        logging.warning("Proceeding without Telegram features.")

    trading_task = None
    try:
        # Start Telegram polling if initialized
        if telegram_app:
            logging.info("Starting Telegram bot polling...")
            await telegram_app.start()
            await telegram_app.updater.start_polling(drop_pending_updates=True)

        # Start the main trading logic
        logging.info("Starting main trading loop...")
        trading_task = asyncio.create_task(run_trading_logic(client))

        # Keep main running while trading_task is active
        await trading_task

    except asyncio.CancelledError:
        logging.info("Main task cancelled, initiating shutdown...")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Initiating shutdown...")
        async with state_lock:
            state['is_running'] = False
    except Exception as e:
        logging.critical(f"Unhandled exception in main execution block: {e}", exc_info=True)
        await send_telegram_message(f"{EMOJI_ERROR} CRITICAL: Bot crashed unexpectedly in main: {e}")
        async with state_lock:
            state['is_running'] = False

    finally:
        logging.info("--- Initiating Graceful Shutdown ---")
        # 1. Signal trading loop to stop
        async with state_lock:
            state['is_running'] = False

        # 2. Wait briefly for loop task (if it exists and hasn't finished)
        if trading_task and not trading_task.done():
            logging.info("Waiting briefly for trading loop task to finish...")
            try:
                await asyncio.wait_for(trading_task, timeout=max(5.0, SLEEP_INTERVAL_SECONDS / 2.0))
            except asyncio.TimeoutError:
                logging.warning("Trading loop did not finish gracefully in time, cancelling task...")
                trading_task.cancel()
                try:
                    await trading_task
                except asyncio.CancelledError:
                    logging.info("Trading loop task cancelled.")
            except Exception as task_wait_e:
                logging.error(f"Error waiting for trading task: {task_wait_e}")

        # 3. Stop Telegram polling
        if telegram_app and telegram_app.updater:
            logging.info("Stopping Telegram bot polling...")
            if getattr(telegram_app.updater, 'running', False):
                try:
                    await telegram_app.updater.stop()
                    logging.info("Telegram bot polling stopped successfully.")
                except Exception as updater_stop_e:
                    logging.warning(f"Error stopping Telegram updater: {updater_stop_e}")

        # 4. Liquidate remaining positions & Stable Up
        if client:
            logging.info("Executing Stable Up procedure on shutdown...")
            try:
                await execute_stable_up(client, reason="Bot Shutdown")
                logging.info("Stable Up procedure completed.")
            except Exception as cleanup_e:
                logging.error(f"Error during execute_stable_up on shutdown: {cleanup_e}")

        # 5. Stop Telegram application
        if telegram_app:
            logging.info("Stopping Telegram application...")
            try:
                await telegram_app.stop()
                logging.info("Telegram application stopped.")
            except Exception as tg_stop_e:
                logging.warning(f"Error during telegram_app.stop(): {tg_stop_e}")
            try:
                await telegram_app.shutdown()
                logging.info("Telegram application shutdown completed.")
            except Exception as tg_shut_e:
                logging.warning(f"Error during telegram_app.shutdown(): {tg_shut_e}")

        # 6. Close Binance client connection
        if client:
            logging.info("Closing Binance client connection...")
            try:
                await client.close_connection()
                logging.info("Binance client connection closed successfully.")
            except Exception as cl_close_e:
                logging.error(f"Error closing client connection: {cl_close_e}")

        logging.info("--- Enhanced Bot shutdown complete ---")

if __name__ == "__main__":
    # Ensure asyncio event loop runs the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as main_run_e:
        logging.critical(f"Fatal error running asyncio.run(main): {main_run_e}", exc_info=True)