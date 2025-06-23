# Crypto Backtesting System
#
# This script performs a backtest of multiple trading strategies on cryptocurrency
# data, calculates key performance metrics, and generates an interactive HTML report.
#
# ENHANCEMENTS:
# - CHANGED: Replaced Symbol PnL Pie Chart with a clear Bar Chart to show winners & losers.
# - ADDED: Stop Loss Performance analysis to show max favorable excursion before stop out.
# - ADDED: Capital Utilization analysis to measure idle capital.
# - ADDED: Breakout Cooldown logic to prevent re-entering trades too quickly.
# - ADDED: Missed Profit Analysis to measure potential upside after trade exits.
# - ADDED: Granular, per-strategy controls for volume validation filters.
# - ADDED: Parameter Tuning Insights table in the report to guide optimization.
# - ADDED: Full Trailing Stop Loss (TSL) logic.
#
# Requirements:
# pip install pandas python-binance pandas-ta plotly

import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# --- CONFIGURATION ---
# 1. API Credentials (Optional)
client = Client() # Public client is sufficient for historical data

# 2. Backtest Parameters
INITIAL_CAPITAL = 100000
START_DATE = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d") # 6 months ago
END_DATE = datetime.now().strftime("%Y-%m-%d") # Today
WATCHLIST = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'SUIUSDT', 'XRPUSDT']
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE

# 3. Strategy & Volume Parameters
# Grouped for easy optimization. You can experiment by changing these values.
STRATEGY_PARAMS = {
    # --- Trailing Stop Loss Parameters ---
    'TrailingStop': {
        'use_trailing_stop': True,
        'activation_percent': 1.3 / 100,
        'lockin_profit_percent': 0.7 / 100 
    },
    # --- Strategy-Specific Parameters & Volume Controls ---
    'Trend': {
        'active': True,
        'use_direction_check': True,
        'use_obv_check': False,
        'ema_period': 200, 'rsi_period': 14, 'rsi_threshold': 30, 'bbands_period': 20,
        'tp_pct': 2.4 / 100, 'sl_pct': 2.7 / 100, 'position_size': 0.35,
    },
    'MeanReversion': {
        'active': True,
        'use_direction_check': True,
        'use_obv_check': True,
        'ema_period': 50, 'rsi_period': 14, 'rsi_threshold': 24, # Optimized from 26
        'tp_pct': 2.5 / 100, 'sl_pct': 2.7 / 100, 'position_size': 0.35, # TP & Size increased
    },
    'Breakout': {
        'active': True,
        'use_direction_check': True,
        'use_obv_check': True,
        'high_period': 48, 'adx_period': 14, 'adx_threshold': 30, # Optimized from 25
        'rsi_period': 14, 'rsi_threshold': 55,
        'tp_pct': 3.5 / 100, 'sl_pct': 2.8 / 100, 'position_size': 0.35, 
        'cooldown_hours': 12, 
    },
    # --- Global Volume Parameters ---
    'Volume': {
        'obv_periods': 3,
        'volume_sma_period': 20, 'price_change_threshold': 0.002, 'volume_ratio_threshold': 1.2,
        'obv_rising_pct': 0.6
    },
    # --- Analysis Parameters ---
    'Analysis': {
        'missed_profit_lookahead_periods': 24 # How many candles (15min) to look ahead after exit
    }
}

# --- DATA FETCHING ---
def fetch_data(symbol, timeframe, start_str, end_str):
    """Fetches OHLCV data from Binance."""
    print(f"Fetching {symbol} data from {start_str} to {end_str}...")
    try:
        klines = client.get_historical_klines(symbol, timeframe, start_str, end_str)
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        return data[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"An error occurred fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- INDICATOR CALCULATION ---
def calculate_indicators(df):
    """Calculates all necessary technical indicators."""
    print(f"Calculating indicators for {df.index.min()} to {df.index.max()}...")
    # Trend
    trend_params = STRATEGY_PARAMS['Trend']
    df['ema_trend'] = ta.ema(df['close'], length=trend_params['ema_period'])
    df['rsi_trend'] = ta.rsi(df['close'], length=trend_params['rsi_period'])
    bbands = ta.bbands(df['close'], length=trend_params['bbands_period'])
    df['bb_lower'] = bbands[f'BBL_{trend_params["bbands_period"]}_2.0']
    # Mean Reversion
    mr_params = STRATEGY_PARAMS['MeanReversion']
    df['ema_mr'] = ta.ema(df['close'], length=mr_params['ema_period'])
    df['rsi_mr'] = ta.rsi(df['close'], length=mr_params['rsi_period'])
    # Breakout
    bo_params = STRATEGY_PARAMS['Breakout']
    df['period_high'] = df['high'].rolling(window=bo_params['high_period']).max().shift(1)
    adx = ta.adx(df['high'], df['low'], df['close'], length=bo_params['adx_period'])
    df['adx'] = adx[f'ADX_{bo_params["adx_period"]}']
    df['rsi_bo'] = ta.rsi(df['close'], length=bo_params['rsi_period'])
    # Volume
    vol_params = STRATEGY_PARAMS['Volume']
    df['volume_sma'] = ta.sma(df['volume'], length=vol_params['volume_sma_period'])
    df.ta.obv(close='close', volume='volume', append=True, col_names=('OBV',))

    df.dropna(inplace=True)
    return df

# --- VOLUME VALIDATION LOGIC ---
def analyze_volume_direction(df_slice, vol_params):
    if len(df_slice) < 2: return "INSUFFICIENT_DATA"
    current, prev = df_slice.iloc[-1], df_slice.iloc[-2]
    price_change = (current['close'] - prev['close']) / prev['close']
    volume_ratio = current['volume'] / current['volume_sma'] if current['volume_sma'] > 0 else 1
    if volume_ratio > vol_params['volume_ratio_threshold']:
        if price_change > vol_params['price_change_threshold']: return "BUYING_VOLUME"
        if price_change < -vol_params['price_change_threshold']: return "SELLING_VOLUME"
        return "NEUTRAL_VOLUME"
    return "LOW_VOLUME"

def validate_obv_trend(df_slice, vol_params):
    periods = vol_params['obv_periods']
    if len(df_slice) < periods + 1: return False
    obv_values = df_slice['OBV'].tail(periods + 1).values
    rising_count = sum(1 for i in range(1, len(obv_values)) if obv_values[i] > obv_values[i-1])
    return rising_count >= (periods * vol_params['obv_rising_pct'])

def enhanced_volume_validation(df_slice, vol_params):
    """Performs volume validation based on the provided strategy-specific parameters."""
    if vol_params.get('use_direction_check', True):
        volume_direction = analyze_volume_direction(df_slice, vol_params)
        if volume_direction not in ["BUYING_VOLUME", "NEUTRAL_VOLUME"]:
            return False, f"Failed Direction Check: {volume_direction}"
    if vol_params.get('use_obv_check', True):
        if not validate_obv_trend(df_slice, vol_params):
            return False, "Failed OBV Trend Check"
    return True, "Volume validation passed"


# --- BACKTESTING ENGINE WITH TSL ---
def run_backtest(data_dict):
    """Runs the backtest simulation with integrated Trailing Stop Loss (TSL)."""
    print("Starting backtest simulation with TSL...")
    capital = INITIAL_CAPITAL
    equity_curve = {list(data_dict.values())[0].index[0]: capital}
    trades = []
    open_positions = {}
    last_breakout_times = {} 
    tsl_params = STRATEGY_PARAMS['TrailingStop']
    capital_deployment_history = [] 

    master_index = sorted(list(data_dict.values())[0].index)

    for i, timestamp in enumerate(master_index):
        if i == 0: continue
        current_equity = capital + sum(p['pnl'] for p in open_positions.values())
        
        invested_capital_sum = sum(pos['invested'] for pos in open_positions.values())
        deployment_pct = (invested_capital_sum / current_equity) * 100 if current_equity > 0 else 0
        capital_deployment_history.append(deployment_pct)

        # Check for exits and manage TSL first
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            if timestamp not in data_dict[symbol].index: continue
            
            current_price = data_dict[symbol].loc[timestamp]['close']
            exit_reason = None
            exit_price = current_price

            # --- TSL Management Logic ---
            if tsl_params['use_trailing_stop']:
                if current_price > pos['highest_since_entry']:
                    pos['highest_since_entry'] = current_price
                activation_target = pos['entry_price'] * (1 + tsl_params['activation_percent'])
                if not pos['trailing_stop_active'] and pos['highest_since_entry'] >= activation_target:
                    pos['trailing_stop_active'] = True
                    new_tsl_price = pos['entry_price'] * (1 + tsl_params['lockin_profit_percent'])
                    pos['fixed_tsl_offset'] = pos['highest_since_entry'] - new_tsl_price
                    pos['trailing_stop_price'] = new_tsl_price
                    print(f"  ** TSL Activated for {symbol} at {new_tsl_price:.4f}")
                if pos['trailing_stop_active']:
                    potential_new_tsl = pos['highest_since_entry'] - pos['fixed_tsl_offset']
                    if potential_new_tsl > pos['trailing_stop_price']:
                        pos['trailing_stop_price'] = potential_new_tsl
                    pos['stop_loss_price'] = pos['trailing_stop_price']
            
            # --- Exit Conditions Check ---
            pnl = (current_price - pos['entry_price']) * pos['amount']
            pos['pnl'] = pnl
            if current_price >= pos['take_profit_price']:
                exit_reason = "TP"
                exit_price = pos['take_profit_price']
            elif current_price <= pos['stop_loss_price']:
                exit_reason = "TSL" if pos.get('trailing_stop_active') else "SL"
                exit_price = pos['stop_loss_price']

            if exit_reason:
                trade_pnl = (exit_price - pos['entry_price']) * pos['amount']
                capital += trade_pnl
                
                lookahead = STRATEGY_PARAMS['Analysis']['missed_profit_lookahead_periods']
                future_slice = data_dict[symbol].loc[timestamp:].head(lookahead)
                peak_price_after_exit = future_slice['high'].max() if not future_slice.empty else exit_price
                missed_profit_pct = ((peak_price_after_exit / exit_price) - 1) * 100 if exit_price > 0 else 0
                
                mfe_pct = ((pos['highest_since_entry'] / pos['entry_price']) - 1) * 100 if pos['entry_price'] > 0 else 0
                
                trades.append({
                    'symbol': symbol, 'strategy': pos['strategy'], 'exit_reason': exit_reason,
                    'entry_time': pos['entry_time'], 'exit_time': timestamp, 'entry_price': pos['entry_price'], 
                    'exit_price': exit_price, 'pnl': trade_pnl, 'return_pct': (trade_pnl / pos['invested']) * 100,
                    'entry_indicators': pos['entry_indicators'], 'missed_profit_pct': missed_profit_pct,
                    'mfe_pct': mfe_pct
                })
                
                if pos['strategy'] == 'Breakout':
                    last_breakout_times[symbol] = timestamp 
                
                del open_positions[symbol]

        # Check for new entries
        for symbol, df in data_dict.items():
            if symbol in open_positions or timestamp not in df.index: continue
            row = df.loc[timestamp]
            df_slice = df.loc[:timestamp]
            
            # Trend Strategy
            trend_params = STRATEGY_PARAMS['Trend']
            if trend_params['active'] and row['close'] > row['ema_trend'] and row['rsi_trend'] <= trend_params['rsi_threshold'] and row['close'] < row['bb_lower']:
                trend_vol_params = {**STRATEGY_PARAMS['Volume'], **trend_params}
                volume_valid, _ = enhanced_volume_validation(df_slice, trend_vol_params)
                if volume_valid:
                    enter_trade(symbol, 'Trend', row, timestamp, open_positions, current_equity)

            # Mean Reversion Strategy
            mr_params = STRATEGY_PARAMS['MeanReversion']
            if symbol not in open_positions and mr_params['active'] and row['rsi_mr'] < mr_params['rsi_threshold'] and row['close'] < row['ema_mr']:
                 mr_vol_params = {**STRATEGY_PARAMS['Volume'], **mr_params}
                 volume_valid, _ = enhanced_volume_validation(df_slice, mr_vol_params)
                 if volume_valid:
                    enter_trade(symbol, 'MeanReversion', row, timestamp, open_positions, current_equity)

            # Breakout Strategy
            bo_params = STRATEGY_PARAMS['Breakout']
            cooldown_period = timedelta(hours=bo_params.get('cooldown_hours', 0))
            if symbol in last_breakout_times and (timestamp - last_breakout_times[symbol]) < cooldown_period:
                continue
            if symbol not in open_positions and bo_params['active'] and row['close'] > row['period_high'] and row['adx'] > bo_params['adx_threshold'] and row['rsi_bo'] > bo_params['rsi_threshold']:
                bo_vol_params = {**STRATEGY_PARAMS['Volume'], **bo_params}
                volume_valid, _ = enhanced_volume_validation(df_slice, bo_vol_params)
                if volume_valid:
                    enter_trade(symbol, 'Breakout', row, timestamp, open_positions, current_equity)

        if i > 0 and master_index[i].date() != master_index[i-1].date():
             equity_curve[timestamp] = current_equity

    final_equity = capital + sum(p['pnl'] for p in open_positions.values())
    print(f"Backtest finished. Final Equity: ${final_equity:,.2f}")
    return pd.DataFrame(trades), pd.Series(equity_curve, name="Equity"), capital_deployment_history

def enter_trade(symbol, strategy_name, row, timestamp, open_positions, current_equity):
    """Helper function to create a new trade position with TSL fields and entry indicators."""
    params = STRATEGY_PARAMS[strategy_name]
    entry_price = row['close']
    invested_capital = current_equity * params['position_size']
    amount = invested_capital / entry_price
    initial_sl_price = entry_price * (1 - params['sl_pct'])
    
    entry_indicators = {}
    if strategy_name == 'Trend': entry_indicators['RSI'] = row['rsi_trend']
    elif strategy_name == 'MeanReversion': entry_indicators['RSI'] = row['rsi_mr']
    elif strategy_name == 'Breakout':
        entry_indicators['RSI'] = row['rsi_bo']
        entry_indicators['ADX'] = row['adx']

    open_positions[symbol] = {
        'strategy': strategy_name, 'entry_price': entry_price, 'amount': amount,
        'entry_time': timestamp, 'pnl': 0, 'invested': invested_capital,
        'take_profit_price': entry_price * (1 + params['tp_pct']),
        'stop_loss_price': initial_sl_price, 'highest_since_entry': entry_price,
        'trailing_stop_active': False, 'trailing_stop_price': initial_sl_price,
        'fixed_tsl_offset': None, 'entry_indicators': entry_indicators
    }
    print(f"  -> ENTER {strategy_name} on {symbol} at ${entry_price:.2f} on {timestamp.date()}")

# --- PERFORMANCE ANALYSIS ---
def analyze_performance(trades_df, equity_curve, capital_deployment_history):
    """Calculates key performance metrics and parameter tuning insights."""
    if trades_df.empty: return {}
    
    total_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['pnl'] > 0]) / total_trades) * 100
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(35040) if returns.std() != 0 else 0
    avg_capital_deployed_pct = np.mean(capital_deployment_history) if capital_deployment_history else 0
    avg_idle_capital_pct = 100 - avg_capital_deployed_pct
    symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().to_dict()

    strategy_analysis = {}
    param_insights = {}

    for name in ['Trend', 'MeanReversion', 'Breakout']:
        strategy_trades = trades_df[trades_df['strategy'] == name].copy()
        if strategy_trades.empty: continue

        s_total = len(strategy_trades)
        s_wins = len(strategy_trades[strategy_trades['pnl'] > 0])
        exit_counts = strategy_trades['exit_reason'].value_counts().to_dict()
        sl_trades = strategy_trades[strategy_trades['exit_reason'] == 'SL']
        tsl_trades = strategy_trades[strategy_trades['exit_reason'] == 'TSL']
        avg_mfe_on_sl = sl_trades['mfe_pct'].mean() if not sl_trades.empty else 0
        avg_mfe_on_tsl = tsl_trades['mfe_pct'].mean() if not tsl_trades.empty else 0

        strategy_analysis[name] = {
            "Total Trades": s_total, "Win Rate (%)": (s_wins / s_total) * 100,
            "Total PnL ($)": strategy_trades['pnl'].sum(), "Avg. Return (%)": strategy_trades['return_pct'].mean(),
            "TPs": exit_counts.get("TP", 0), "SLs": exit_counts.get("SL", 0), "TSLs": exit_counts.get("TSL", 0),
            "Avg Missed Profit (%)": strategy_trades['missed_profit_pct'].mean(),
            "Avg MFE on SL (%)": avg_mfe_on_sl, "Avg MFE on TSL (%)": avg_mfe_on_tsl
        }

        try:
            indicators_df = pd.json_normalize(strategy_trades['entry_indicators'].dropna())
            strategy_trades = strategy_trades.join(indicators_df)
        except Exception: continue
        
        insights = {}
        if 'RSI' in strategy_trades.columns:
            insights['RSI'] = {'Win': strategy_trades[strategy_trades['pnl'] > 0]['RSI'].mean(), 'Loss': strategy_trades[strategy_trades['pnl'] <= 0]['RSI'].mean()}
        if 'ADX' in strategy_trades.columns:
            insights['ADX'] = {'Win': strategy_trades[strategy_trades['pnl'] > 0]['ADX'].mean(), 'Loss': strategy_trades[strategy_trades['pnl'] <= 0]['ADX'].mean()}
        if insights: param_insights[name] = insights
            
    return {
        "Total PnL ($)": trades_df['pnl'].sum(), "Win Rate (%)": win_rate, "Total Trades": total_trades,
        "Average Return (%)": trades_df['return_pct'].mean(), "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio (Ann.)": sharpe_ratio, "Avg Idle Capital (%)": avg_idle_capital_pct,
        "strategy_analysis": strategy_analysis, "param_insights": param_insights,
        "symbol_pnl": symbol_pnl
    }

# --- REPORTING ---
def generate_html_report(trades_df, equity_curve, performance_metrics, filename="backtest_report_tsl.html"):
    """Generates an interactive HTML report with TSL stats and parameter insights."""
    print(f"Generating HTML report: {filename}")
    fig = make_subplots(
        rows=7, cols=2,
        specs=[
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "table"}, {"type": "pie"}],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "bar", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None]
        ],
        subplot_titles=(
            "Equity Curve", "Overall Performance", "Trade Distribution by Strategy",
            "Performance per Strategy", "Parameter Tuning Insights", 
            "Profitability by Symbol", "Exit & Missed Profit Analysis", 
            "Stop Loss Performance"
        ),
        vertical_spacing=0.06, row_heights=[0.28, 0.14, 0.14, 0.14, 0.14, 0.08, 0.08]
    )
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity', line=dict(color='#1f77b4')), row=1, col=1)

    metrics_display = {k: v for k, v in performance_metrics.items() if k not in ['strategy_analysis', 'param_insights', 'symbol_pnl']}
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color='#cce5ff', align='left', font=dict(color='black')),
        cells=dict(values=[list(metrics_display.keys()), [f"{v:,.2f}" for v in metrics_display.values()]], fill_color='#f0f8ff', align='left')
    ), row=2, col=1)

    if not trades_df.empty:
        strategy_counts = trades_df['strategy'].value_counts()
        fig.add_trace(go.Pie(labels=strategy_counts.index, values=strategy_counts.values, name="Strategies", hole=.3), row=2, col=2)

    if 'strategy_analysis' in performance_metrics and performance_metrics['strategy_analysis']:
        strat_analysis = performance_metrics['strategy_analysis']
        header_vals = ["Strategy", "Trades", "Win %", "PnL $", "Avg Ret %", "TPs", "SLs", "TSLs"]
        cell_values = [list(strat_analysis.keys())]
        for key in ["Total Trades", "Win Rate (%)", "Total PnL ($)", "Avg. Return (%)", "TPs", "SLs", "TSLs"]:
            cell_values.append([f"{strat.get(key, 0):,.2f}" for strat in strat_analysis.values()])
        fig.add_trace(go.Table(header=dict(values=header_vals, fill_color='#cce5ff', align='left', font=dict(color='black')), cells=dict(values=cell_values, fill_color='#f0f8ff', align='left')), row=3, col=1)
    
    if 'param_insights' in performance_metrics and performance_metrics['param_insights']:
        insights_data = performance_metrics['param_insights']
        strat_col, param_col, win_col, loss_col = [], [], [], []
        for strat_name, params in insights_data.items():
            for param_name, values in params.items():
                strat_col.append(strat_name); param_col.append(param_name)
                win_col.append(f"{values.get('Win', float('nan')):.2f}"); loss_col.append(f"{values.get('Loss', float('nan')):.2f}")
        header_vals = ["Strategy", "Parameter", "Avg Value on Wins", "Avg Value on Losers"]
        fig.add_trace(go.Table(header=dict(values=header_vals, fill_color='#d4edda', align='left', font=dict(color='black')), cells=dict(values=[strat_col, param_col, win_col, loss_col], fill_color='#f3fcf5', align='left')), row=4, col=1)

    # --- New Symbol PnL Bar Chart ---
    if 'symbol_pnl' in performance_metrics and performance_metrics['symbol_pnl']:
        symbol_pnl = performance_metrics['symbol_pnl']
        symbols = list(symbol_pnl.keys())
        pnl_values = list(symbol_pnl.values())
        colors = ['#28a745' if p > 0 else '#dc3545' for p in pnl_values]
        
        fig.add_trace(go.Bar(
            x=symbols, 
            y=pnl_values,
            marker_color=colors,
            name="Symbol PnL"), row=5, col=1)

    if 'strategy_analysis' in performance_metrics and performance_metrics['strategy_analysis']:
        strat_analysis = performance_metrics['strategy_analysis']
        header_vals = ["Strategy", "Avg Missed Profit (%)"]
        cell_values = [list(strat_analysis.keys()), [f"{strat.get('Avg Missed Profit (%)', 0):.2f}%" for strat in strat_analysis.values()]]
        fig.add_trace(go.Table(header=dict(values=header_vals, fill_color='#ffeeba', align='left', font=dict(color='black')), cells=dict(values=cell_values, fill_color='#fffcf5', align='left')), row=6, col=1)

    if 'strategy_analysis' in performance_metrics and performance_metrics['strategy_analysis']:
        strat_analysis = performance_metrics['strategy_analysis']
        header_vals = ["Strategy", "Avg MFE on SL (%)", "Avg MFE on TSL (%)"]
        cell_values = [list(strat_analysis.keys()), [f"{strat.get('Avg MFE on SL (%)', 0):.2f}%" for strat in strat_analysis.values()], [f"{strat.get('Avg MFE on TSL (%)', 0):.2f}%" for strat in strat_analysis.values()]]
        fig.add_trace(go.Table(header=dict(values=header_vals, fill_color='#f8d7da', align='left', font=dict(color='black')), cells=dict(values=cell_values, fill_color='#fcf3f4', align='left')), row=7, col=1)


    fig.update_layout(title_text=f"Crypto Backtest Report (TSL Enabled) | {START_DATE} to {END_DATE}", height=2000, showlegend=False, template="plotly_white")
    fig.write_html(filename)
    print("Report generation complete.")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    start_time = time.time()
    all_data = {symbol: fetch_data(symbol, TIMEFRAME, START_DATE, END_DATE) for symbol in WATCHLIST}
    all_data = {k: v for k, v in all_data.items() if not v.empty}
    
    for symbol in all_data:
        all_data[symbol] = calculate_indicators(all_data[symbol])

    if not all_data:
        print("No data available to backtest. Exiting.")
    else:
        trades, equity, capital_deployment = run_backtest(all_data) 
        if not trades.empty:
            metrics = analyze_performance(trades, equity, capital_deployment) 
            generate_html_report(trades, equity, metrics)
        else:
            print("No trades were executed during the backtest.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")