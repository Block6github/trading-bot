import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Binance Futures Testnet API credentials
API_KEY    = "LHqkWXfY6sCHkUej4v5UnPJqfLfu9KcJlUFVdfz6mE2yigLYlxX43iB1MpsHkaVl"
API_SECRET = "GpOemOKzm76zHqwdpRFrzmeKgf6ZUKdZkZ6YdQpnVoSPtY7P0o2seVUjyOloxgfv"

# Trading configuration
SYMBOL          = "BTCUSDT"
LEVERAGE        = 20
MODEL_PATH      = "btc_asia_SPOT_LONG.joblib"
LOG_FILE        = "live_asia_futures_bot.log"

# Risk management - ALWAYS risk exactly $50 (1%) per trade
TARGET_RISK_PCT = 1.0
MIN_RISK_PCT    = 0.7
MAX_RISK_PCT    = 1.4

# UNLIMITED POSITIONS MODE
MAX_OPEN_POSITIONS = None  # No limit on simultaneous positions

# Margin safety thresholds (only safety mechanism)
MARGIN_WARNING_PCT = 40.0   # Warn at 40% margin usage
MARGIN_CRITICAL_PCT = 70.0  # Emergency stop at 70% margin usage

# Position sizing constraints
MIN_QTY         = 0.001
MIN_NOTIONAL    = 10.0
SLIPPAGE_EST    = 0.003
MIN_STOP_PCT    = 0.5

# Technical settings
MAX_RETRIES     = 3
QTY_PRECISION   = 3
PRICE_PRECISION = 2

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]
)
log = logging.getLogger()

# Initialize Binance Futures client
try:
    client = Client(API_KEY, API_SECRET, testnet=True)
    client.futures_ping()
    log.info("Binance Futures Testnet connected successfully")
except Exception as e:
    log.error(f"Failed to connect to Binance Futures: {e}")
    exit(1)

# Load pre-trained machine learning model
try:
    model = joblib.load(MODEL_PATH)
    log.info(f"ML model loaded from {MODEL_PATH}")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    exit(1)

# Set leverage for the trading pair
try:
    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    log.info(f"Leverage set to {LEVERAGE}x for {SYMBOL}")
except Exception as e:
    log.error(f"Failed to set leverage: {e}")
    exit(1)

# Set margin type to CROSSED
try:
    client.futures_change_margin_type(symbol=SYMBOL, marginType='CROSSED')
    log.info("Margin type set to CROSSED")
except BinanceAPIException as e:
    if e.code == -4046:
        log.info("Already using CROSSED margin")
    else:
        log.warning(f"Margin type change warning: {e}")

def atr(df_, n):
    """Calculate Average True Range - measure of volatility"""
    return (df_['high'] - df_['low']).rolling(n).mean()

def build_feat(rh, rl, candle, momentum, atr20):
    """Build feature set for ML model prediction"""
    orb_range = rh - rl
    range_ratio = orb_range / atr20 if atr20 and atr20 > 0 else 0
    tod = 0
    return pd.DataFrame([[orb_range, candle, momentum, atr20, tod, range_ratio]],
                        columns=['orb_range','candle_range','momentum','atr20','tod','range_ratio'])

def retry_api_call(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Retry mechanism for API calls with exponential backoff"""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (BinanceRequestException, ConnectionError, TimeoutError) as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                log.warning(f"API error (attempt {attempt}/{max_retries}): {e}")
                log.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                log.error(f"API call failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error in API call: {e}")
            raise

def usdt_balance():
    """Get available USDT balance in futures account"""
    try:
        account = retry_api_call(client.futures_account)
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0.0
    except Exception as e:
        log.error(f"Failed to get USDT balance: {e}")
        return 0.0

def get_account_info():
    """Get detailed account information including margin usage"""
    try:
        account = retry_api_call(client.futures_account)
        total_wallet = float(account['totalWalletBalance'])
        total_margin = float(account['totalInitialMargin'])
        available = float(account['availableBalance'])
        unrealized_pnl = float(account['totalUnrealizedProfit'])
        
        margin_usage_pct = (total_margin / total_wallet * 100) if total_wallet > 0 else 0
        
        return {
            'total_wallet': total_wallet,
            'total_margin': total_margin,
            'available': available,
            'unrealized_pnl': unrealized_pnl,
            'margin_usage_pct': margin_usage_pct
        }
    except Exception as e:
        log.error(f"Failed to get account info: {e}")
        return None

def get_open_positions():
    """Get all open positions for the trading pair"""
    try:
        positions = retry_api_call(client.futures_position_information, symbol=SYMBOL)
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        return open_positions
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return []

def get_open_orders():
    """Get all open orders"""
    try:
        orders = retry_api_call(client.futures_get_open_orders, symbol=SYMBOL)
        return orders
    except Exception as e:
        log.error(f"Failed to get open orders: {e}")
        return []

def cancel_all_orders():
    """Cancel all pending orders for the trading pair"""
    try:
        result = retry_api_call(client.futures_cancel_all_open_orders, symbol=SYMBOL)
        log.info("Cancelled all open orders")
        return True
    except Exception as e:
        log.error(f"Failed to cancel orders: {e}")
        return False

def close_all_positions():
    """Close all open positions immediately at market price"""
    try:
        positions = get_open_positions()
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt != 0:
                side = SIDE_SELL if amt > 0 else SIDE_BUY
                qty = abs(amt)
                retry_api_call(
                    client.futures_create_order,
                    symbol=SYMBOL,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=qty
                )
                log.info(f"Closed position: {qty} {side}")
        return True
    except Exception as e:
        log.error(f"Failed to close positions: {e}")
        return False

def fetch_1m_since(start_ts):
    """Fetch 1-minute candlestick data from a given timestamp"""
    klines = retry_api_call(client.futures_klines, symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, startTime=start_ts)
    df = pd.DataFrame(klines, columns=['ts','open','high','low','close','vol','close_time',
                                       'quote_vol','trades','taker_base','taker_quote','ignore'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col])
    return df

def round_quantity(qty):
    """Round quantity to exchange precision"""
    return round(qty, QTY_PRECISION)

def round_price(price):
    """Round price to exchange precision"""
    return round(price, PRICE_PRECISION)

def validate_min_notional(price, quantity, min_notional=MIN_NOTIONAL):
    """Validate that position value meets Binance minimum notional requirement"""
    notional = price * quantity
    required = min_notional * 1.01
    
    if notional < required:
        min_qty = required / price
        log.warning(f"MIN_NOTIONAL check failed:")
        log.warning(f"  Current: ${notional:.2f} < Required: ${required:.2f}")
        log.warning(f"  Need minimum: {min_qty:.6f} BTC @ ${price:.2f}")
        return False, round_quantity(min_qty), notional
    
    log.info(f"MIN_NOTIONAL passed: ${notional:.2f} >= ${required:.2f}")
    return True, quantity, notional

def calculate_position_with_leverage(balance, entry_price, sl_price, num_open_positions=0, leverage=LEVERAGE):
    """
    Calculate position size - ALWAYS risk exactly 1% ($50)
    
    No matter how many positions are open, each new trade risks exactly $50
    """
    
    # ALWAYS use 1% risk - no scaling based on open positions
    target_risk_amount = balance * TARGET_RISK_PCT / 100.0
    min_risk_amount = balance * MIN_RISK_PCT / 100.0
    max_risk_amount = balance * MAX_RISK_PCT / 100.0
    
    risk_distance = entry_price - sl_price
    
    if risk_distance <= 0:
        log.error(f"Invalid stop loss: Entry ${entry_price:.2f} <= SL ${sl_price:.2f}")
        return None, None, None
    
    stop_distance_pct = (risk_distance / entry_price) * 100
    
    if stop_distance_pct < MIN_STOP_PCT:
        log.error(f"Stop loss too tight: {stop_distance_pct:.2f}% < {MIN_STOP_PCT}%")
        log.error(f"  Minimum stop distance: ${entry_price * MIN_STOP_PCT / 100:.2f}")
        return None, None, None
    
    # Calculate quantity to risk exactly $50
    qty = target_risk_amount / risk_distance
    qty = round_quantity(qty)
    
    if qty < MIN_QTY:
        log.warning(f"Calculated qty {qty:.6f} < MIN_QTY {MIN_QTY}")
        qty = MIN_QTY
    
    position_value = qty * entry_price
    required_margin = position_value / leverage
    
    # Safety check: don't use more than 80% of balance as margin
    max_margin = balance * 0.80
    
    if required_margin > max_margin:
        log.warning(f"Required margin ${required_margin:.2f} > 80% of balance")
        adjusted_position_value = max_margin * leverage
        qty = adjusted_position_value / entry_price
        qty = round_quantity(qty)
        
        position_value = qty * entry_price
        required_margin = position_value / leverage
        log.warning(f"  Adjusted qty to: {qty:.3f} BTC")
    
    actual_loss_if_sl_hit = qty * risk_distance
    actual_risk_pct = (actual_loss_if_sl_hit / balance) * 100
    
    log.info("")
    log.info("=" * 70)
    log.info(f"POSITION SIZING - RISK ${actual_loss_if_sl_hit:.2f} IF STOP HITS")
    log.info("=" * 70)
    log.info(f"Account Balance:     ${balance:,.2f} USDT")
    log.info(f"Open Positions:      {num_open_positions}")
    log.info(f"Target Risk:         ${target_risk_amount:.2f} ({TARGET_RISK_PCT:.1f}%) - ALWAYS FIXED")
    log.info("")
    log.info(f"Entry Price:         ${entry_price:,.2f}")
    log.info(f"Stop Loss:           ${sl_price:,.2f}")
    log.info(f"Risk per BTC:        ${risk_distance:,.2f} ({stop_distance_pct:.2f}%)")
    log.info("")
    log.info(f"Position Size:       {qty:.3f} BTC")
    log.info(f"Position Value:      ${position_value:,.2f}")
    log.info(f"Leverage:            {leverage}x")
    log.info(f"Margin Required:     ${required_margin:,.2f} ({(required_margin/balance)*100:.1f}% of balance)")
    log.info("")
    log.info("IF STOP LOSS HITS:")
    log.info(f"  You Lose:          ${actual_loss_if_sl_hit:.2f} ({actual_risk_pct:.2f}%)")
    log.info(f"  Remaining Balance: ${balance - actual_loss_if_sl_hit:,.2f}")
    log.info("=" * 70)
    log.info("")
    
    return qty, risk_distance, actual_risk_pct

def place_futures_market_long(quantity):
    """Place a market buy order to open a long position"""
    try:
        log.info(f"Sending FUTURES LONG market order: {quantity:.3f} BTC")
        
        order = retry_api_call(
            client.futures_create_order,
            symbol=SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        
        filled_qty = float(order.get('executedQty', 0))
        avg_price = float(order.get('avgPrice', 0))
        
        if filled_qty == 0:
            log.error("Order placed but ZERO filled")
            return None, None, None
        
        log.info("LONG ORDER FILLED")
        log.info(f"  Order ID: {order.get('orderId', 'N/A')}")
        log.info(f"  Filled: {filled_qty:.3f} BTC")
        log.info(f"  Avg Price: ${avg_price:,.2f}")
        
        return order, filled_qty, avg_price
        
    except BinanceAPIException as e:
        log.error(f"Futures order failed: {e}")
        return None, None, None

def place_stop_loss_order(quantity, stop_price):
    """Place a stop market order to protect position"""
    try:
        stop_price = round_price(stop_price)
        log.info(f"Placing STOP LOSS at ${stop_price:,.2f}")
        
        order = retry_api_call(
            client.futures_create_order,
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            quantity=quantity,
            stopPrice=stop_price,
            reduceOnly='true'
        )
        
        log.info("STOP LOSS PLACED")
        log.info(f"  Order ID: {order.get('orderId', 'N/A')}")
        log.info(f"  Stop Price: ${stop_price:,.2f}")
        
        return order
        
    except BinanceAPIException as e:
        log.error(f"Stop loss order failed: {e}")
        return None

def place_take_profit_order(quantity, tp_price):
    """Place a take profit market order"""
    try:
        tp_price = round_price(tp_price)
        log.info(f"Placing TAKE PROFIT at ${tp_price:,.2f}")
        
        order = retry_api_call(
            client.futures_create_order,
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            quantity=quantity,
            stopPrice=tp_price,
            reduceOnly='true'
        )
        
        log.info("TAKE PROFIT PLACED")
        log.info(f"  Order ID: {order.get('orderId', 'N/A')}")
        log.info(f"  TP Price: ${tp_price:,.2f}")
        
        return order
        
    except BinanceAPIException as e:
        log.error(f"Take profit order failed: {e}")
        return None

def execute_futures_trade(entry_price, sl, tp, balance, pred_rr, num_open_positions=0):
    """Execute trade - ALWAYS risk exactly $50 (1%)"""
    log.info("")
    log.info("=" * 80)
    log.info("FUTURES TRADE EXECUTION STARTED")
    log.info("=" * 80)
    
    log.info("STEP 1: Calculate position with leverage")
    log.info(f"  Balance: ${balance:,.2f} USDT")
    log.info(f"  Leverage: {LEVERAGE}x")
    log.info(f"  Open Positions: {num_open_positions}")
    log.info(f"  Entry (expected): ${entry_price:,.2f}")
    log.info(f"  Stop Loss: ${sl:,.2f}")
    log.info(f"  Take Profit: ${tp:,.2f}")
    log.info(f"  Model Predicted R:R: {pred_rr:.2f}")
    
    qty, worst_risk, target_risk_pct = calculate_position_with_leverage(
        balance, entry_price, sl, num_open_positions
    )
    
    if qty is None:
        log.error("Position calculation failed - aborting trade")
        return False
    
    log.info("STEP 2: Validate MIN_NOTIONAL")
    
    valid, adjusted_qty, notional = validate_min_notional(entry_price, qty)
    if not valid:
        log.error("MIN_NOTIONAL failed - adjusting quantity")
        qty = adjusted_qty
        valid, adjusted_qty, notional = validate_min_notional(entry_price, qty)
        if not valid:
            log.error("Still below MIN_NOTIONAL - aborting")
            return False
    
    log.info("STEP 3: Execute FUTURES LONG market order")
    
    buy_order, filled_qty, avg_fill_price = place_futures_market_long(qty)
    
    if buy_order is None or filled_qty is None or avg_fill_price is None:
        log.error("Long order failed - aborting trade")
        return False
    
    log.info("LONG POSITION OPENED")
    
    time.sleep(2)
    
    log.info("STEP 4: Place STOP LOSS order")
    
    sl_order = place_stop_loss_order(filled_qty, sl)
    
    if sl_order is None:
        log.error("")
        log.error("=" * 80)
        log.error("CRITICAL: STOP LOSS FAILED")
        log.error("=" * 80)
        log.error("YOU HAVE AN UNPROTECTED POSITION - CLOSING IMMEDIATELY")
        log.error("=" * 80)
        
        close_all_positions()
        return False
    
    log.info("STEP 5: Place TAKE PROFIT order")
    
    tp_order = place_take_profit_order(filled_qty, tp)
    
    if tp_order is None:
        log.warning("Take profit order failed - position still has stop loss protection")
    
    actual_rr = (tp - avg_fill_price) / (avg_fill_price - sl) if (avg_fill_price - sl) > 0 else 0
    actual_risk = filled_qty * (avg_fill_price - sl)
    
    log.info("")
    log.info("=" * 80)
    log.info("FUTURES TRADE EXECUTION COMPLETE")
    log.info("=" * 80)
    log.info("TRADE SUMMARY:")
    log.info(f"  Position: {filled_qty:.3f} BTC LONG ({LEVERAGE}x leverage)")
    log.info(f"  Entry: ${avg_fill_price:,.2f}")
    log.info(f"  Stop Loss: ${sl:,.2f}")
    log.info(f"  Take Profit: ${tp:,.2f}")
    log.info(f"  Risk: ${actual_risk:.2f} (target: $50)")
    log.info(f"  Model Predicted R:R: {pred_rr:.2f}")
    log.info(f"  Actual R:R: {actual_rr:.2f}")
    log.info("=" * 80)
    log.info("")
    
    return True

def display_position_summary():
    """Display detailed summary of all open positions"""
    positions = get_open_positions()
    account_info = get_account_info()
    
    if not positions:
        log.info("No open positions")
        return
    
    log.info("")
    log.info("=" * 80)
    log.info("OPEN POSITIONS SUMMARY")
    log.info("=" * 80)
    
    if account_info:
        log.info(f"Total Wallet Balance: ${account_info['total_wallet']:,.2f}")
        log.info(f"Total Margin Used:    ${account_info['total_margin']:,.2f}")
        log.info(f"Margin Usage:         {account_info['margin_usage_pct']:.2f}%")
        log.info(f"Available Balance:    ${account_info['available']:,.2f}")
        log.info(f"Unrealized P&L:       ${account_info['unrealized_pnl']:+,.2f}")
        log.info("")
    
    total_position_value = 0
    total_unrealized_pnl = 0
    
    for idx, pos in enumerate(positions, 1):
        amt = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark_price = float(pos['markPrice'])
        unrealized = float(pos['unRealizedProfit'])
        position_value = abs(amt) * entry
        margin = position_value / LEVERAGE
        
        total_position_value += position_value
        total_unrealized_pnl += unrealized
        
        log.info(f"Position {idx}:")
        log.info(f"  Size: {abs(amt):.3f} BTC")
        log.info(f"  Entry: ${entry:,.2f}")
        log.info(f"  Mark Price: ${mark_price:,.2f}")
        log.info(f"  Position Value: ${position_value:,.2f}")
        log.info(f"  Margin: ${margin:,.2f}")
        log.info(f"  Unrealized P&L: ${unrealized:+,.2f}")
        log.info("")
    
    log.info(f"Total Positions: {len(positions)}")
    log.info(f"Total Position Value: ${total_position_value:,.2f}")
    log.info(f"Total Unrealized P&L: ${total_unrealized_pnl:+,.2f}")
    log.info(f"Total Risk Exposure: ${len(positions) * 50:.0f} ({len(positions)}%)")
    
    if account_info:
        if account_info['margin_usage_pct'] >= MARGIN_CRITICAL_PCT:
            log.error(f"🚨 CRITICAL: Margin usage {account_info['margin_usage_pct']:.1f}% >= {MARGIN_CRITICAL_PCT}%!")
            log.error("New trades will be BLOCKED until margin drops")
        elif account_info['margin_usage_pct'] >= MARGIN_WARNING_PCT:
            log.warning(f"⚠️ WARNING: Margin usage {account_info['margin_usage_pct']:.1f}% >= {MARGIN_WARNING_PCT}%")
    
    log.info("=" * 80)
    log.info("")

# Global variables for ORB tracking
RH = RL = ORB_LAST_CLOSE = None
POST_ORB_DATA = None

def run():
    """
    Main trading loop - UNLIMITED POSITIONS MODE
    
    Rules:
    1. Take ONE trade per day maximum
    2. No limit on total positions (can have 7, 10, 20+ positions)
    3. Each trade always risks exactly $50 (1%)
    4. Emergency stop only at 70% margin usage
    """
    log.info("")
    log.info("=" * 80)
    log.info(f"UNLIMITED POSITIONS FUTURES BOT - {LEVERAGE}x LEVERAGE")
    log.info("=" * 80)
    log.info(f"Symbol: {SYMBOL}")
    log.info(f"Leverage: {LEVERAGE}x")
    log.info(f"Risk Per Trade: {TARGET_RISK_PCT}% FIXED ($50)")
    log.info(f"Max Positions: UNLIMITED")
    log.info("")
    log.info("TRADING RULES:")
    log.info("  ✓ ONE trade per day (strictly enforced)")
    log.info("  ✓ UNLIMITED simultaneous positions allowed")
    log.info("  ✓ Each position risks exactly $50 (1%)")
    log.info("  ✓ Can trade daily even with 7+ positions open")
    log.info("  ✓ Safety: Emergency stop at 70% margin usage")
    log.info("")
    log.info("RISK EXAMPLES:")
    log.info("  • 1 position:  $50 risk (1%)")
    log.info("  • 7 positions: $350 risk (7%)")
    log.info("  • 10 positions: $500 risk (10%)")
    log.info("  • 20 positions: $1,000 risk (20%)")
    log.info("=" * 80)
    log.info("")
    
    # Track which dates have had trades (not positions!)
    trades_by_date = {}  # Format: {'2024-12-09': True, '2024-12-10': True}
    
    last_detailed_log = None
    last_reset_date = None
    scan_count = 0
    global RH, RL, ORB_LAST_CLOSE, POST_ORB_DATA

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date()
            today_str = today.strftime('%Y-%m-%d')

            # Midnight reset - clear ORB but KEEP trade history
            if now.hour == 0 and now.minute == 0 and last_reset_date != today:
                RH = RL = ORB_LAST_CLOSE = POST_ORB_DATA = None
                scan_count = 0
                log.info("")
                log.info("=" * 80)
                log.info("MIDNIGHT RESET")
                log.info("=" * 80)
                log.info(f"New trading day: {today}")
                log.info(f"Total days with trades: {len(trades_by_date)}")
                display_position_summary()
                log.info("=" * 80)
                log.info("")
                last_reset_date = today
                time.sleep(60)
                continue

            # 2-hour status update
            if last_detailed_log is None or (now - last_detailed_log).total_seconds() >= 7200:
                display_position_summary()
                log.info(f"Traded Today: {'Yes' if today_str in trades_by_date else 'No'}")
                log.info(f"Total Days Traded: {len(trades_by_date)}")
                if RH:
                    log.info(f"ORB: RH=${RH:,.2f} RL=${RL:,.2f}")
                log.info(f"Scans since last update: {scan_count}")
                log.info("")
                last_detailed_log = now
                scan_count = 0

            # Heartbeat every 30 minutes
            if now.minute % 30 == 0 and now.second < 32:
                open_positions = get_open_positions()
                log.info(f"Heartbeat: {now.strftime('%H:%M:%S UTC')} | Positions: {len(open_positions)} | Traded today: {'Yes' if today_str in trades_by_date else 'No'}")

            # Check if we already traded TODAY - if yes, skip scanning
            if today_str in trades_by_date:
                time.sleep(60)
                continue

            # Build ORB at 04:00 UTC
            if now.hour == 4 and now.minute == 0 and today_str not in trades_by_date and RH is None:
                log.info("")
                log.info("=" * 80)
                log.info("04:00 UTC - BUILDING ORB")
                log.info("=" * 80)
                
                try:
                    start_ts = int((now - timedelta(hours=4, minutes=1)).timestamp() * 1000)
                    df = fetch_1m_since(start_ts)
                    orb = df[df['ts'].dt.hour < 4]
                    
                    if len(orb) < 30:
                        log.warning("Insufficient ORB data - skipping today")
                        trades_by_date[today_str] = True
                        time.sleep(60)
                        continue
                    
                    RH, RL = orb['high'].max(), orb['low'].min()
                    ORB_LAST_CLOSE = orb.iloc[-1]['close']
                    POST_ORB_DATA = df[df['ts'].dt.hour >= 4].copy()
                    
                    log.info("ORB BUILT SUCCESSFULLY")
                    log.info(f"  RH: ${RH:,.2f}")
                    log.info(f"  RL: ${RL:,.2f}")
                    log.info(f"  Range: ${RH-RL:,.2f}")
                    log.info("  Ready to scan for breakout signals...")
                    log.info("=" * 80)
                    log.info("")
                    
                except Exception as e:
                    log.error(f"ORB build failed: {e}", exc_info=True)
                    trades_by_date[today_str] = True
                
                time.sleep(60)
                continue

            # Signal scanning - only if ORB exists and haven't traded TODAY
            if RH is not None and today_str not in trades_by_date:
                scan_count += 1
                
                try:
                    # Get current positions and account info
                    open_positions = get_open_positions()
                    num_open_positions = len(open_positions)
                    account_info = get_account_info()
                    
                    # ONLY SAFETY CHECK: Critical margin usage (70%)
                    if account_info and account_info['margin_usage_pct'] >= MARGIN_CRITICAL_PCT:
                        log.error("")
                        log.error("=" * 80)
                        log.error(f"🚨 CRITICAL MARGIN USAGE: {account_info['margin_usage_pct']:.1f}%")
                        log.error("=" * 80)
                        log.error("Margin usage >= 70% - EMERGENCY STOP")
                        log.error("NO NEW TRADES until margin drops below 70%")
                        log.error("Consider closing some positions manually")
                        log.error("=" * 80)
                        log.error("")
                        trades_by_date[today_str] = True
                        time.sleep(60)
                        continue
                    
                    start_ts = int((now - timedelta(hours=5)).timestamp() * 1000)
                    df_live = fetch_1m_since(start_ts)
                    post = df_live[df_live['ts'].dt.hour >= 4].copy()
                    
                    if len(post) < 5:
                        time.sleep(30)
                        continue
                    
                    curr_price = post.iloc[-1]['close']
                    candle_range = post.iloc[-1]['high'] - post.iloc[-1]['low']
                    momentum = curr_price - ORB_LAST_CLOSE
                    atr20_val = atr(post, 20).iloc[-1] if len(post) >= 20 else (RH - RL)
                    
                    X = build_feat(RH, RL, candle_range, momentum, atr20_val)
                    pred_rr = model.predict(X)[0]
                    
                    # Signal detection: Price breaks ORB high AND model predicts R:R >= 2.0
                    if curr_price > RH and pred_rr >= 2.0:
                        log.info("")
                        log.info("=" * 80)
                        log.info("🎯 SIGNAL DETECTED")
                        log.info("=" * 80)
                        log.info(f"Current Price: ${curr_price:,.2f}")
                        log.info(f"ORB High: ${RH:,.2f}")
                        log.info(f"Breakout: +${curr_price - RH:,.2f}")
                        log.info(f"Model Predicted R:R: {pred_rr:.2f}")
                        log.info(f"Today's Date: {today_str}")
                        log.info(f"Traded Today: No")
                        log.info(f"Current Open Positions: {num_open_positions} (UNLIMITED MODE)")
                        
                        if account_info:
                            log.info(f"Current Margin Usage: {account_info['margin_usage_pct']:.2f}%")
                            if account_info['margin_usage_pct'] >= MARGIN_WARNING_PCT:
                                log.warning(f"⚠️ Margin usage above {MARGIN_WARNING_PCT}% - proceeding with caution")
                        
                        if open_positions:
                            log.info("")
                            log.info("Existing positions (from previous days):")
                            for idx, p in enumerate(open_positions, 1):
                                amt = float(p['positionAmt'])
                                entry = float(p['entryPrice'])
                                unrealized_pnl = float(p['unRealizedProfit'])
                                log.info(f"  {idx}. {abs(amt):.3f} BTC @ ${entry:,.2f} | P&L: ${unrealized_pnl:+,.2f}")
                        
                        log.info("=" * 80)
                        log.info("")
                        
                        balance = usdt_balance()
                        
                        if balance < 100:
                            log.error(f"Insufficient balance: ${balance:.2f}")
                            trades_by_date[today_str] = True
                            time.sleep(60)
                            continue
                        
                        entry = curr_price
                        sl = RL
                        
                        # TP CALCULATION: Uses ONLY model's predicted R:R
                        # Formula: TP = Entry + (Model_RR × Risk_Distance)
                        # THIS IS THE KEY PART - TP IS 100% MODEL-DRIVEN
                        tp = entry + pred_rr * (entry - sl)
                        
                        log.info("=" * 80)
                        log.info("TP LOCKED TO MODEL PREDICTION")
                        log.info("=" * 80)
                        log.info(f"Entry: ${entry:,.2f}")
                        log.info(f"Stop Loss: ${sl:,.2f}")
                        log.info(f"Risk Distance: ${entry - sl:,.2f}")
                        log.info(f"Model R:R: {pred_rr:.2f}")
                        log.info(f"TP = ${entry:,.2f} + ({pred_rr:.2f} × ${entry - sl:,.2f})")
                        log.info(f"TP = ${tp:,.2f}")
                        log.info("This TP will NOT be adjusted regardless of fill price")
                        log.info("=" * 80)
                        log.info("")
                        
                        stop_dist_pct = ((entry - sl) / entry) * 100
                        if stop_dist_pct < MIN_STOP_PCT:
                            log.warning(f"Stop too tight: {stop_dist_pct:.2f}% < {MIN_STOP_PCT}%")
                            trades_by_date[today_str] = True
                            time.sleep(60)
                            continue
                        
                        # Execute the trade
                        success = execute_futures_trade(entry, sl, tp, balance, pred_rr, num_open_positions)
                        
                        if success:
                            # Mark today as traded - prevents multiple trades same day
                            trades_by_date[today_str] = True
                            log.info("")
                            log.info("=" * 80)
                            log.info("✅ TRADE COMPLETE - DONE FOR TODAY")
                            log.info("=" * 80)
                            log.info(f"Trade Date: {today_str}")
                            log.info(f"Total Days Traded: {len(trades_by_date)}")
                            log.info("Will resume scanning tomorrow at 04:00 UTC")
                            log.info("")
                            
                            # Show updated position summary
                            time.sleep(3)
                            display_position_summary()
                            
                            log.info("=" * 80)
                            log.info("")
                        else:
                            log.error("Trade execution failed - marking day as traded to prevent retries")
                            trades_by_date[today_str] = True
                        
                        time.sleep(300)
                        continue
                
                except Exception as e:
                    log.error(f"Signal check error: {e}", exc_info=True)
            
            time.sleep(30)
        
        except KeyboardInterrupt:
            log.info("")
            log.info("=" * 80)
            log.info("🛑 BOT STOPPED BY USER")
            log.info("=" * 80)
            display_position_summary()
            log.info("Please manage open positions manually if needed")
            log.info("=" * 80)
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    run()