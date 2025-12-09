#!/usr/bin/env python3
"""
futures_dashboard.py
Minimalistic portfolio dashboard for futures trading - monitoring performance and logs only.
"""

from flask import Flask, render_template_string
from binance.client import Client
import os, json, datetime

API_KEY    = "LHqkWXfY6sCHkUej4v5UnPJqfLfu9KcJlUFVdfz6mE2yigLYlxX43iB1MpsHkaVl"
API_SECRET = "GpOemOKzm76zHqwdpRFrzmeKgf6ZUKdZkZ6YdQpnVoSPtY7P0o2seVUjyOloxgfv"
SYMBOL     = "BTCUSDT"
LOG_FILE   = "live_asia_futures_bot.log"

client = Client(API_KEY, API_SECRET, testnet=True)
app = Flask(__name__)

def calculate_futures_pnl(positions):
    """Calculate total unrealized PNL from open futures positions"""
    total_pnl = 0.0
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            total_pnl += float(pos['unRealizedProfit'])
    return total_pnl

PRIMARY_COLOR = "#00BFA5"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Futures Trading Dashboard</title>
  <meta http-equiv="refresh" content="30">
  
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    :root {
      --primary-color: #00BFA5;
    }

    body {
      font-family: 'Inter', sans-serif;
    }
    
    .card-minimal {
      border: 1px solid rgb(229 231 235);
      transition: all 0.2s ease;
    }
    .card-minimal:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transform: translateY(-2px);
    }
    
    .metric-value {
      font-size: 2.25rem;
      font-weight: 700;
      color: var(--primary-color);
      letter-spacing: -0.025em;
    }

    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.5rem;
        }
    }
  </style>
</head>
<body class="bg-gray-50 text-gray-800 antialiased">
  
  <div class="container mx-auto p-4 md:p-8 max-w-7xl">
    
    <header class="mb-8 p-6 bg-white rounded-xl shadow-lg border-b-4 border-teal-400">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center">
        <h1 class="text-3xl font-extrabold text-gray-900 flex items-center mb-2 md:mb-0">
          <i class="fas fa-chart-line mr-3 text-teal-500"></i> Futures Trading Dashboard
        </h1>
        <p class="text-sm text-gray-500">Last Updated: {{ server_time }} (UTC)</p>
      </div>
    </header>

    <div class="grid grid-cols-2 gap-4 mb-8">
      
      <div class="card-minimal bg-white p-4 rounded-xl shadow-md flex flex-col justify-between">
        <h6 class="text-sm font-semibold text-gray-500 mb-2">Unrealized P&L</h6>
        <div class="metric-value {% if total_pnl >= 0 %}text-teal-500{% else %}text-red-500{% endif %}">
          ${{ total_pnl }}
        </div>
      </div>

      <div class="card-minimal bg-white p-4 rounded-xl shadow-md flex flex-col justify-between">
        <h6 class="text-sm font-semibold text-gray-500 mb-2">Open Positions</h6>
        <div class="metric-value">{{ open_positions }}</div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      
      <div class="lg:col-span-1 space-y-4">
        <h3 class="text-xl font-bold text-gray-700 border-b pb-2 mb-4">Account Overview</h3>
        
        <div class="card-minimal bg-white p-5 rounded-xl shadow-md">
          <h5 class="text-lg font-semibold text-gray-900 mb-1">Current BTC Price</h5>
          <div class="text-3xl font-bold text-gray-600">${{ btc_price }}</div>
          <p class="text-xs text-gray-400 mt-2">Source: Binance Futures Testnet</p>
        </div>

        <div class="card-minimal bg-white p-5 rounded-xl shadow-md">
          <h5 class="text-lg font-semibold text-gray-900 mb-1">Available Balance</h5>
          <div class="text-3xl font-bold text-gray-600">${{ usdt_bal }}</div>
          <p class="text-xs text-gray-400 mt-2">USDT Available</p>
        </div>
        
        <div class="card-minimal bg-white p-5 rounded-xl shadow-md">
          <h5 class="text-lg font-semibold text-gray-900 mb-1">Margin Balance</h5>
          <div class="text-3xl font-bold text-gray-600">${{ margin_balance }}</div>
          <p class="text-xs text-gray-400 mt-2">Total Margin</p>
        </div>

        <div class="card-minimal bg-white p-5 rounded-xl shadow-md">
          <h5 class="text-lg font-semibold text-gray-900 mb-1">Leverage</h5>
          <div class="text-3xl font-bold text-teal-600">50x</div>
        </div>

      </div>
      
      <div class="lg:col-span-1 space-y-8">

        <div class="mb-8">
          <h3 class="text-xl font-bold text-gray-700 border-b pb-2 mb-4"><i class="fas fa-chart-bar mr-2 text-teal-500"></i> Position Details</h3>
          
          {% if positions_data %}
            {% for pos in positions_data %}
            <div class="card-minimal bg-white p-6 rounded-xl shadow-md border-l-4 {% if pos.side == 'LONG' %}border-teal-500{% else %}border-red-500{% endif %} mb-4">
              
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                
                <div class="p-2 bg-gray-50 rounded">
                  <strong class="text-gray-500 block">Side</strong> 
                  <span class="font-semibold text-lg {% if pos.side == 'LONG' %}text-teal-600{% else %}text-red-600{% endif %}">{{ pos.side }}</span>
                </div>
                
                <div class="p-2 bg-gray-50 rounded">
                  <strong class="text-gray-500 block">Size</strong> 
                  <span class="font-semibold text-lg text-gray-800">{{ pos.size }} BTC</span>
                </div>
                
                <div class="p-2 bg-gray-50 rounded">
                  <strong class="text-gray-500 block">Entry Price</strong> 
                  <span class="font-semibold text-lg text-gray-800">${{ pos.entry }}</span>
                </div>
                
                <div class="p-2 bg-gray-50 rounded">
                  <strong class="text-gray-500 block">Mark Price</strong> 
                  <span class="font-semibold text-lg text-gray-800">${{ pos.mark }}</span>
                </div>
              </div>
              
              <div class="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-gray-100">
                <div class="text-center">
                  <strong class="text-gray-500 block">Unrealized P&L</strong> 
                  <span class="text-2xl font-bold {% if pos.pnl >= 0 %}text-teal-600{% else %}text-red-600{% endif %}">${{ pos.pnl }}</span>
                </div>
                <div class="text-center">
                  <strong class="text-gray-500 block">ROE</strong> 
                  <span class="text-2xl font-bold {% if pos.roe >= 0 %}text-teal-600{% else %}text-red-600{% endif %}">{{ pos.roe }}%</span>
                </div>
              </div>
              
            </div>
            {% endfor %}
          {% else %}
            <div class="card-minimal bg-white p-6 rounded-xl shadow-md border-l-4 border-gray-300">
              <p class="text-gray-500 text-center">No open positions</p>
            </div>
          {% endif %}
        </div>

        <div class="mb-8">
          <h3 class="text-xl font-bold text-gray-700 border-b pb-2 mb-4"><i class="fas fa-terminal mr-2 text-teal-500"></i> Live Logs (Last 5)</h3>
          <div class="card-minimal bg-white p-4 rounded-xl shadow-md border-l-4 border-gray-300">
            <div class="space-y-1">
              {% for line in log_tail %}
                <pre class="log-row text-sm text-gray-600 bg-gray-50 p-1 rounded overflow-x-auto">{{ line | trim }}</pre>
              {% endfor %}
            </div>
          </div>
        </div>

      </div>

    </div>

    <footer class="text-center mt-10 p-4 border-t border-gray-200 text-gray-500 text-sm">
      <p>Dashboard auto-refreshes every 30 seconds. Built with Python, Flask, and Binance Futures API.</p>
    </footer>
    
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    btc_price, usdt_bal, margin_balance, total_pnl = 0.0, 0.0, 0.0, 0.0
    
    try:
        account = client.futures_account()
        
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                usdt_bal = float(asset['availableBalance'])
                margin_balance = float(asset['marginBalance'])
                break
        
        ticker = client.futures_symbol_ticker(symbol=SYMBOL)
        btc_price = float(ticker['price'])
        
        positions = client.futures_position_information(symbol=SYMBOL)
        total_pnl = calculate_futures_pnl(positions)
        
        positions_data = []
        open_positions = 0
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt != 0:
                open_positions += 1
                side = 'LONG' if amt > 0 else 'SHORT'
                positions_data.append({
                    'side': side,
                    'size': abs(round(amt, 3)),
                    'entry': round(float(pos['entryPrice']), 2),
                    'mark': round(float(pos['markPrice']), 2),
                    'pnl': round(float(pos['unRealizedProfit']), 2),
                    'roe': round(float(pos['unRealizedProfit']) / float(pos['initialMargin']) * 100, 2) if float(pos['initialMargin']) > 0 else 0
                })
        
    except Exception as e:
        print(f"Error fetching futures data: {e}")
        positions_data = []
        open_positions = 0

    log_tail = []
    try:
        with open(LOG_FILE, "r") as f:
            log_tail = [line.strip() for line in f.readlines()[-5:]]
    except FileNotFoundError:
        log_tail = ["Log file not found: Please ensure 'live_asia_futures_bot.log' exists."]
    except Exception as e:
        log_tail = [f"Error reading log file: {e}"]

    now_utc = datetime.datetime.utcnow()

    return render_template_string(HTML,
                                  server_time=now_utc.strftime('%Y-%m-%d %H:%M:%S'),
                                  btc_price=round(btc_price, 2),
                                  usdt_bal=round(usdt_bal, 2),
                                  margin_balance=round(margin_balance, 2),
                                  total_pnl=round(total_pnl, 2),
                                  open_positions=open_positions,
                                  positions_data=positions_data,
                                  log_tail=log_tail)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)