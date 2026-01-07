import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import requests
import os
# 修改 1: 引入 timezone
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. 配置設定
# ==========================================
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1458076152504520771/HZoL7eh3KpncZW7zbZEqBMqM0SHJDczVXmCBJHv9QuXV8qfLlEUPCCDs-Z4LnLcaGQ_B")

exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# ==========================================
# 參數設定
# PIVOT_LEN = 10 (Swing 結構，接近 LuxAlgo 的大結構)
# ==========================================
PIVOT_LEN = 10  

# ==========================================
# 2. SMC 策略核心邏輯
# ==========================================
def process_smc_data(df):
    if len(df) < 100: return None, None, None, None, None, None

    # 1. 識別 Pivot Points (Swings)
    df['high_max'] = df['high'].rolling(window=PIVOT_LEN*2+1, center=True).max()
    df['low_min'] = df['low'].rolling(window=PIVOT_LEN*2+1, center=True).min()

    df['is_pivot_high'] = (df['high'] == df['high_max'])
    df['is_pivot_low'] = (df['low'] == df['low_min'])

    obs = [] 
    
    last_pivot_high_candle = None
    last_pivot_low_candle = None

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    is_ph = df['is_pivot_high'].values
    is_pl = df['is_pivot_low'].values
    
    last_swing_high = highs[0]
    last_swing_low = lows[0]

    start_idx = PIVOT_LEN * 2 + 1
    current_trend = 0 
    
    final_side = None
    final_entry = 0
    final_sl = 0
    final_tp1 = 0
    final_tp2 = 0
    
    for i in range(start_idx, len(df)):
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        
        # --- 1. 更新結構與記錄候選 OB ---
        pivot_idx = i - PIVOT_LEN
        if pivot_idx >= 0:
            if is_ph[pivot_idx]:
                last_swing_high = highs[pivot_idx]
                last_pivot_high_candle = {
                    'type': 'bear',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx], 
                    'mitigated': False,
                    'idx': pivot_idx
                }
            
            if is_pl[pivot_idx]:
                last_swing_low = lows[pivot_idx]
                last_pivot_low_candle = {
                    'type': 'bull',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx],
                    'mitigated': False,
                    'idx': pivot_idx
                }

        # --- 2. 判斷結構破壞 (BOS) 並生成 OB ---
        
        # Bullish BOS (向上突破)
        if curr_close > last_swing_high:
            if current_trend != 1:
                if last_pivot_low_candle is not None:
                    if not any(ob['idx'] == last_pivot_low_candle['idx'] for ob in obs):
                        obs.append(last_pivot_low_candle.copy())
            current_trend = 1
            
        # Bearish BOS (向下跌破)
        elif curr_close < last_swing_low:
            if current_trend != -1:
                if last_pivot_high_candle is not None:
                    if not any(ob['idx'] == last_pivot_high_candle['idx'] for ob in obs):
                        obs.append(last_pivot_high_candle.copy())
            current_trend = -1
            
        # --- 3. 判斷進場 (回踩 OB - 掛單邏輯) ---
        
        # [做多]
        if current_trend == 1: 
            valid_obs = [ob for ob in obs if ob['type'] == 'bull' and not ob['mitigated'] and ob['top'] < curr_close]
            
            if valid_obs:
                target_ob = valid_obs[-1]
                if curr_low <= target_ob['top']:
                    if i == len(df) - 1:
                        final_side = "LONG"
                        final_entry = target_ob['top']
                        final_sl = target_ob['bottom']
                        final_tp1 = last_swing_high
                        risk = final_entry - final_sl
                        final_tp2 = final_entry + risk if risk > 0 else final_entry * 1.01
                    target_ob['mitigated'] = True
        
        # [做空]
        elif current_trend == -1:
            valid_obs = [ob for ob in obs if ob['type'] == 'bear' and not ob['mitigated'] and ob['bottom'] > curr_close]
            
            if valid_obs:
                target_ob = valid_obs[-1]
                if curr_high >= target_ob['bottom']:
                    if i == len(df) - 1:
                        final_side = "SHORT"
                        final_entry = target_ob['bottom']
                        final_sl = target_ob['top']
                        final_tp1 = last_swing_low
                        risk = final_sl - final_entry
                        final_tp2 = final_entry - risk if risk > 0 else final_entry * 0.99
                    target_ob['mitigated'] = True

        # --- 4. 清理無效 OB ---
        for ob in obs:
            if not ob['mitigated']:
                if ob['type'] == 'bull' and curr_close < ob['bottom']:
                    ob['mitigated'] = True 
                elif ob['type'] == 'bear' and curr_close > ob['top']:
                    ob['mitigated'] = True

    return final_side, final_entry, final_sl, final_tp1, final_tp2, df

# ==========================================
# 3. 機器人主程式
# ==========================================
class TradingBot:
    def __init__(self):
        self.last_signals = {} 
        self.symbols = []
        self.last_update = datetime.min

    def update_top_symbols(self):
        if datetime.now() - self.last_update > timedelta(hours=4):
            try:
                tickers = exchange.fetch_tickers()
                valid_tickers = []
                exclude = ['USDC', 'DAI', 'FDUSD', 'USDE', 'BUSD', 'TUSD', 'PYUSD', 'USDD']
                for s, t in tickers.items():
                    if '/USDT' in s:
                        is_stable = any(ex in s for ex in exclude)
                        if not is_stable:
                            vol = t['quoteVolume'] if t.get('quoteVolume') else 0
                            valid_tickers.append({'symbol': s, 'vol': vol})
                            
                self.symbols = [x['symbol'] for x in sorted(valid_tickers, key=lambda x: x['vol'], reverse=True)[:50]]
                self.last_update = datetime.now()
                print(f"[{datetime.now().strftime('%H:%M')}] 更新監控清單: {len(self.symbols)} 幣種")
            except: 
                self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        return self.symbols

    def run_analysis(self):
        symbols = self.update_top_symbols()
        timeframes = ['30m', '1h', '4h'] 
        
        for symbol in symbols:
            for tf in timeframes:
                try:
                    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=300)
                    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
                    df = df.astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    side, entry, sl, tp1, tp2, _ = process_smc_data(df)
                    
                    if side:
                        signal_key = f"{symbol}_{tf}_{side}"
                        last_ts = self.last_signals.get(signal_key, 0)
                        current_ts = df['timestamp'].iloc[-1]
                        
                        if current_ts != last_ts:
                            self.send_discord(symbol, side, tf, entry, sl, tp1, tp2)
                            self.last_signals[signal_key] = current_ts
                    
                    time.sleep(0.1) 
                except Exception as e:
                    print(f"Error {symbol}: {e}")

    def send_discord(self, symbol, side, interval, entry, sl, tp1, tp2):
        # 修改 2: 使用 timezone.utc 來修復 DeprecationWarning
        # datetime.now(timezone.utc) 獲取當前 UTC 時間，再加上 8 小時
        tw_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime('%H:%M')
        
        side_cn = "做多" if side == "LONG" else "做空"
        exchange_name = "BYBIT"
        
        def fmt(num): 
            if num < 1: return f"{num:.5f}".rstrip('0').rstrip('.')
            return f"{num:.4f}".rstrip('0').rstrip('.')
            
        try:
            risk = abs(entry - sl)
            reward_tp1 = abs(tp1 - entry)
            rr_ratio = reward_tp1 / risk if risk > 0 else 0
            rr_str = f"1:{rr_ratio:.1f}"
        except:
            rr_str = "N/A"

        tp2_rr_str = "1:1"

        msg = (
            f"🚨\n"
            f"{symbol} 訊號 {exchange_name}\n"
            f"方向 {side_cn}\n"
            f"週期:{interval.upper()}\n"
            f"進場:{fmt(entry)}\n"
            f"SL:{fmt(sl)}\n"
            f"TP1: {fmt(tp1)}({rr_str})\n"
            f"TP2: {fmt(tp2)}({tp2_rr_str})\n\n"
            f"偵測時間: 台灣時間 {tw_time}"
        )
        
        payload = {"content": msg}
        
        try:
            requests.post(DISCORD_URL, json=payload)
            print(f"✅ 已發送 SMC 訊號: {symbol} {side} ({rr_str})")
        except Exception as e:
            print(f"Discord 發送失敗: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    print("🚀 Zeabur SMC OrderBlock Bot (Fix Timezone) 已啟動...")
    print(f"策略參數: PIVOT_LEN={PIVOT_LEN}")
    
    # 測試訊號
    bot.send_discord("TEST/USDT", "LONG", "1h", 1.2345, 1.2000, 1.3000, 1.2690)
    
    while True:
        bot.run_analysis()
        time.sleep(60)
