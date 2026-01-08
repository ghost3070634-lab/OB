import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import requests
import os
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. 配置設定
# ==========================================
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1458076152504520771/HZoL7eh3KpncZW7zbZEqBMqM0SHJDczVXmCBJHv9QuXV8qfLlEUPCCDs-Z4LnLcaGQ_B")

exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

PIVOT_LEN = 10  

# ==========================================
# 2. SMC 策略核心邏輯 (含 OB 計數)
# ==========================================
def process_smc_data(df):
    if len(df) < 100: return None, None, None, None, None, None, None

    # 1. 識別 Pivot Points
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
    
    # 狀態變數
    current_trend = 0      # 1=多頭, -1=空頭
    trend_seq = 0          # 目前是趨勢中的第幾個 OB
    
    # 最終訊號變數
    final_side = None
    final_entry = 0
    final_sl = 0
    final_tp1 = 0
    final_tp2 = 0
    final_ob_seq = 0       # 用來回傳該訊號是第幾個 OB
    
    for i in range(start_idx, len(df)):
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        
        # --- A. 更新潛在 OB (Pivots) ---
        pivot_idx = i - PIVOT_LEN
        if pivot_idx >= 0:
            if is_ph[pivot_idx]:
                last_swing_high = highs[pivot_idx]
                last_pivot_high_candle = {
                    'type': 'bear',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx], 
                    'mitigated': False,
                    'idx': pivot_idx,
                    'seq': 0 # 之後會填入
                }
            
            if is_pl[pivot_idx]:
                last_swing_low = lows[pivot_idx]
                last_pivot_low_candle = {
                    'type': 'bull',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx],
                    'mitigated': False,
                    'idx': pivot_idx,
                    'seq': 0 # 之後會填入
                }

        # --- B. 判斷結構破壞 (MSS / BOS) ---
        
        # === 向上突破 (潛在多頭) ===
        if curr_close > last_swing_high:
            # 判斷是 MSS 還是 BOS
            if current_trend != 1:
                # 趨勢從 空/無 -> 多 (MSS)
                current_trend = 1
                trend_seq = 1  # 重置計數為 1
            else:
                # 趨勢維持多頭 (BOS)
                trend_seq += 1 # 計數遞增
                
            # 產生多頭 OB
            if last_pivot_low_candle is not None:
                if not any(ob['idx'] == last_pivot_low_candle['idx'] for ob in obs):
                    new_ob = last_pivot_low_candle.copy()
                    new_ob['seq'] = trend_seq  # 寫入這是第幾個 OB
                    obs.append(new_ob)
            
        # === 向下跌破 (潛在空頭) ===
        elif curr_close < last_swing_low:
            # 判斷是 MSS 還是 BOS
            if current_trend != -1:
                # 趨勢從 多/無 -> 空 (MSS)
                current_trend = -1
                trend_seq = 1  # 重置計數為 1
            else:
                # 趨勢維持空頭 (BOS)
                trend_seq += 1 # 計數遞增
                
            # 產生空頭 OB
            if last_pivot_high_candle is not None:
                if not any(ob['idx'] == last_pivot_high_candle['idx'] for ob in obs):
                    new_ob = last_pivot_high_candle.copy()
                    new_ob['seq'] = trend_seq  # 寫入這是第幾個 OB
                    obs.append(new_ob)
            
        # --- C. 判斷進場 (回踩 OB) ---
        
        # [做多]
        if current_trend == 1: 
            # 找未被測試過的 Bullish OB
            valid_obs = [ob for ob in obs if ob['type'] == 'bull' and not ob['mitigated'] and ob['top'] < curr_close]
            
            if valid_obs:
                target_ob = valid_obs[-1] # 取最新的
                
                # 價格回踩 OB 上緣
                if curr_low <= target_ob['top']:
                    if i == len(df) - 1: # 必須是當前 K 棒發生的
                        final_side = "LONG"
                        final_entry = target_ob['top']
                        final_sl = target_ob['bottom']
                        final_tp1 = last_swing_high
                        risk = final_entry - final_sl
                        final_tp2 = final_entry + risk if risk > 0 else final_entry * 1.01
                        final_ob_seq = target_ob['seq'] # 抓取這個 OB 的序號 (1, 2, 3...)
                        
                    target_ob['mitigated'] = True
        
        # [做空]
        elif current_trend == -1:
            # 找未被測試過的 Bearish OB
            valid_obs = [ob for ob in obs if ob['type'] == 'bear' and not ob['mitigated'] and ob['bottom'] > curr_close]
            
            if valid_obs:
                target_ob = valid_obs[-1] # 取最新的
                
                # 價格回彈 OB 下緣
                if curr_high >= target_ob['bottom']:
                    if i == len(df) - 1: # 必須是當前 K 棒發生的
                        final_side = "SHORT"
                        final_entry = target_ob['bottom']
                        final_sl = target_ob['top']
                        final_tp1 = last_swing_low
                        risk = final_sl - final_entry
                        final_tp2 = final_entry - risk if risk > 0 else final_entry * 0.99
                        final_ob_seq = target_ob['seq'] # 抓取這個 OB 的序號 (1, 2, 3...)
                        
                    target_ob['mitigated'] = True

        # --- D. 清理失效 OB ---
        for ob in obs:
            if not ob['mitigated']:
                if ob['type'] == 'bull' and curr_close < ob['bottom']:
                    ob['mitigated'] = True 
                elif ob['type'] == 'bear' and curr_close > ob['top']:
                    ob['mitigated'] = True

    return final_side, final_entry, final_sl, final_tp1, final_tp2, final_ob_seq, df

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
                print(f"[{datetime.now().strftime('%H:%M')}] 更新監控: {len(self.symbols)} 幣種")
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
                    
                    # 接收 ob_seq
                    side, entry, sl, tp1, tp2, ob_seq, _ = process_smc_data(df)
                    
                    if side:
                        # 將 ob_seq 寫入 key，避免同方向但不同 OB 的訊號被擋
                        signal_key = f"{symbol}_{tf}_{side}_{ob_seq}"
                        last_ts = self.last_signals.get(signal_key, 0)
                        current_ts = df['timestamp'].iloc[-1]
                        
                        if current_ts != last_ts:
                            self.send_discord(symbol, side, tf, entry, sl, tp1, tp2, ob_seq)
                            self.last_signals[signal_key] = current_ts
                    
                    time.sleep(0.1) 
                except Exception as e:
                    print(f"Error {symbol}: {e}")

    def send_discord(self, symbol, side, interval, entry, sl, tp1, tp2, ob_seq):
        tw_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime('%H:%M')
        
        # 顯示處理：方向 做多(1)
        side_cn = "做多" if side == "LONG" else "做空"
        
        # 這裡就是你要的格式核心
        direction_str = f"{side_cn}({ob_seq})"
        
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

        # 輸出格式對齊
        msg = (
            f"{symbol} 訊號 {exchange_name}\n"
            f"方向 {direction_str}\n"
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
            print(f"✅ 已發送: {symbol} {direction_str}")
        except Exception as e:
            print(f"Discord 發送失敗: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    print("🚀 Zeabur SMC Bot (OB計數版) 已啟動...")
    print(f"策略參數: PIVOT_LEN={PIVOT_LEN}")
    
    # 測試顯示格式是否正確
    bot.send_discord("BCH/USDT", "LONG", "30m", 627.3, 622.2, 653.3, 632.4, 1)
    
    while True:
        bot.run_analysis()
        time.sleep(60)
