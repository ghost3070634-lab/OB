import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import requests
import os
from datetime import datetime, timedelta

# ==========================================
# 1. 配置設定
# ==========================================
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1458076152504520771/HZoL7eh3KpncZW7zbZEqBMqM0SHJDczVXmCBJHv9QuXV8qfLlEUPCCDs-Z4LnLcaGQ_B")

exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# SMC 參數
PIVOT_LEN = 5  # 定義碎形(Fractal)的左右 K 線數量，用於判斷 OB 和結構

# ==========================================
# 2. SMC 策略核心邏輯 (已修改為 LuxAlgo 邏輯：BOS 觸發 OB)
# ==========================================
def process_smc_data(df):
    """
    計算 SMC 指標：市場結構(Structure) 與 訂單塊(Order Block)
    修改版：模仿 LuxAlgo，在結構破壞(BOS)時才確認 OB
    """
    if len(df) < 100: return None, None, None, None, None, None

    # 1. 識別 Pivot Points (Swings)
    df['high_max'] = df['high'].rolling(window=PIVOT_LEN*2+1, center=True).max()
    df['low_min'] = df['low'].rolling(window=PIVOT_LEN*2+1, center=True).min()

    df['is_pivot_high'] = (df['high'] == df['high_max'])
    df['is_pivot_low'] = (df['low'] == df['low_min'])

    obs = [] 
    
    # 暫存最新的 Pivot K線資訊，等待 BOS 確認後轉為 OB
    last_pivot_high_candle = None
    last_pivot_low_candle = None

    # 轉為 numpy array 加速
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    is_ph = df['is_pivot_high'].values
    is_pl = df['is_pivot_low'].values
    
    last_swing_high = highs[0]
    last_swing_low = lows[0]

    start_idx = PIVOT_LEN * 2 + 1
    current_trend = 0 # 0: Unknown, 1: Bullish, -1: Bearish
    
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
                # 記錄這個 Pivot High 為潛在 Bearish OB (等待跌破 Swing Low 觸發)
                last_pivot_high_candle = {
                    'type': 'bear',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx], 
                    'mitigated': False,
                    'idx': pivot_idx
                }
            
            if is_pl[pivot_idx]:
                last_swing_low = lows[pivot_idx]
                # 記錄這個 Pivot Low 為潛在 Bullish OB (等待突破 Swing High 觸發)
                last_pivot_low_candle = {
                    'type': 'bull',
                    'top': highs[pivot_idx],
                    'bottom': lows[pivot_idx],
                    'mitigated': False,
                    'idx': pivot_idx
                }

        # --- 2. 判斷結構破壞 (BOS/CHoCH) 並生成 OB ---
        # LuxAlgo 邏輯：只有當結構被破壞(BOS)時，才將"造成這次破壞的起點(Pivot)"視為有效 OB
        
        # 情況 A: 向上突破 (Bullish BOS)
        if curr_close > last_swing_high:
            # 如果趨勢改變或延續，且我們有一個未加入的潛在 Bullish Pivot
            if current_trend != 1 or True: # 簡化邏輯：只要創新高就檢查是否要加入 OB
                if last_pivot_low_candle is not None:
                    # 防止重複加入同一個 Pivot (檢查 index)
                    if not any(ob['idx'] == last_pivot_low_candle['idx'] for ob in obs):
                        obs.append(last_pivot_low_candle.copy())
            current_trend = 1
            
        # 情況 B: 向下跌破 (Bearish BOS)
        elif curr_close < last_swing_low:
            if current_trend != -1 or True:
                if last_pivot_high_candle is not None:
                    if not any(ob['idx'] == last_pivot_high_candle['idx'] for ob in obs):
                        obs.append(last_pivot_high_candle.copy())
            current_trend = -1
            
        # --- 3. 判斷進場 (回踩 OB) ---
        # (保持原邏輯不變，僅變數來源 obs 已改變計算方式)
        if current_trend == 1: 
            valid_obs = [ob for ob in obs if ob['type'] == 'bull' and not ob['mitigated'] and ob['top'] < curr_close]
            if valid_obs:
                target_ob = valid_obs[-1]
                if curr_low <= target_ob['top'] and curr_close >= target_ob['bottom']:
                    if i == len(df) - 1:
                        final_side = "LONG"
                        final_entry = curr_close
                        final_sl = target_ob['bottom']
                        final_tp1 = last_swing_high
                        risk = final_entry - final_sl
                        final_tp2 = final_entry + risk if risk > 0 else final_entry * 1.01
                    target_ob['mitigated'] = True
                    
        elif current_trend == -1:
            valid_obs = [ob for ob in obs if ob['type'] == 'bear' and not ob['mitigated'] and ob['bottom'] > curr_close]
            if valid_obs:
                target_ob = valid_obs[-1]
                if curr_high >= target_ob['bottom'] and curr_close <= target_ob['top']:
                    if i == len(df) - 1:
                        final_side = "SHORT"
                        final_entry = curr_close
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
        # 維持原有的幣種篩選邏輯
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
        timeframes = ['30m', '1h','4H' ]
        
        for symbol in symbols:
            for tf in timeframes:
                try:
                    # 抓取足夠的 K 線以識別 Pivot 和 OB
                    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=300)
                    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
                    df = df.astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 執行 SMC 策略
                    side, entry, sl, tp1, tp2, _ = process_smc_data(df)
                    
                    if side:
                        signal_key = f"{symbol}_{tf}_{side}"
                        last_ts = self.last_signals.get(signal_key, 0)
                        current_ts = df['timestamp'].iloc[-1]
                        
                        # 簡單的去重機制：同一根 K 棒只發一次
                        if current_ts != last_ts:
                            self.send_discord(symbol, side, tf, entry, sl, tp1, tp2)
                            self.last_signals[signal_key] = current_ts
                    
                    time.sleep(0.1) # API 保護
                except Exception as e:
                    print(f"Error {symbol}: {e}")

    # ==========================================
    # 4. 通知格式 (依照你的 SMC 模板)
    # ==========================================
    def send_discord(self, symbol, side, interval, entry, sl, tp1, tp2):
        tw_time = (datetime.utcnow() + timedelta(hours=8)).strftime('%H:%M')
        
        side_cn = "做多" if side == "LONG" else "做空"
        exchange_name = "BYBIT"
        
        # 格式化數字 function
        def fmt(num): 
            if num < 1: return f"{num:.5f}".rstrip('0').rstrip('.')
            return f"{num:.4f}".rstrip('0').rstrip('.')
            
        # 計算盈虧比 (RR) 顯示在 TP1 後面
        # Risk = |Entry - SL|
        # Reward = |TP1 - Entry|
        # RR = Reward / Risk
        try:
            risk = abs(entry - sl)
            reward_tp1 = abs(tp1 - entry)
            rr_ratio = reward_tp1 / risk if risk > 0 else 0
            rr_str = f"1:{rr_ratio:.1f}"
        except:
            rr_str = "N/A"

        # TP2 固定顯示 1:1，因為程式邏輯是算在 1:1 的位置
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
    print("🚀 Zeabur SMC OrderBlock Bot 已啟動...")
    print("策略：SMC OB回踩 + 結構篩選 (LuxAlgo Logic)")
    
    # 測試訊號格式
    print("測試發送...")
    bot.send_discord("PENGU/USDT", "SHORT", "15m", 0.0127, 0.0133, 0.0123, 0.0121)
    
    while True:
        bot.run_analysis()
        time.sleep(60)
