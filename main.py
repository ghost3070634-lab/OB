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
# 請確認此處的 Webhook URL 是正確的
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1457246379242950797/LB6npSWu5J9ZbB8NYp90N-gpmDrjOK2qPqtkaB5AP6YztzdfzmBF6oxesKJybWQ04xoU")

# 交易所設定
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# 策略參數
VIDYA_LEN = 10
VIDYA_MOM = 20
CCI_LEN = 200
ATR_LEN = 5
SWING_Yz = 5 # 用於檢測波段高低點的窗口大小 (模擬 OB)

# ==========================================
# 2. 指標計算邏輯 (核心演算法)
# ==========================================
def calculate_vidya(df, length=10, momentum=20):
    """計算 VIDYA 指標"""
    src = df['close']
    mom = src.diff()
    
    pos_mom = mom.where(mom >= 0, 0).rolling(momentum).sum()
    neg_mom = (-mom.where(mom < 0, 0)).rolling(momentum).sum()
    
    denominator = pos_mom + neg_mom
    cmo = (100 * (pos_mom - neg_mom) / denominator.replace(0, 1)).abs()
    
    alpha = 2 / (length + 1)
    vidya = np.zeros_like(src)
    vidya[:] = np.nan
    
    start_idx = momentum 
    if start_idx < len(src):
        vidya[start_idx] = src.iloc[start_idx]

    src_values = src.values
    cmo_values = cmo.values
    
    for i in range(start_idx + 1, len(df)):
        val_alpha = (alpha * cmo_values[i] / 100)
        prev_vidya = vidya[i-1] if not np.isnan(vidya[i-1]) else src_values[i]
        vidya[i] = val_alpha * src_values[i] + (1 - val_alpha) * prev_vidya
        
    return ta.sma(pd.Series(vidya), length=15)

def get_swing_levels(df, lookback=10):
    """
    計算波段高低點 (模擬 OB/BOS 位置)
    回傳: 最近的一個高點(High) 和 最近的一個低點(Low)
    """
    highs = df['high']
    lows = df['low']
    
    # 簡單的波段檢測：如果該點是前後 N 根K線的最高/低點
    # 這裡使用 rolling max/min 來近似
    # 實務上我們取最近的顯著高低點
    
    # 取得最近 50 根 K 線
    recent_df = df.iloc[-50:].copy()
    
    # 尋找局部高點
    swing_highs = recent_df['high'][(recent_df['high'].shift(1) < recent_df['high']) & (recent_df['high'].shift(-1) < recent_df['high'])]
    # 尋找局部低點
    swing_lows = recent_df['low'][(recent_df['low'].shift(1) > recent_df['low']) & (recent_df['low'].shift(-1) > recent_df['low'])]
    
    return swing_highs, swing_lows

def process_data(df):
    """計算所有需要的指標並產生訊號"""
    if len(df) < 250: return None, None
    
    # 基礎指標
    df['ema7'] = ta.ema(df['close'], length=7)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['atr_200'] = ta.atr(df['high'], df['low'], df['close'], length=200)
    df['tr'] = ta.true_range(df['high'], df['low'], df['close'])
    
    # VIDYA & Trend Up
    df['vidya_sma'] = calculate_vidya(df, VIDYA_LEN, VIDYA_MOM)
    df['upper_band'] = df['vidya_sma'] + df['atr_200'] * 2
    df['lower_band'] = df['vidya_sma'] - df['atr_200'] * 2
    
    # 計算 is_trend_up
    is_trend_up = np.full(len(df), False)
    close_vals = df['close'].values
    u_band = df['upper_band'].values
    l_band = df['lower_band'].values
    
    for i in range(1, len(df)):
        if np.isnan(u_band[i]): 
            is_trend_up[i] = is_trend_up[i-1]
            continue
        if close_vals[i] > u_band[i]:
            is_trend_up[i] = True
        elif close_vals[i] < l_band[i]:
            is_trend_up[i] = False
        else:
            is_trend_up[i] = is_trend_up[i-1]
            
    df['is_trend_up'] = is_trend_up

    # Buffer & Magic Trend
    sma_tr_5 = ta.sma(df['tr'], length=ATR_LEN)
    df['cci_200'] = ta.cci(df['high'], df['low'], df['close'], length=CCI_LEN)
    df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    buffer_up = np.zeros(len(df))
    buffer_dn = np.zeros(len(df))
    x_line = np.zeros(len(df))
    magic_trend = np.zeros(len(df))
    
    highs = df['high'].values
    lows = df['low'].values
    cci_200 = df['cci_200'].values
    atr_vals = sma_tr_5.values
    cci_20 = df['cci_20'].values
    
    for i in range(1, len(df)):
        curr_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0
        b_dn = highs[i] + curr_atr
        b_up = lows[i] - curr_atr
        prev_cci = cci_200[i-1]
        curr_cci = cci_200[i]
        
        if curr_cci >= 0 and prev_cci < 0: b_up = buffer_dn[i-1]
        if curr_cci <= 0 and prev_cci > 0: b_dn = buffer_up[i-1]
        
        if curr_cci >= 0:
            if b_up < buffer_up[i-1]: b_up = buffer_up[i-1]
        else:
            if b_dn > buffer_dn[i-1]: b_dn = buffer_dn[i-1]
            
        buffer_up[i] = b_up
        buffer_dn[i] = b_dn
        
        if curr_cci >= 0: x_line[i] = b_up
        elif curr_cci <= 0: x_line[i] = b_dn
        else: x_line[i] = x_line[i-1]
        
        up_t = lows[i] - curr_atr
        down_t = highs[i] + curr_atr
        prev_magic = magic_trend[i-1]
        
        if cci_20[i] >= 0:
            if up_t < prev_magic: magic_trend[i] = prev_magic
            else: magic_trend[i] = up_t
        else:
            if down_t > prev_magic: magic_trend[i] = prev_magic
            else: magic_trend[i] = down_t
            
    df['x'] = x_line
    df['magic_trend'] = magic_trend
    
    # 訊號判斷
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    cross_over_x = (prev['close'] <= prev['x']) and (curr['close'] > curr['x'])
    cross_under_x = (prev['close'] >= prev['x']) and (curr['close'] < curr['x'])
    cross_over_magic = (prev['close'] <= prev['magic_trend']) and (curr['close'] > curr['magic_trend'])
    cross_under_magic = (prev['close'] >= prev['magic_trend']) and (curr['close'] < curr['magic_trend'])
    cross_over_ema200 = (prev['close'] <= prev['ema200']) and (curr['close'] > curr['ema200'])
    cross_under_ema200 = (prev['close'] >= prev['ema200']) and (curr['close'] < curr['ema200'])

    sorignal = curr['cci_20'] >= 0
    bigmagicTrend = curr['cci_200'] >= 0
    
    original_long = (curr['is_trend_up'] and cross_over_x and cross_over_magic and curr['close'] > curr['ema200'] and curr['close'] > curr['ema7'] and curr['ema7'] > curr['ema21'])
    original_short = (not curr['is_trend_up'] and cross_under_x and cross_under_magic and curr['close'] < curr['ema200'] and curr['close'] < curr['ema7'] and curr['ema7'] < curr['ema21'])
    
    cross200_long = (sorignal and bigmagicTrend and curr['close'] > curr['ema7'] and curr['close'] > curr['ema21'] and cross_over_ema200)
    cross200_short = (not sorignal and not bigmagicTrend and curr['close'] < curr['ema7'] and curr['close'] < curr['ema21'] and cross_under_ema200)

    side = None
    if original_long or cross200_long:
        side = "LONG"
    elif original_short or cross200_short:
        side = "SHORT"
        
    return side, df

# ==========================================
# 3. 機器人主程式
# ==========================================
class TradingBot:
    def __init__(self):
        self.last_signals = {} 
        self.symbols = []
        self.last_update = datetime.min

    def update_top_symbols(self):
        """
        篩選邏輯：
        1. 獲取所有 USDT 對
        2. 排除穩定幣 (USDC, FDUSD, DAI, TUSD, USDE 等)
        3. 依照 Quote Volume (成交額) 排序，取前 50 名
        """
        if datetime.now() - self.last_update > timedelta(hours=4):
            try:
                tickers = exchange.fetch_tickers()
                valid_tickers = []
                # 擴充後的穩定幣排除名單
                exclude = ['USDC', 'DAI', 'FDUSD', 'USDE', 'BUSD', 'TUSD', 'PYUSD', 'USDD', 'EUR', 'GBP']
                
                for s, t in tickers.items():
                    if '/USDT' in s:
                        # 檢查 symbol 前綴是否在排除名單內 (例如 USDC/USDT)
                        base_currency = s.split('/')[0]
                        if base_currency not in exclude:
                            vol = t['quoteVolume'] if t.get('quoteVolume') else 0
                            # 過濾掉成交量過小的 (例如 < 100萬 U)
                            if vol > 1000000:
                                valid_tickers.append({'symbol': s, 'vol': vol})
                            
                # 排序並取前 50
                self.symbols = [x['symbol'] for x in sorted(valid_tickers, key=lambda x: x['vol'], reverse=True)[:50]]
                self.last_update = datetime.now()
                print(f"[{datetime.now().strftime('%H:%M')}] 更新監控幣種 (Top {len(self.symbols)})")
            except Exception as e:
                print(f"Update symbols error: {e}")
                self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT']
        return self.symbols

    def calculate_sl_tp(self, df, side):
        """
        修改後的 TP/SL 邏輯：
        1. SL: 放置在最近的波段高點/低點 (模擬 OB 上方/下方)
        2. TP1: 放置在反向的最近波段高點/低點 (模擬反向 OB/BOS)
        3. TP2: 1:1 盈虧比 (與 SL 距離相同)
        4. TP3: 1:2 盈虧比
        """
        curr = df.iloc[-1]
        entry = curr['close']
        
        # 取得波段高低點
        swing_highs, swing_lows = get_swing_levels(df)
        
        rr_ratio_str = "N/A" # 預設字串
        
        if side == "LONG":
            # SL: 找最近的一個波段低點 (Swing Low) 作為支撐下方
            # 如果找不到，使用 ATR 作為保底
            recent_lows = swing_lows[swing_lows < entry]
            if not recent_lows.empty:
                sl = recent_lows.iloc[-1] # 取最近的一個
            else:
                sl = entry - (ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] * 2)

            # TP1: 找上方最近的一個波段高點 (Swing High) 作為壓力
            recent_highs = swing_highs[swing_highs > entry]
            if not recent_highs.empty:
                tp1 = recent_highs.iloc[-1] # 取最近的一個 (通常是最近的阻力)
            else:
                # 如果上方沒有歷史高點 (突破新高)，用 1.5 倍風險距離
                tp1 = entry + abs(entry - sl) * 1.5

            # 計算盈虧比 (TP1)
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            if risk > 0:
                rr = reward / risk
                rr_ratio_str = f"1:{rr:.2f}"

            # TP2: 1:1 盈虧比
            tp2 = entry + risk
            
            # TP3 (保留原本的結構)
            tp3 = entry + (risk * 2)

        else: # SHORT
            # SL: 找最近的一個波段高點 (Swing High) 作為壓力上方
            recent_highs = swing_highs[swing_highs > entry]
            if not recent_highs.empty:
                sl = recent_highs.iloc[-1]
            else:
                sl = entry + (ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] * 2)

            # TP1: 找下方最近的一個波段低點 (Swing Low) 作為支撐
            recent_lows = swing_lows[swing_lows < entry]
            if not recent_lows.empty:
                tp1 = recent_lows.iloc[-1]
            else:
                tp1 = entry - abs(sl - entry) * 1.5

            # 計算盈虧比 (TP1)
            risk = abs(sl - entry)
            reward = abs(entry - tp1)
            if risk > 0:
                rr = reward / risk
                rr_ratio_str = f"1:{rr:.2f}"

            # TP2: 1:1
            tp2 = entry - risk
            tp3 = entry - (risk * 2)
            
        return entry, sl, tp1, tp2, tp3, rr_ratio_str

    def send_discord(self, symbol, side, interval, entry, sl, tp1, tp2, tp3, rr_str):
        # 強制加 8 小時 (台灣時間)
        tw_time = (datetime.utcnow() + timedelta(hours=8)).strftime('%H:%M')
        side_cn = "做多" if side == "LONG" else "做空"
        exchange_name = "BYBIT"
        
        def fmt(num): return f"{num:.4f}".rstrip('0').rstrip('.')
        
        msg = (
            f"🚨\n"
            f"{symbol} 訊號 {exchange_name}\n"
            f"方向 {side_cn}\n"
            f"週期:{interval.upper()}\n"
            f"進場:{fmt(entry)}\n"
            f"SL:{fmt(sl)}\n"
            f"TP1: {fmt(tp1)} (盈虧比 {rr_str})\n"
            f"TP2: {fmt(tp2)} (1:1)\n"
            f"偵測時間: 台灣時間 {tw_time}"
            # TP3 可選擇是否顯示，這裡依照您原本格式TP2為止
        )
        
        payload = {"content": msg}
        try:
            requests.post(DISCORD_URL, json=payload)
            print(f"已發送: {symbol} {side}")
        except Exception as e:
            print(f"Discord 失敗: {e}")

    def send_test_signal(self):
        """發送測試推播"""
        print("正在發送測試推播...")
        self.send_discord(
            symbol="TEST/USDT",
            side="LONG",
            interval="TEST",
            entry=1.2345,
            sl=1.2000,
            tp1=1.2800,
            tp2=1.2690,
            tp3=1.3000,
            rr_str="1:1.32"
        )

    def run_analysis(self):
        symbols = self.update_top_symbols()
        timeframes = ['15m', '30m', '1h']
        
        for symbol in symbols:
            for tf in timeframes:
                try:
                    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=500)
                    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
                    df = df.astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    side, df_result = process_data(df)
                    
                    if side:
                        signal_key = f"{symbol}_{tf}_{side}"
                        last_ts = self.last_signals.get(signal_key, 0)
                        current_ts = df['timestamp'].iloc[-1]
                        
                        if current_ts != last_ts:
                            entry, sl, tp1, tp2, tp3, rr_str = self.calculate_sl_tp(df_result, side)
                            self.send_discord(symbol, side, tf, entry, sl, tp1, tp2, tp3, rr_str)
                            self.last_signals[signal_key] = current_ts
                    time.sleep(0.1) # 避免 API 請求過快
                except Exception as e:
                    # 某些幣種可能會報錯，忽略即可
                    pass

if __name__ == "__main__":
    bot = TradingBot()
    print("🚀 Zeabur Trading Bot (SMC TP Logic + Filter Update) 已啟動...")
    
    # 啟動時發送一次測試訊號
    bot.send_test_signal()
    
    while True:
        try:
            bot.run_analysis()
        except Exception as e:
            print(f"Main Loop Error: {e}")
        time.sleep(60)
