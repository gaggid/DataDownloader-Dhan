# Part 1: Imports and Basic Setup
import mysql.connector
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime, timedelta


class TechnicalIndicatorCalculator:
    def __init__(self):
        """Initialize the calculator with database configuration and device setup."""
        self.db_config = {
            'host': 'localhost',
            'user': 'dhan_hq',
            'password': 'Passw0rd@098',
            'database': 'dhanhq_db',
            'auth_plugin': 'mysql_native_password',
            'use_pure': True
        }
        # Add progress tracking attributes
        self.start_time = None
        self.processed_symbols = 0
        self.total_symbols = 0
        self.total_records_processed = 0
        self.errors_encountered = 0
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def log_progress(self, message, level="INFO"):
        """Log progress with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def pad_tensor(self, tensor, target_length, pad_value=0):
        """Pad tensor to target length."""
        current_length = len(tensor)
        if current_length < target_length:
            padding = torch.full((target_length - current_length,), pad_value, 
                            dtype=tensor.dtype, device=tensor.device)
            return torch.cat([padding, tensor])
        return tensor[:target_length]
    
  
    def estimate_remaining_time(self):
        """Estimate remaining processing time."""
        if self.processed_symbols == 0:
            return "Calculating..."
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_symbol = elapsed_time / self.processed_symbols
        remaining_symbols = self.total_symbols - self.processed_symbols
        remaining_seconds = remaining_symbols * avg_time_per_symbol
        
        return str(timedelta(seconds=int(remaining_seconds)))
    def print_progress_stats(self):
        """Print current progress statistics."""
        elapsed_time = time.time() - self.start_time
        remaining_time = self.estimate_remaining_time()
        
        self.log_progress(
            f"\nProgress Update:"
            f"\n- Processed: {self.processed_symbols}/{self.total_symbols} symbols"
            f"\n- Completion: {(self.processed_symbols/self.total_symbols)*100:.2f}%"
            f"\n- Records processed: {self.total_records_processed:,}"
            f"\n- Errors encountered: {self.errors_encountered}"
            f"\n- Elapsed time: {str(timedelta(seconds=int(elapsed_time)))}"
            f"\n- Estimated remaining time: {remaining_time}\n"
        )

    def connect_to_db(self):
        """Establish database connection."""
        return mysql.connector.connect(**self.db_config)

    def fetch_historical_data(self, symbol):
        """Fetch historical data for a given symbol."""
        conn = self.connect_to_db()
        query = """
            SELECT date, trading_symbol, open, high, low, close, volume 
            FROM historical_data 
            WHERE trading_symbol = %s 
            ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(symbol,))
        conn.close()
        return df

    def moving_average(self, tensor, window):
        """Calculate moving average using PyTorch."""
        weights = torch.ones(window).to(self.device) / window
        return torch.conv1d(
            tensor.view(1, 1, -1), 
            weights.view(1, 1, -1), 
            padding=window-1
        )[0, 0, window-1:]

    def exponential_moving_average(self, tensor, span):
        """Calculate EMA using PyTorch."""
        alpha = 2.0 / (span + 1)
        weights = torch.pow(1 - alpha, torch.arange(len(tensor)-1, -1, -1).float()).to(self.device)
        weights = weights / weights.sum()
        return torch.conv1d(
            tensor.view(1, 1, -1),
            weights.view(1, 1, -1),
            padding=len(weights)-1
        )[0, 0, len(weights)-1:]

    def calculate_standard_deviation(self, tensor, window):
        """Calculate rolling standard deviation."""
        mean = self.moving_average(tensor, window)
        squared_diff = (tensor - mean) ** 2
        variance = self.moving_average(squared_diff, window)
        return torch.sqrt(variance + 1e-8)

    def calculate_atr(self, high, low, close, period):
        """Calculate Average True Range."""
        high_low = high - low
        high_close_prev = torch.abs(high[1:] - close[:-1])
        low_close_prev = torch.abs(low[1:] - close[:-1])
        
        tr = torch.maximum(high_low[1:], 
                        torch.maximum(high_close_prev, low_close_prev))
        tr = torch.cat([high_low[0:1], tr])
        return self.exponential_moving_average(tr, period)
    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index (ADX)."""
        # True Range
        high_low = high - low
        high_close = torch.abs(high - torch.roll(close, 1))
        low_close = torch.abs(low - torch.roll(close, 1))
        tr = torch.maximum(high_low, torch.maximum(high_close, low_close))
        
        # Directional Movement
        up_move = high - torch.roll(high, 1)
        down_move = torch.roll(low, 1) - low
        
        pos_dm = torch.where((up_move > down_move) & (up_move > 0), up_move, torch.zeros_like(up_move))
        neg_dm = torch.where((down_move > up_move) & (down_move > 0), down_move, torch.zeros_like(down_move))
        
        # Smoothed TR and DM
        smoothed_tr = self.exponential_moving_average(tr, period)
        smoothed_pos_dm = self.exponential_moving_average(pos_dm, period)
        smoothed_neg_dm = self.exponential_moving_average(neg_dm, period)
        
        # Directional Indicators
        pos_di = 100 * smoothed_pos_dm / (smoothed_tr + 1e-10)
        neg_di = 100 * smoothed_neg_dm / (smoothed_tr + 1e-10)
        
        # ADX
        dx = 100 * torch.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = self.exponential_moving_average(dx, period)
        
        return adx, pos_di, neg_di

    def calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price - torch.roll(typical_price, 1)
        positive_flow = torch.where(delta > 0, money_flow, torch.zeros_like(money_flow))
        negative_flow = torch.where(delta < 0, money_flow, torch.zeros_like(money_flow))
        
        positive_mf = self.moving_average(positive_flow, period)
        negative_mf = self.moving_average(negative_flow, period)
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi

    def calculate_vwap(self, high, low, close, volume):
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = torch.cumsum(typical_price * volume, dim=0) / (torch.cumsum(volume, dim=0) + 1e-10)
        return vwap

    def calculate_ichimoku(self, high, low, close):
        """Calculate Ichimoku Cloud components."""
        # Conversion Line (Tenkan-sen)
        high_9 = torch.stack([high[i:i+9].max() for i in range(len(high)-8)])
        low_9 = torch.stack([low[i:i+9].min() for i in range(len(low)-8)])
        conversion = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = torch.stack([high[i:i+26].max() for i in range(len(high)-25)])
        low_26 = torch.stack([low[i:i+26].min() for i in range(len(low)-25)])
        base = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        span_a = (conversion + base) / 2
        
        # Leading Span B (Senkou Span B)
        high_52 = torch.stack([high[i:i+52].max() for i in range(len(high)-51)])
        low_52 = torch.stack([low[i:i+52].min() for i in range(len(low)-51)])
        span_b = (high_52 + low_52) / 2
        
        return conversion, base, span_a, span_b

    def calculate_support_resistance(self, high, low, close, window=20):
        """Calculate support and resistance levels using pivot points."""
        pivot = (high + low + close) / 3
        support1 = 2 * pivot - high
        resistance1 = 2 * pivot - low
        
        # Smooth the levels
        support1 = self.moving_average(support1, window)
        resistance1 = self.moving_average(resistance1, window)
        return support1, resistance1

    def calculate_fibonacci_levels(self, high, low):
        """Calculate Fibonacci retracement levels."""
        price_range = high - low
        fib_23_6 = high - (price_range * 0.236)
        fib_38_2 = high - (price_range * 0.382)
        fib_50_0 = high - (price_range * 0.500)
        fib_61_8 = high - (price_range * 0.618)
        return fib_23_6, fib_38_2, fib_50_0, fib_61_8

    def calculate_momentum_indicators(self, close, volume):
        """Calculate various momentum indicators with strictly bounded volume ratio."""
        try:
            # Price Momentum Rate (5-day)
            price_momentum = (close / torch.roll(close, 5) - 1) * 100
            
            # Volume Momentum Rate (5-day)
            volume_momentum = (volume / torch.roll(volume, 5) - 1) * 100
            
            # Volume Ratio calculation with strict bounds
            up_volume = torch.where(close > torch.roll(close, 1), volume, torch.zeros_like(volume))
            down_volume = torch.where(close < torch.roll(close, 1), volume, torch.zeros_like(volume))
            
            # Use exponential moving average for smoothing
            up_vol_ma = self.exponential_moving_average(up_volume, 10)
            down_vol_ma = self.exponential_moving_average(down_volume, 10)
            
            # Initialize volume ratio
            volume_ratio = torch.ones_like(close)  # Default to neutral (1.0)
            
            # Safe division with strict bounds
            valid_mask = down_vol_ma >= 1e-6
            raw_ratio = torch.where(valid_mask,
                                up_vol_ma / (down_vol_ma + 1e-6),
                                torch.ones_like(close))
            
            # Normalize the ratio to 0-100 range using log scale
            # log(1) = 0, which will be our neutral point
            log_ratio = torch.log(raw_ratio + 1e-6)
            
            # Scale to 0-100 range
            scaled_ratio = 50 * (1 + torch.tanh(log_ratio))
            
            # Final cleanup
            volume_ratio = torch.clamp(scaled_ratio, min=0.0, max=100.0)
            
            # Replace any remaining invalid values
            volume_ratio = torch.nan_to_num(volume_ratio, nan=50.0, posinf=100.0, neginf=0.0)
            
            return price_momentum, volume_momentum, volume_ratio

        except Exception as e:
            print(f"Error in calculate_momentum_indicators: {str(e)}")
            raise

    def calculate_volatility(self, high, low, close):
        """Calculate volatility indicators."""
        # Price Range
        price_range = high - low

        # Position within range
        highest_high = torch.stack([high[i:i+20].max() for i in range(len(high)-19)])
        lowest_low = torch.stack([low[i:i+20].min() for i in range(len(low)-19)])

        # Calculate price position with proper handling of edge cases
        denominator = highest_high - lowest_low
        price_position = torch.zeros_like(close)

        # Handle the case where denominator is 0 or very small
        valid_mask = denominator > 1e-10
        price_position[valid_mask] = ((close[valid_mask] - lowest_low[valid_mask]) / 
                                    denominator[valid_mask]) * 100

        # Clamp values between 0 and 100
        price_position = torch.clamp(price_position, min=0.0, max=100.0)

        return price_range, price_position, highest_high, lowest_low
    
    def calculate_zigzag(self, high, low, deviation_percentage=5.0):
        """
        Calculate ZigZag indicator.

        Args:
            high: High prices tensor
            low: Low prices tensor
            deviation_percentage: Minimum percentage change for a pivot point (default 5%)

        Returns:
            Tensor containing zigzag values
        """
        try:
            n = len(high)
            zigzag = torch.zeros_like(high)
            
            # Initialize variables
            swing_high = high[0]
            swing_low = low[0]
            is_uptrend = True
            last_pivot_idx = 0
            
            for i in range(1, n):
                current_high = high[i]
                current_low = low[i]
                
                if is_uptrend:
                    # Look for higher highs
                    if current_high > swing_high:
                        swing_high = current_high
                        last_pivot_idx = i
                    # Check for trend reversal
                    elif current_low < swing_low * (1 - deviation_percentage/100):
                        # Mark the last swing high
                        zigzag[last_pivot_idx] = swing_high
                        # Start downtrend
                        is_uptrend = False
                        swing_low = current_low
                        last_pivot_idx = i
                else:
                    # Look for lower lows
                    if current_low < swing_low:
                        swing_low = current_low
                        last_pivot_idx = i
                    # Check for trend reversal
                    elif current_high > swing_high * (1 + deviation_percentage/100):
                        # Mark the last swing low
                        zigzag[last_pivot_idx] = swing_low
                        # Start uptrend
                        is_uptrend = True
                        swing_high = current_high
                        last_pivot_idx = i
            
            # Mark the last pivot point
            if is_uptrend:
                zigzag[last_pivot_idx] = swing_high
            else:
                zigzag[last_pivot_idx] = swing_low
            
            return zigzag

        except Exception as e:
            print(f"Error calculating ZigZag: {str(e)}")
            return torch.zeros_like(high)
        
    def calculate_obv(self, close, volume):
        """Calculate On Balance Volume with proper normalization."""
        try:
            # Normalize volume first
            normalized_volume = self.normalize_volume(volume)
            
            obv = torch.zeros_like(normalized_volume)
            
            # Calculate OBV
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + normalized_volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - normalized_volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # Apply rolling normalization to prevent accumulation of large values
            window_size = 20
            rolling_max = torch.zeros_like(obv)
            rolling_min = torch.zeros_like(obv)
            
            for i in range(window_size, len(obv)):
                window = obv[i-window_size:i]
                rolling_max[i] = window.max()
                rolling_min[i] = window.min()
            
            # Scale OBV to a more reasonable range (0-100)
            range_denominator = rolling_max - rolling_min + 1e-10
            normalized_obv = ((obv - rolling_min) / range_denominator) * 100
            
            return normalized_obv

        except Exception as e:
            self.log_progress(f"Error in calculate_obv: {str(e)}", "ERROR")
            return torch.zeros_like(volume)
    
    def calculate_volume_sma(self, volume):
        """Calculate Volume SMA with proper normalization."""
        try:
            # Normalize volume first
            normalized_volume = self.normalize_volume(volume)
            
            volume_sma = torch.zeros_like(normalized_volume)
            window_size = 20
            
            for i in range(window_size-1, len(normalized_volume)):
                window = normalized_volume[i-window_size+1:i+1]
                # Remove outliers within the window
                mean = window.mean()
                std = window.std()
                valid_mask = (window - mean).abs() <= 2 * std
                filtered_window = window[valid_mask]
                
                if len(filtered_window) > 0:
                    volume_sma[i] = filtered_window.mean()
                else:
                    volume_sma[i] = mean
            
            # Scale to a reasonable range
            max_sma = volume_sma.max()
            if max_sma > 0:
                volume_sma = (volume_sma / max_sma) * 1000  # Scale to 0-1000 range
            
            return volume_sma

        except Exception as e:
            self.log_progress(f"Error in calculate_volume_sma: {str(e)}", "ERROR")
            return torch.zeros_like(volume)
            
    def normalize_volume(self, volume):
        """Normalize volume data to handle extreme values."""
        try:
            # Convert to tensor if not already
            if not isinstance(volume, torch.Tensor):
                volume = torch.tensor(volume, dtype=torch.float32).to(self.device)
            
            # Calculate statistics
            median_volume = torch.median(volume)
            mean_volume = torch.mean(volume)
            std_volume = torch.std(volume)
            
            # Define reasonable limits
            lower_bound = torch.max(torch.tensor(0.0).to(self.device), 
                                mean_volume - 3 * std_volume)
            upper_bound = mean_volume + 3 * std_volume
            
            # Clip extreme values
            normalized_volume = torch.clamp(volume, lower_bound, upper_bound)
            
            # Scale to 0-1 range
            min_vol = normalized_volume.min()
            max_vol = normalized_volume.max()
            if max_vol > min_vol:
                normalized_volume = (normalized_volume - min_vol) / (max_vol - min_vol)
            else:
                normalized_volume = torch.zeros_like(volume)
            
            return normalized_volume
        
        except Exception as e:
            self.log_progress(f"Error in normalize_volume: {str(e)}", "ERROR")
            return volume
        
    
    def calculate_volume_metrics(self, close, volume):
        """Calculate volume metrics with proper normalization."""
        try:
            n = len(volume)
            volume_sma = torch.zeros_like(volume)
            volume_ratio = torch.zeros_like(volume)

            # Normalize volume first
            normalized_volume = self.normalize_volume(volume)

            # Calculate Volume SMA
            for i in range(19, n):
                window = normalized_volume[i-19:i+1]
                # Remove outliers within the window
                mean = window.mean()
                std = window.std()
                valid_mask = (window - mean).abs() <= 2 * std
                filtered_window = window[valid_mask]
                
                volume_sma[i] = filtered_window.mean() if len(filtered_window) > 0 else mean

            # Calculate Volume Ratio
            for i in range(19, n):
                try:
                    # Get volume windows
                    vol_window = normalized_volume[i-9:i+1]
                    price_window = close[i-10:i+1]
                    
                    # Calculate price changes
                    price_changes = price_window[1:] > price_window[:-1]
                    
                    # Calculate up and down volume
                    up_vol = torch.sum(vol_window[price_changes])
                    down_vol = torch.sum(vol_window[~price_changes])
                    
                    # Calculate ratio with bounds
                    if down_vol > 0:
                        ratio = up_vol / down_vol
                        # Scale ratio to 0-100 range using sigmoid-like function
                        volume_ratio[i] = 50 * (1 + torch.tanh(torch.log(ratio + 1e-6)))
                    else:
                        volume_ratio[i] = 100 if up_vol > 0 else 50  # Default to neutral if no volume
                    
                except Exception as e:
                    volume_ratio[i] = 50  # Default to neutral on error
                    continue

            # Final normalization
            volume_sma = volume_sma * 100  # Scale to percentage
            volume_ratio = torch.clamp(volume_ratio, 0, 100)  # Ensure ratio is between 0 and 100

            return volume_sma, volume_ratio

        except Exception as e:
            self.log_progress(f"Error in calculate_volume_metrics: {str(e)}", "ERROR")
            return torch.zeros_like(volume), torch.ones_like(volume) * 50

    def calculate_indicators(self, df):
        """Calculate all technical indicators using PyTorch."""
        try:
            def safe_index(tensor, idx):
                """Safely access tensor index with bounds checking."""
                if idx < 0:
                    return tensor[0]
                if idx >= len(tensor):
                    return tensor[-1]
                return tensor[idx]

            def safe_window(tensor, start, end):
                """Safely get a window of tensor values with bounds checking."""
                safe_start = max(0, start)
                safe_end = min(len(tensor), end)
                if safe_start >= safe_end:
                    return tensor[0:1]  # Return single value if window is invalid
                return tensor[safe_start:safe_end]

            # Convert price data to PyTorch tensors
            close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device)
            high = torch.tensor(df['high'].values, dtype=torch.float32).to(self.device)
            low = torch.tensor(df['low'].values, dtype=torch.float32).to(self.device)
            volume = torch.tensor(df['volume'].values, dtype=torch.float32).to(self.device)

            # Get the length of data
            n = len(close)
            if n < 365:  # Minimum required length for all calculations
                raise ValueError(f"Insufficient data points: {n} (minimum required: 365)")

            # Initialize results DataFrame
            results = pd.DataFrame()
            results['date'] = df['date']
            results['trading_symbol'] = df['trading_symbol']

            def to_numpy(tensor):
                """Safely convert tensor to numpy with value clamping."""
                if isinstance(tensor, torch.Tensor):
                    # Clamp extreme values
                    tensor = torch.nan_to_num(tensor, 0.0)  # Replace NaN with 0
                    tensor = torch.clamp(tensor, -1e6, 1e6)  # Clamp to reasonable range
                    return tensor.cpu().numpy()
                return tensor

            # ----------------------
            # Moving Averages
            # ----------------------
            # Initialize arrays
            sma_20 = torch.zeros_like(close)
            sma_50 = torch.zeros_like(close)
            sma_200 = torch.zeros_like(close)
            ema_20 = torch.zeros_like(close)
            ema_50 = torch.zeros_like(close)

            # Calculate SMAs
            for i in range(n):
                if i >= 19:
                    sma_20[i] = close[i-19:i+1].mean()
                else:
                    sma_20[i] = close[:i+1].mean()
                
                if i >= 49:
                    sma_50[i] = close[i-49:i+1].mean()
                else:
                    sma_50[i] = close[:i+1].mean()
                
                if i >= 199:
                    sma_200[i] = close[i-199:i+1].mean()
                else:
                    sma_200[i] = close[:i+1].mean()

            # Calculate EMAs
            alpha_20 = 2.0 / (20 + 1)
            alpha_50 = 2.0 / (50 + 1)
            
            ema_20[0] = close[0]
            ema_50[0] = close[0]
            
            for i in range(1, n):
                ema_20[i] = alpha_20 * close[i] + (1 - alpha_20) * ema_20[i-1]
                ema_50[i] = alpha_50 * close[i] + (1 - alpha_50) * ema_50[i-1]

            results['sma_20'] = to_numpy(sma_20)
            results['sma_50'] = to_numpy(sma_50)
            results['sma_200'] = to_numpy(sma_200)
            results['ema_20'] = to_numpy(ema_20)
            results['ema_50'] = to_numpy(ema_50)

            # ----------------------
            # RSI Calculation
            # ----------------------
            rsi = torch.zeros_like(close)
            delta = torch.zeros_like(close)
            delta[1:] = close[1:] - close[:-1]
            
            gain = torch.zeros_like(delta)
            loss = torch.zeros_like(delta)
            
            gain[delta > 0] = delta[delta > 0]
            loss[delta < 0] = -delta[delta < 0]
            
            avg_gain = torch.zeros_like(close)
            avg_loss = torch.zeros_like(close)
            
            # Initialize first 14 periods
            avg_gain[13] = gain[:14].mean()
            avg_loss[13] = loss[:14].mean()
            
            # Calculate subsequent values
            for i in range(14, n):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            results['rsi_14'] = to_numpy(rsi)

            # ----------------------
            # MACD
            # ----------------------
            ema_12 = torch.zeros_like(close)
            ema_26 = torch.zeros_like(close)
            
            ema_12[0] = close[0]
            ema_26[0] = close[0]
            
            alpha_12 = 2.0 / (12 + 1)
            alpha_26 = 2.0 / (26 + 1)
            
            for i in range(1, n):
                ema_12[i] = alpha_12 * close[i] + (1 - alpha_12) * ema_12[i-1]
                ema_26[i] = alpha_26 * close[i] + (1 - alpha_26) * ema_26[i-1]
            
            macd_line = ema_12 - ema_26
            
            macd_signal = torch.zeros_like(close)
            macd_signal[0] = macd_line[0]
            alpha_9 = 2.0 / (9 + 1)
            
            for i in range(1, n):
                macd_signal[i] = alpha_9 * macd_line[i] + (1 - alpha_9) * macd_signal[i-1]
            
            results['macd_line'] = to_numpy(macd_line)
            results['macd_signal'] = to_numpy(macd_signal)
            results['macd_histogram'] = to_numpy(macd_line - macd_signal)

            # ----------------------
            # Bollinger Bands
            # ----------------------
            bb_middle = torch.zeros_like(close)
            bb_upper = torch.zeros_like(close)
            bb_lower = torch.zeros_like(close)
            
            for i in range(n):
                if i >= 19:
                    window = close[i-19:i+1]
                    bb_middle[i] = window.mean()
                    std = window.std()
                    bb_upper[i] = bb_middle[i] + (2 * std)
                    bb_lower[i] = bb_middle[i] - (2 * std)
                else:
                    window = close[:i+1]
                    bb_middle[i] = window.mean()
                    std = window.std() if len(window) > 1 else torch.tensor(0.0).to(self.device)
                    bb_upper[i] = bb_middle[i] + (2 * std)
                    bb_lower[i] = bb_middle[i] - (2 * std)
            
            results['bollinger_middle'] = to_numpy(bb_middle)
            results['bollinger_upper'] = to_numpy(bb_upper)
            results['bollinger_lower'] = to_numpy(bb_lower)

            # ----------------------
            # ATR Calculation
            # ----------------------
            tr = torch.zeros_like(close)
            atr = torch.zeros_like(close)
            
            # Calculate True Range with proper index handling
            tr[0] = high[0] - low[0]  # First TR is just the first day's range
            for i in range(1, n):
                try:
                    hl = high[i] - low[i]
                    hc = torch.abs(high[i] - close[i-1])
                    lc = torch.abs(low[i] - close[i-1])
                    tr[i] = torch.max(torch.max(hl, hc), lc)
                except Exception as e:
                    tr[i] = tr[i-1]  # Use previous value if calculation fails
            
            # Calculate ATR with proper initialization
            atr[0:14] = tr[0:14].mean()  # Initialize first 14 periods with TR mean
            for i in range(14, n):
                try:
                    atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                except Exception as e:
                    atr[i] = atr[i-1]  # Use previous value if calculation fails
            
            results['atr_14'] = to_numpy(atr)

            # Calculate ZigZag
            zigzag = self.calculate_zigzag(high, low)
            results['zigzag'] = to_numpy(zigzag)

            # ----------------------
            # ADX Calculation
            # ----------------------
            adx = torch.zeros_like(close)
            plus_di = torch.zeros_like(close)
            minus_di = torch.zeros_like(close)
            
            # Calculate +DM and -DM
            high_diff = high[1:] - high[:-1]
            low_diff = low[:-1] - low[1:]
            
            plus_dm = torch.zeros_like(close)
            minus_dm = torch.zeros_like(close)
            
            for i in range(1, n):
                if high_diff[i-1] > low_diff[i-1] and high_diff[i-1] > 0:
                    plus_dm[i] = high_diff[i-1]
                if low_diff[i-1] > high_diff[i-1] and low_diff[i-1] > 0:
                    minus_dm[i] = low_diff[i-1]
            
            # Calculate TR, +DI14, -DI14
            tr = torch.zeros_like(close)
            smoothed_tr = torch.zeros_like(close)
            smoothed_plus_dm = torch.zeros_like(close)
            smoothed_minus_dm = torch.zeros_like(close)
            
            for i in range(1, n):
                tr[i] = torch.max(torch.max(
                    high[i] - low[i],
                    torch.abs(high[i] - close[i-1])),
                    torch.abs(low[i] - close[i-1])
                )
            
            # Initialize first smoothed values
            smoothed_tr[13] = tr[0:14].sum()
            smoothed_plus_dm[13] = plus_dm[0:14].sum()
            smoothed_minus_dm[13] = minus_dm[0:14].sum()
            
            # Calculate smoothed values
            for i in range(14, n):
                smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1]/14) + tr[i]
                smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1]/14) + plus_dm[i]
                smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1]/14) + minus_dm[i]
                
                # Calculate +DI14 and -DI14
                plus_di[i] = 100 * smoothed_plus_dm[i] / (smoothed_tr[i] + 1e-10)
                minus_di[i] = 100 * smoothed_minus_dm[i] / (smoothed_tr[i] + 1e-10)
                
                # Calculate DX
                dx = 100 * torch.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i] + 1e-10)
                
                # Calculate ADX
                if i == 14:
                    adx[i] = dx
                else:
                    adx[i] = ((adx[i-1] * 13) + dx) / 14
            
            results['adx_14'] = to_numpy(adx)
            results['di_plus_14'] = to_numpy(plus_di)
            results['di_minus_14'] = to_numpy(minus_di)
            # ----------------------
            # Keltner Channels
            # ----------------------
            keltner_middle = torch.zeros_like(close)
            keltner_upper = torch.zeros_like(close)
            keltner_lower = torch.zeros_like(close)
            
            # Use EMA-20 for middle line
            keltner_middle[0] = close[0]
            alpha_20 = 2.0 / (20 + 1)
            
            for i in range(1, n):
                try:
                    keltner_middle[i] = alpha_20 * close[i] + (1 - alpha_20) * keltner_middle[i-1]
                    keltner_upper[i] = keltner_middle[i] + (2 * safe_index(atr, i))
                    keltner_lower[i] = keltner_middle[i] - (2 * safe_index(atr, i))
                except Exception as e:
                    # Use previous values if calculation fails
                    keltner_middle[i] = keltner_middle[i-1]
                    keltner_upper[i] = keltner_upper[i-1]
                    keltner_lower[i] = keltner_lower[i-1]
            
            results['keltner_middle'] = to_numpy(keltner_middle)
            results['keltner_upper'] = to_numpy(keltner_upper)
            results['keltner_lower'] = to_numpy(keltner_lower)

            # ----------------------
            # Stochastic Oscillator
            # ----------------------
            stoch_k = torch.zeros_like(close)
            stoch_d = torch.zeros_like(close)
            
            for i in range(13, n):
                window_low = low[max(0, i-13):i+1].min()
                window_high = high[max(0, i-13):i+1].max()
                stoch_k[i] = 100 * (close[i] - window_low) / (window_high - window_low + 1e-10)
            
            # Calculate %D (3-period SMA of %K)
            for i in range(15, n):
                stoch_d[i] = stoch_k[i-2:i+1].mean()
            
            results['stochastic_k'] = to_numpy(stoch_k)
            results['stochastic_d'] = to_numpy(stoch_d)

            # ----------------------
            # Williams %R
            # ----------------------
            williams_r = torch.zeros_like(close)
            
            for i in range(13, n):
                window_low = low[max(0, i-13):i+1].min()
                window_high = high[max(0, i-13):i+1].max()
                williams_r[i] = -100 * (window_high - close[i]) / (window_high - window_low + 1e-10)
            
            results['williams_r'] = to_numpy(williams_r)

            # ----------------------
            # Money Flow Index
            # ----------------------
            mfi = torch.zeros_like(close)
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            pos_flow = torch.zeros_like(close)
            neg_flow = torch.zeros_like(close)
            
            for i in range(1, n):
                if typical_price[i] > typical_price[i-1]:
                    pos_flow[i] = money_flow[i]
                else:
                    neg_flow[i] = money_flow[i]
            
            for i in range(13, n):
                pos_sum = pos_flow[i-13:i+1].sum()
                neg_sum = neg_flow[i-13:i+1].sum()
                mfi[i] = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
            
            results['mfi_14'] = to_numpy(mfi)

            # ----------------------
            # VWAP
            # ----------------------
            vwap = torch.zeros_like(close)
            cumul_vol = torch.zeros_like(close)
            cumul_vol_price = torch.zeros_like(close)
            
            for i in range(n):
                cumul_vol[i] = volume[:i+1].sum()
                cumul_vol_price[i] = (volume[:i+1] * ((high[:i+1] + low[:i+1] + close[:i+1]) / 3)).sum()
                vwap[i] = cumul_vol_price[i] / (cumul_vol[i] + 1e-10)
            
            results['vwap'] = to_numpy(vwap)

            # ----------------------
            # Ichimoku Cloud
            # ----------------------
            conversion = torch.zeros_like(close)
            base = torch.zeros_like(close)
            span_a = torch.zeros_like(close)
            span_b = torch.zeros_like(close)
            
            for i in range(8, n):
                high_9 = high[i-8:i+1].max()
                low_9 = low[i-8:i+1].min()
                conversion[i] = (high_9 + low_9) / 2
            
            for i in range(25, n):
                high_26 = high[i-25:i+1].max()
                low_26 = low[i-25:i+1].min()
                base[i] = (high_26 + low_26) / 2
                span_a[i] = (conversion[i] + base[i]) / 2
            
            for i in range(51, n):
                high_52 = high[i-51:i+1].max()
                low_52 = low[i-51:i+1].min()
                span_b[i] = (high_52 + low_52) / 2
            
            results['ichimoku_conversion'] = to_numpy(conversion)
            results['ichimoku_base'] = to_numpy(base)
            results['ichimoku_span_a'] = to_numpy(span_a)
            results['ichimoku_span_b'] = to_numpy(span_b)

            # ----------------------
            # On Balance Volume (OBV)
            # ----------------------
            # Calculate OBV with normalization
            obv = self.calculate_obv(close, volume)
            results['obv'] = to_numpy(obv)

            # ----------------------
            # CCI (Commodity Channel Index)
            # ----------------------
            cci = torch.zeros_like(close)
            typical_price = (high + low + close) / 3
            for i in range(19, n):
                window = typical_price[i-19:i+1]
                sma = window.mean()
                mean_deviation = torch.abs(window - sma).mean()
                cci[i] = (typical_price[i] - sma) / (0.015 * mean_deviation + 1e-10)
            results['cci_20'] = to_numpy(cci)

            # ----------------------
            # Volatility (20-day)
            # ----------------------
            volatility = torch.zeros_like(close)
            for i in range(19, n):
                window = close[i-19:i+1]
                volatility[i] = (window.std() / window.mean()) * 100
            results['volatility_20'] = to_numpy(volatility)

            # ----------------------
            # Chaikin Money Flow
            # ----------------------
            cmf = torch.zeros_like(close)
            for i in range(19, n):
                window_high = high[i-19:i+1]
                window_low = low[i-19:i+1]
                window_close = close[i-19:i+1]
                window_volume = volume[i-19:i+1]
                
                mf_multiplier = ((window_close - window_low) - (window_high - window_close)) / (window_high - window_low + 1e-10)
                mf_volume = mf_multiplier * window_volume
                cmf[i] = mf_volume.sum() / (window_volume.sum() + 1e-10)
            results['chaikin_money_flow'] = to_numpy(cmf)

            # ----------------------
            # Standard Deviation
            # ----------------------
            std_dev = torch.zeros_like(close)
            for i in range(19, n):
                std_dev[i] = close[i-19:i+1].std()
            results['standard_deviation_20'] = to_numpy(std_dev)

            # ----------------------
            # Price Position
            # ----------------------
            price_position = torch.zeros_like(close)
            for i in range(n):
                try:
                    window_high = safe_window(high, i-19, i+1).max()
                    window_low = safe_window(low, i-19, i+1).min()
                    if window_high > window_low:
                        price_position[i] = ((close[i] - window_low) / (window_high - window_low)) * 100
                    else:
                        price_position[i] = 50.0  # Default to middle when high equals low
                except Exception as e:
                    price_position[i] = 50.0  # Default to middle on error
            
            # Clamp values between 0 and 100
            price_position = torch.clamp(price_position, min=0.0, max=100.0)
            results['price_position'] = to_numpy(price_position)

            # ----------------------
            # Price Levels
            # ----------------------
            highest_high = torch.zeros_like(close)
            lowest_low = torch.zeros_like(close)
            
            for i in range(19, n):
                highest_high[i] = high[i-19:i+1].max()
                lowest_low[i] = low[i-19:i+1].min()
            
            results['highest_high_20'] = to_numpy(highest_high)
            results['lowest_low_20'] = to_numpy(lowest_low)
            
            # Modified price position calculation with clamping
            price_position = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
            # Clamp values between 0 and 100
            price_position = torch.clamp(price_position, min=0.0, max=100.0)
            results['price_position'] = to_numpy(price_position)

            # ----------------------
            # Average Directional Index (Already calculated in ADX section)
            # ----------------------
            results['average_directional_index'] = results['adx_14']  # Use existing ADX calculation

            # ----------------------
            # Parabolic SAR
            # ----------------------
            sar = torch.zeros_like(close)
            af = 0.02  # Acceleration factor
            max_af = 0.2
            is_long = True
            ep = low[0]  # Extreme point
            sar[0] = high[0]

            for i in range(1, n):
                prev_sar = sar[i-1]
                
                if is_long:
                    sar[i] = prev_sar + af * (ep - prev_sar)
                    if low[i] < sar[i]:
                        is_long = False
                        sar[i] = ep
                        ep = low[i]
                        af = 0.02
                    else:
                        if high[i] > ep:
                            ep = high[i]
                            af = min(af + 0.02, max_af)
                else:
                    sar[i] = prev_sar - af * (prev_sar - ep)
                    if high[i] > sar[i]:
                        is_long = True
                        sar[i] = ep
                        ep = high[i]
                        af = 0.02
                    else:
                        if low[i] < ep:
                            ep = low[i]
                            af = min(af + 0.02, max_af)
            results['parabolic_sar'] = to_numpy(sar)

            # ----------------------
            # TRIX
            # ----------------------
            # Triple smoothed EMA
            ema1 = torch.zeros_like(close)
            ema2 = torch.zeros_like(close)
            ema3 = torch.zeros_like(close)
            trix = torch.zeros_like(close)

            alpha = 2.0 / (15 + 1)  # 15-period TRIX

            ema1[0] = close[0]
            for i in range(1, n):
                ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]

            ema2[0] = ema1[0]
            for i in range(1, n):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]

            ema3[0] = ema2[0]
            for i in range(1, n):
                ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
                if i > 0:
                    trix[i] = (ema3[i] - ema3[i-1]) / (ema3[i-1] + 1e-10) * 100
                    
            results['trix'] = to_numpy(trix)

            # ----------------------
            # Ultimate Oscillator
            # ----------------------
            uo = torch.zeros_like(close)
            for i in range(6, n):
                bp_7 = torch.zeros(7)
                tr_7 = torch.zeros(7)
                bp_14 = torch.zeros(14)
                tr_14 = torch.zeros(14)
                bp_28 = torch.zeros(28)
                tr_28 = torch.zeros(28)
                
                for j in range(7):
                    bp_7[j] = close[i-j] - torch.min(low[i-j], close[i-j-1])
                    tr_7[j] = torch.max(high[i-j], close[i-j-1]) - torch.min(low[i-j], close[i-j-1])
                
                for j in range(14):
                    bp_14[j] = close[i-j] - torch.min(low[i-j], close[i-j-1])
                    tr_14[j] = torch.max(high[i-j], close[i-j-1]) - torch.min(low[i-j], close[i-j-1])
                
                for j in range(28):
                    bp_28[j] = close[i-j] - torch.min(low[i-j], close[i-j-1])
                    tr_28[j] = torch.max(high[i-j], close[i-j-1]) - torch.min(low[i-j], close[i-j-1])
                
                avg7 = bp_7.sum() / (tr_7.sum() + 1e-10)
                avg14 = bp_14.sum() / (tr_14.sum() + 1e-10)
                avg28 = bp_28.sum() / (tr_28.sum() + 1e-10)
                
                uo[i] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)

            results['ultimate_oscillator'] = to_numpy(uo)

            # ----------------------
            # Rate of Change
            # ----------------------
            roc = torch.zeros_like(close)
            for i in range(10, n):
                roc[i] = ((close[i] - close[i-10]) / close[i-10]) * 100
            results['rate_of_change'] = to_numpy(roc)

            # ----------------------
            # Gap Up/Down
            # ----------------------
            gap = torch.zeros_like(close)
            for i in range(1, n):
                gap[i] = ((close[i] - close[i-1]) / close[i-1]) * 100
            results['gap_up_down'] = to_numpy(gap)

            # ----------------------
            # Volume Trend
            # ----------------------
            vol_trend = torch.zeros_like(close)
            for i in range(1, n):
                if volume[i] > volume[i-1]:
                    vol_trend[i] = 1
                elif volume[i] < volume[i-1]:
                    vol_trend[i] = -1
            results['volume_trend'] = to_numpy(vol_trend)

            # ----------------------
            # Support and Resistance Levels
            # ----------------------
            support = torch.zeros_like(close)
            resistance = torch.zeros_like(close)
            window_size = 20

            for i in range(window_size, n):
                window = close[i-window_size:i]
                window_low = window.min()
                window_high = window.max()
                
                # Simple support/resistance based on local mins/maxs
                support[i] = window_low
                resistance[i] = window_high

            results['support_level_1'] = to_numpy(support)
            results['resistance_level_1'] = to_numpy(resistance)

            # ----------------------
            # Fibonacci Levels
            # ----------------------
            fib_23_6 = torch.zeros_like(close)
            fib_38_2 = torch.zeros_like(close)
            fib_50_0 = torch.zeros_like(close)
            fib_61_8 = torch.zeros_like(close)

            for i in range(window_size, n):
                window_high = high[i-window_size:i].max()
                window_low = low[i-window_size:i].min()
                price_range = window_high - window_low
                
                fib_23_6[i] = window_high - (price_range * 0.236)
                fib_38_2[i] = window_high - (price_range * 0.382)
                fib_50_0[i] = window_high - (price_range * 0.500)
                fib_61_8[i] = window_high - (price_range * 0.618)

            results['fibonacci_23_6'] = to_numpy(fib_23_6)
            results['fibonacci_38_2'] = to_numpy(fib_38_2)
            results['fibonacci_50_0'] = to_numpy(fib_50_0)
            results['fibonacci_61_8'] = to_numpy(fib_61_8)

            # ----------------------
            # Price Momentum and Volume Metrics
            # ----------------------
            price_momentum = torch.zeros_like(close)
            volume_momentum = torch.zeros_like(close)
            
            for i in range(5, n):
                price_momentum[i] = ((close[i] / close[i-5]) - 1) * 100
                volume_momentum[i] = ((volume[i] / volume[i-5]) - 1) * 100
            
            results['price_momentum_rate'] = to_numpy(price_momentum)
            results['volume_momentum_rate'] = to_numpy(volume_momentum)
            results['price_range'] = to_numpy(high - low)

            # ----------------------
            # Volume Analysis
            # ----------------------
            # In your calculate_indicators method, replace the Volume Analysis section with:
            volume_sma, volume_ratio = self.calculate_volume_metrics(close, volume)
            results['volume_sma_20'] = to_numpy(volume_sma)
            results['volume_ratio'] = to_numpy(volume_ratio)

            # ----------------------
            # Price Levels
            # ----------------------
            highest_high = torch.zeros_like(close)
            lowest_low = torch.zeros_like(close)
            
            for i in range(19, n):
                highest_high[i] = high[i-19:i+1].max()
                lowest_low[i] = low[i-19:i+1].min()
            
            results['highest_high_20'] = to_numpy(highest_high)
            results['lowest_low_20'] = to_numpy(lowest_low)
            
            price_position = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
            results['price_position'] = to_numpy(price_position)

            # ----------------------
            # Returns Calculation
            # ----------------------
            returns_dict = self.calculate_returns(close)
            for period, values in returns_dict.items():
                results[period] = to_numpy(values)

            # Add debug logging for returns
            self.log_progress("Returns calculation debug info:")
            for period in [20, 55, 90, 180, 365]:
                return_col = f'return_{period}'
                if return_col in results:
                    recent_returns = results[return_col][-5:]  # Last 5 values
                    self.log_progress(f"{return_col} last 5 values: {recent_returns}")

            # Replace inf and -inf with 0
            results = results.replace([np.inf, -np.inf], 0)
            
            # Replace NaN with 0
            results = results.fillna(0)
            
            # Round numeric columns to 6 decimal places
            numeric_columns = results.select_dtypes(include=[np.number]).columns
            results[numeric_columns] = results[numeric_columns].round(6)

            return results

        except Exception as e:
            print(f"Error in calculate_indicators: {str(e)}")
            raise
    def save_indicators(self, df):
        """Save calculated indicators to the database with strict validation."""
        try:
            # Volume metrics validation
            if 'volume_sma_20' in df.columns:
                orig_max = df['volume_sma_20'].max()
                orig_min = df['volume_sma_20'].min()
                df['volume_sma_20'] = df['volume_sma_20'].clip(0, 100)
                if orig_max > 100 or orig_min < 0:
                    self.log_progress(
                        f"Volume SMA adjusted: Original range [{orig_min:.2f}, {orig_max:.2f}] "
                        f"-> New range [{df['volume_sma_20'].min():.2f}, {df['volume_sma_20'].max():.2f}]"
                    )

            if 'volume_ratio' in df.columns:
                orig_max = df['volume_ratio'].max()
                orig_min = df['volume_ratio'].min()
                df['volume_ratio'] = df['volume_ratio'].clip(0, 100)
                if orig_max > 100 or orig_min < 0:
                    self.log_progress(
                        f"Volume ratio adjusted: Original range [{orig_min:.2f}, {orig_max:.2f}] "
                        f"-> New range [{df['volume_ratio'].min():.2f}, {df['volume_ratio'].max():.2f}]"
                    )
            
            # Existing validations
            if 'price_position' in df.columns:
                df['price_position'] = df['price_position'].clip(0, 100)
            
            # Replace inf and -inf with 0
            df = df.replace([np.inf, -np.inf], 0)
            
            # Replace NaN with 0
            df = df.fillna(0)
            
            # Round numeric columns to 6 decimal places
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(6)

            conn = self.connect_to_db()
            cursor = conn.cursor()
            
            required_columns = [
                'date', 'trading_symbol', 
                'sma_20', 'sma_50', 'sma_200', 
                'ema_20', 'ema_50',
                'rsi_14',
                'macd_line', 'macd_signal', 'macd_histogram',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'atr_14', 
                'obv',
                'volume_sma_20',
                'return_20', 'return_55', 'return_90', 'return_180', 'return_365',
                'stochastic_k', 'stochastic_d',
                'williams_r',
                'cci_20',
                'adx_14', 'di_plus_14', 'di_minus_14',
                'price_momentum_rate', 'volume_momentum_rate',
                'volatility_20',
                'chaikin_money_flow',
                'mfi_14',
                'vwap',
                'standard_deviation_20',
                'keltner_upper', 'keltner_middle', 'keltner_lower',
                'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b',
                'price_channel_high', 'price_channel_low',
                'average_directional_index',
                'parabolic_sar',
                'zigzag',
                'trix',
                'ultimate_oscillator',
                'rate_of_change',
                'gap_up_down',
                'price_range',
                'volume_ratio',
                'highest_high_20', 'lowest_low_20',
                'price_position',
                'volume_trend',
                'support_level_1', 'resistance_level_1',
                'fibonacci_23_6', 'fibonacci_38_2', 'fibonacci_50_0', 'fibonacci_61_8'
            ]

            # Check for missing columns and add them with zeros if necessary
            for col in required_columns:
                if col not in df.columns and col not in ['date', 'trading_symbol']:
                    df[col] = 0.0

            # Create the query with exact matching placeholders
            placeholders = ', '.join(['%s'] * len(required_columns))
            columns_str = ', '.join(required_columns)
            
            query = f"""
                INSERT INTO technical_indicators (
                    {columns_str}
                ) VALUES (
                    {placeholders}
                )
            """

            # Verify column count matches placeholder count
            placeholder_count = query.count('%s')
            if placeholder_count != len(required_columns):
                raise ValueError(f"Mismatch between placeholders ({placeholder_count}) and columns ({len(required_columns)})")

            # Batch insert for better performance
            values_list = []
            for _, row in df.iterrows():
                values = tuple(row[col] for col in required_columns)
                values_list.append(values)

            # Execute batch insert
            cursor.executemany(query, values_list)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error in save_indicators: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise
    def get_last_processed_date(self, symbol):
        """Get the last processed date for a given symbol."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) 
                FROM technical_indicators 
                WHERE trading_symbol = %s
            """, (symbol,))
            last_date = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return last_date
        except Exception as e:
            print(f"Error getting last processed date for {symbol}: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None

    def fetch_new_historical_data(self, symbol, last_date):
        """Fetch only new historical data after the last processed date."""
        conn = self.connect_to_db()
        query = """
            SELECT date, trading_symbol, open, high, low, close, volume 
            FROM historical_data 
            WHERE trading_symbol = %s 
            AND date > %s
            ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(symbol, last_date))
        conn.close()
        return df

    def fetch_lookback_data(self, symbol, start_date, lookback_days=365):
        """Fetch historical data including lookback period for accurate calculations."""
        try:
            conn = self.connect_to_db()
            
            # First, get the lookback data
            lookback_query = """
                SELECT date, trading_symbol, open, high, low, close, volume 
                FROM historical_data 
                WHERE trading_symbol = %s 
                AND date BETWEEN DATE_SUB(%s, INTERVAL %s DAY) AND %s
                ORDER BY date
            """
            lookback_df = pd.read_sql(lookback_query, conn, 
                                    params=(symbol, start_date, lookback_days, start_date))
            
            # Then, get the new data
            new_data_query = """
                SELECT date, trading_symbol, open, high, low, close, volume 
                FROM historical_data 
                WHERE trading_symbol = %s 
                AND date > %s
                ORDER BY date
            """
            new_df = pd.read_sql(new_data_query, conn, params=(symbol, start_date))
            
            # Combine the data
            df = pd.concat([lookback_df, new_df])
            
            conn.close()
            return df
        except Exception as e:
            self.log_progress(f"Error in fetch_lookback_data for {symbol}: {str(e)}", "ERROR")
            if 'conn' in locals():
                conn.close()
            raise
    
    def calculate_returns(self, close):
        """Calculate returns for different periods."""
        n = len(close)
        returns = {}
        periods = [20, 55, 90, 180, 365]
        
        for period in periods:
            return_values = torch.zeros_like(close)
            for i in range(period, n):
                return_values[i] = ((close[i] / close[i-period]) - 1) * 100
            returns[f'return_{period}'] = return_values
        
        return returns
    def validate_returns(self, df):
        """
        Validate return calculations and print diagnostic information.
        """
        periods = [20, 55, 90, 180, 365]
        for period in periods:
            col_name = f'return_{period}'
            if col_name in df.columns:
                non_zero = (df[col_name] != 0).sum()
                total = len(df)
                print(f"\n{col_name} statistics:")
                print(f"Non-zero values: {non_zero}/{total} ({(non_zero/total)*100:.2f}%)")
                print("Recent values:")
                print(df[['date', col_name]].tail())
                print(f"Min: {df[col_name].min():.2f}")
                print(f"Max: {df[col_name].max():.2f}")
                print(f"Mean: {df[col_name].mean():.2f}")
    
    def process_symbol_data(self, symbol, df, last_processed_date):
        """Process data for a single symbol with proper handling of incremental updates."""
        try:
            # Calculate indicators on the full dataset
            df_with_indicators = self.calculate_indicators(df)
            
            # Clean and validate the data
            df_with_indicators = self.clean_and_validate_data(df_with_indicators)
            
            # If this is an incremental update, only save the new records
            if last_processed_date:
                df_to_save = df_with_indicators[df_with_indicators['date'] > last_processed_date]
            else:
                df_to_save = df_with_indicators
            
            return df_to_save
        except Exception as e:
            self.log_progress(f"Error processing data for {symbol}: {str(e)}", "ERROR")
            raise
    
    def clean_and_validate_data(self, df):
        """Clean and validate the calculated indicators."""
        try:
            # Replace inf and -inf with 0
            df = df.replace([np.inf, -np.inf], 0)
            
            # Replace NaN with 0
            df = df.fillna(0)
            
            # Clamp values for various indicators
            if 'price_position' in df.columns:
                df['price_position'] = df['price_position'].clip(0, 100)
            
            if 'volume_ratio' in df.columns:
                df['volume_ratio'] = df['volume_ratio'].clip(0, 1000)
            
            # Round numeric columns to 6 decimal places
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(6)
            
            return df
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            raise

    def process_all_symbols(self):
        """Process all symbols with progress tracking."""
        try:
            # Initialize timing and progress tracking
            self.start_time = time.time()
            self.processed_symbols = 0
            self.total_records_processed = 0
            self.errors_encountered = 0
            
            # Get list of symbols that need processing
            conn = self.connect_to_db()
            cursor = conn.cursor()
            
            self.log_progress("Fetching symbols that need processing...")
            cursor.execute("""
                SELECT DISTINCT h.trading_symbol
                FROM historical_data h
                LEFT JOIN (
                    SELECT trading_symbol, MAX(date) as last_date
                    FROM technical_indicators
                    GROUP BY trading_symbol
                ) t ON h.trading_symbol = t.trading_symbol
                WHERE t.last_date IS NULL 
                    OR h.date > t.last_date
            """)
            
            symbols_to_process = [row[0] for row in cursor.fetchall()]
            self.total_symbols = len(symbols_to_process)
            
            self.log_progress(f"Found {self.total_symbols} symbols requiring updates")
            
            cursor.close()
            conn.close()

            # Process each symbol
            for symbol in symbols_to_process:
                try:
                    last_processed_date = self.get_last_processed_date(symbol)
                    
                    # Fetch appropriate data
                    if last_processed_date is None:
                        self.log_progress(f"Processing complete history for new symbol: {symbol}")
                        df = self.fetch_historical_data(symbol)
                    else:
                        self.log_progress(f"Processing incremental data for {symbol} after {last_processed_date}")
                        df = self.fetch_lookback_data(symbol, last_processed_date)

                    if len(df) < 365:
                        self.log_progress(f"Insufficient data for {symbol}, skipping...", "WARNING")
                        continue

                    # Process the data
                    df_to_save = self.process_symbol_data(symbol, df, last_processed_date)
                    
                    # Save the results
                    if not df_to_save.empty:
                        records_processed = len(df_to_save)
                        self.save_indicators(df_to_save)
                        self.total_records_processed += records_processed
                        self.log_progress(f"Processed {records_processed:,} records for {symbol}")
                    else:
                        self.log_progress(f"No new data to process for {symbol}")

                    # Calculate indicators
                    df_with_indicators = self.calculate_indicators(df)
                    
                    # Validate returns calculation
                    self.log_progress(f"\nValidating returns for {symbol}:")
                    self.validate_returns(df_with_indicators)
                    
                    # Data validation and cleaning
                    # Replace inf and -inf with 0
                    df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], 0)
                    
                    # Replace NaN with 0
                    df_with_indicators = df_with_indicators.fillna(0)
                    
                    # Clamp price_position values
                    if 'price_position' in df_with_indicators.columns:
                        df_with_indicators['price_position'] = df_with_indicators['price_position'].clip(0, 100)
                    
                    # Round numeric columns to 6 decimal places
                    numeric_columns = df_with_indicators.select_dtypes(include=[np.number]).columns
                    df_with_indicators[numeric_columns] = df_with_indicators[numeric_columns].round(6)
                    
                    # Filter new data if incremental update
                    if last_processed_date:
                        df_with_indicators = df_with_indicators[
                            df_with_indicators['date'] > last_processed_date
                        ]

                    # Save indicators
                    if not df_with_indicators.empty:
                        try:
                            # Additional validation before saving
                            if 'volume_ratio' in df_with_indicators.columns:
                                max_vol_ratio = df_with_indicators['volume_ratio'].max()
                                min_vol_ratio = df_with_indicators['volume_ratio'].min()
                                if max_vol_ratio > 1000 or min_vol_ratio < 0:
                                    self.log_progress(f"Warning: volume_ratio out of range for {symbol}: min={min_vol_ratio}, max={max_vol_ratio}", "WARNING")
                                    df_with_indicators['volume_ratio'] = df_with_indicators['volume_ratio'].clip(0, 1000)
                            
                            records_processed = len(df_with_indicators)
                            self.save_indicators(df_with_indicators)
                            self.total_records_processed += records_processed
                            self.log_progress(f"Processed {records_processed:,} records for {symbol}")
                        except Exception as save_error:
                            self.errors_encountered += 1
                            self.log_progress(f"Error saving indicators for {symbol}: {str(save_error)}", "ERROR")
                            continue
                    else:
                        self.log_progress(f"No new data to process for {symbol}")

                    # Update progress
                    self.processed_symbols += 1
                    
                    # Print progress every 10 symbols or at specific percentages
                    if (self.processed_symbols % 10 == 0) or \
                        (self.processed_symbols/self.total_symbols in [0.25, 0.5, 0.75]):
                        self.print_progress_stats()

                except Exception as e:
                    self.errors_encountered += 1
                    self.log_progress(f"Error processing {symbol}: {str(e)}", "ERROR")
                    # Print detailed error information
                    if 'df_with_indicators' in locals() and 'price_position' in df_with_indicators.columns:
                        print(f"Debug info for {symbol}:")
                        print(f"price_position stats:")
                        print(df_with_indicators['price_position'].describe())
                    continue

            # Final progress update
            self.print_progress_stats()
            
            # Print summary
            total_time = time.time() - self.start_time
            self.log_progress(
                f"\nProcessing Summary:"
                f"\n------------------------"
                f"\nTotal symbols processed: {self.processed_symbols:,}"
                f"\nTotal records processed: {self.total_records_processed:,}"
                f"\nTotal errors encountered: {self.errors_encountered}"
                f"\nTotal processing time: {str(timedelta(seconds=int(total_time)))}"
                f"\nAverage time per symbol: {total_time/self.processed_symbols:.2f} seconds"
                f"\n------------------------"
            )

        except Exception as e:
            self.log_progress(f"Fatal error in process_all_symbols: {str(e)}", "ERROR")
            raise

    def cleanup_duplicate_entries(self):
        """Clean up any duplicate entries that might have been created."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            
            # Create temporary table with unique entries
            cursor.execute("""
                CREATE TEMPORARY TABLE temp_indicators AS
                SELECT DISTINCT date, trading_symbol, 
                        MAX(id) as id
                FROM technical_indicators
                GROUP BY date, trading_symbol;
            """)

            # Delete duplicates
            cursor.execute("""
                DELETE ti FROM technical_indicators ti
                LEFT JOIN temp_indicators tmp 
                ON ti.id = tmp.id
                WHERE tmp.id IS NULL;
            """)

            conn.commit()
            cursor.close()
            conn.close()
            print("Duplicate cleanup completed successfully")

        except Exception as e:
            print(f"Error in cleanup_duplicate_entries: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise

def main():
    """Main execution function with progress tracking."""
    start_time = time.time()

    try:
        print("\n" + "="*50)
        print("Technical Indicator Calculator")
        print("="*50 + "\n")
        
        calculator = TechnicalIndicatorCalculator()
        print(f"Using device: {calculator.device}\n")
        
        # Process symbols
        calculator.process_all_symbols()
        
        # Clean up duplicates
        calculator.log_progress("Cleaning up duplicate entries...")
        calculator.cleanup_duplicate_entries()
        
        # Final execution summary
        total_execution_time = time.time() - start_time
        calculator.log_progress(
            f"\nExecution completed successfully!"
            f"\nTotal execution time: {str(timedelta(seconds=int(total_execution_time)))}"
        )
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()