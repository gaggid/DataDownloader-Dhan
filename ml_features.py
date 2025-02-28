import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz  # Import the pytz library
import tqdm  # Import tqdm for progress bars

class FeatureEngineering:
    def __init__(self, db_config):
        self.db_config = db_config

    def connect_to_db(self):
        return mysql.connector.connect(**self.db_config)

    def fetch_data(self, trading_symbols, start_date, end_date):
        """Fetches historical and technical indicator data for given symbols and date range."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor(dictionary=True)

            # Construct the IN clause for trading_symbols
            placeholders = ', '.join(['%s'] * len(trading_symbols))
            query = f"""
                SELECT
                    hd.date,
                    hd.trading_symbol,
                    hd.open,
                    hd.high,
                    hd.low,
                    hd.close,
                    hd.volume,
                    ti.sma_20, ti.sma_50, ti.sma_200,
                    ti.ema_20, ti.ema_50,
                    ti.macd_line, ti.macd_signal, ti.macd_histogram,
                    ti.adx_14, ti.di_plus_14, ti.di_minus_14,
                    ti.bollinger_upper, ti.bollinger_middle, ti.bollinger_lower,
                    ti.atr_14, ti.natr,
                    ti.rsi_14,
                    ti.stochastic_k, ti.stochastic_d,
                    ti.cci_20, ti.mfi_14, ti.williams_r, ti.roc,
                    ti.trix, ti.ultosc, ti.bop, ti.stddev, ti.var,
                    ti.return_20, ti.return_55, ti.return_90, ti.return_180, ti.return_365
                FROM
                    historical_data hd
                JOIN
                    technical_indicators ti ON hd.date = ti.date AND hd.trading_symbol = ti.trading_symbol
                WHERE
                    hd.trading_symbol IN ({placeholders})
                    AND hd.date BETWEEN %s AND %s
                ORDER BY
                    hd.trading_symbol, hd.date;
            """

            # Execute the query with the trading_symbols list and start/end dates
            params = trading_symbols + [start_date, end_date]
            cursor.execute(query, params)
            data = cursor.fetchall()

            conn.close()
            return pd.DataFrame(data)

        except Exception as e:
            print(f"Error fetching data: {e}")
            if 'conn' in locals():
                conn.close()
            return pd.DataFrame()

    def create_features(self, df, prediction_horizon=5, lag_days=[1, 2, 3, 5, 10], threshold=0.01):  # prediction_horizon = 5
        """Creates features and target variable for machine learning."""

        if df.empty:
            print("DataFrame is empty.  Returning empty DataFrame.")
            return df

        print(f"Initial DataFrame length: {len(df)}")  # Debugging

        # Convert close, volume, high, and low to float before creating lag features
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)

        # 1. Lagged Features
        for lag in lag_days:
            print(f"Creating lag {lag} features...")  # Progress message
            df[[col + f'_lag{lag}' for col in ['open', 'high', 'low', 'close', 'volume',
                                                'sma_20', 'rsi_14', 'macd_line', 'atr_14']]] = \
                df.groupby('trading_symbol')[['open', 'high', 'low', 'close', 'volume',
                                                'sma_20', 'rsi_14', 'macd_line', 'atr_14']].shift(lag)

        print(f"DataFrame length after lagging: {len(df)}")  # Debugging

        # 2. Derived Features
        print("Creating derived features...")  # Progress message
        df['prev_close'] = df.groupby('trading_symbol')['close'].shift(1)
        df['prev_volume'] = df.groupby('trading_symbol')['volume'].shift(1)

        # Convert prev_close and prev_volume to float after creating them
        df['prev_close'] = df['prev_close'].astype(float)
        df['prev_volume'] = df['prev_volume'].astype(float)

        df['price_change'] = (df['close'] - df['prev_close']) / (df['prev_close'] + 1e-9)  # Added small constant
        df['volume_change'] = (df['volume'] - df['prev_volume']) / (df['prev_volume'] + 1e-9)  # Added small constant
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['close']

        # 3. Target Variable
        print("Creating target variable...")  # Progress message
        df['future_close'] = df.groupby('trading_symbol')['close'].shift(-prediction_horizon)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        df['target'] = np.where(df['future_return'] > threshold, 1, 0)

        print(f"DataFrame length before dropping future_close NaN: {len(df)}")  # Debugging
        print(f"Number of NaN values before dropping future_close NaN:\n{df.isnull().sum()}")  # Debugging

        # Drop rows where future_close is NaN (we can't train without a target)
        print("Dropping rows with NaN in future_close...")  # Progress message
        df = df.dropna(subset=['future_close'])

        print(f"DataFrame length after dropping future_close NaN: {len(df)}")  # Debugging

        # Remove inf values
        print("Removing infinite values...")  # Progress message
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        print(f"DataFrame length after dropping Inf and NaN: {len(df)}")  # Debugging

        return df

    def get_trading_symbols(self, start_date, end_date):
        """
        Retrieves a list of distinct trading symbols available within the specified date range.
        """
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()

            query = """
                SELECT DISTINCT trading_symbol
                FROM historical_data
                WHERE date BETWEEN %s AND %s
            """
            cursor.execute(query, (start_date, end_date))
            symbols = [row[0] for row in cursor.fetchall()]

            conn.close()
            return symbols

        except Exception as e:
            print(f"Error fetching trading symbols: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def store_features(self, df, table_name='ml_features', batch_size=10000):
        """Stores the generated features in a MySQL table using batch insertion, appending only new data."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()

            # 1. Create the table if it doesn't exist
            # (Adjust data types as needed)
            columns = ", ".join([f"{col} FLOAT" for col in df.columns if col not in ('date', 'trading_symbol')])
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date DATE,
                    trading_symbol VARCHAR(255),
                    {columns},
                    PRIMARY KEY (date, trading_symbol)
                )
            """
            cursor.execute(create_table_query)

            # 2. Fetch existing data from ml_features table
            existing_data_query = f"""
                SELECT date, trading_symbol
                FROM {table_name}
            """
            cursor.execute(existing_data_query)
            existing_data = set((row[0], row[1]) for row in cursor.fetchall())

            # 3. Prepare the INSERT query
            columns = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
            """

            # 4. Iterate over the DataFrame in batches and filter out existing data
            total_rows = len(df)
            new_data_to_insert = []
            for i in tqdm(range(0, total_rows, batch_size), desc="Filtering and preparing new data"):
                batch_df = df[i:i + batch_size]
                for _, row in batch_df.iterrows():
                    date = row['date']
                    trading_symbol = row['trading_symbol']
                    if (date, trading_symbol) not in existing_data:
                        new_data_to_insert.append(tuple(row))

            # 5. Execute the INSERT query for the new data in batches
            total_new_rows = len(new_data_to_insert)
            for i in tqdm(range(0, total_new_rows, batch_size), desc="Storing new features in batches"):
                batch_data = new_data_to_insert[i:i + batch_size]
                cursor.executemany(insert_query, batch_data)
                conn.commit()  # Commit after each batch

            print(f"Successfully stored {total_new_rows} new rows in {table_name}")

            conn.close()

        except Exception as e:
            print(f"Error storing features in MySQL: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()

# Example Usage
if __name__ == '__main__':
    db_config = {
        'host': 'localhost',
        'user': 'dhan_hq',
        'password': 'Passw0rd@098',
        'database': 'dhanhq_db',
        'auth_plugin': 'mysql_native_password',
        'use_pure': True
    }

    feature_engineer = FeatureEngineering(db_config)

    # Option 1: Use UTC
    # end_date = datetime.utcnow().date()

    # Option 2: Delay by one day
    # end_date = (datetime.utcnow() - timedelta(days=1)).date()

    # Option 3: Convert to IST and truncate (be careful about data availability)
    ist = pytz.timezone('Asia/Kolkata')
    now_utc = datetime.utcnow()
    now_ist = now_utc.replace(tzinfo=pytz.utc).astimezone(ist)
    end_date = now_ist.date()

    start_date = '2018-01-01'

    # Fetch all trading symbols
    trading_symbols = feature_engineer.get_trading_symbols(start_date, end_date)

    # Fetch data for all symbols
    df = feature_engineer.fetch_data(trading_symbols, start_date, end_date)

    # Create features
    if not df.empty:
        df_featured = feature_engineer.create_features(df.copy()) # Pass a copy to avoid modifying the original DataFrame
        print(df_featured.head())

        # Store features in MySQL
        feature_engineer.store_features(df_featured)
        print("Features stored in ml_features table")
    else:
        print("No data fetched.  Check your database connection and date range.")