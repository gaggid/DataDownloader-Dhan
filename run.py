"""
    A class to interact with the DhanHQ APIs.

    This library provides methods to manage orders, retrieve market data,
    and perform various trading operations through the DhanHQ API.

    :copyright: (c) 2024 by Dhan.
    :license: see LICENSE for details...
"""


import logging
import requests
import pandas as pd
import os
import json
import time  # Add this import
from json import loads as json_loads, dumps as json_dumps
from pathlib import Path
from webbrowser import open as web_open
from datetime import datetime, timedelta, timezone
import mysql.connector
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from threading import Lock
import yfinance as yf


class RateLimiter:
    """
    A thread-safe rate limiter to ensure API request limits are respected.
    """
    def __init__(self, calls_per_second=0.25):  # Change this to 0.25 (1 call per 4 seconds)
        self.calls_per_second = calls_per_second
        self.last_call_time = time.time()
        self.lock = Lock()

    def wait(self):
        """
        Waits if necessary to ensure the minimum time between API calls.
        """
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            time_to_wait = (1.0 / self.calls_per_second) - time_since_last_call
            
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            
            self.last_call_time = time.time()

class dhanhq:
    """DhanHQ Class to interact with REST APIs"""

    # Constants for Exchange Segment
    NSE = 'NSE_EQ'
    BSE = 'BSE_EQ'
    CUR = 'NSE_CURRENCY'
    MCX = 'MCX_COMM'
    FNO = 'NSE_FNO'
    NSE_FNO = 'NSE_FNO'
    BSE_FNO = 'BSE_FNO'
    INDEX = 'IDX_I'

    # Constants for Transaction Type
    BUY = 'BUY'
    SELL = 'SELL'

    # Constants for Product Type
    CNC = 'CNC'
    INTRA = "INTRADAY"
    MARGIN = 'MARGIN'
    CO = 'CO'
    BO = 'BO'
    MTF = 'MTF'

    # Constants for Order Type
    LIMIT = 'LIMIT'
    MARKET = 'MARKET'
    SL = "STOP_LOSS"
    SLM = "STOP_LOSS_MARKET"

    # Constants for Validity
    DAY = 'DAY'
    IOC = 'IOC'

    # CSV URL for Security ID List
    COMPACT_CSV_URL = 'https://images.dhan.co/api-data/api-scrip-master.csv'
    DETAILED_CSV_URL = 'https://images.dhan.co/api-data/api-scrip-master-detailed.csv'

    # Default client ID and API key (access token)
    client_id = "1000363258"
    access_token = (
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9."
        "eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQxNzAyMTIyLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTAwMDM2MzI1OCJ9."
        "Rj23-14hZFPwa_VzhagzwXys7YywccUhZZTGDvS-lYbrDEynFgIRjtQue8mICp4joTWeWvlgHkAmrFxmuMNCgw"
    )

    def __init__(self, client_id, access_token, disable_ssl=False, pool=None, db_password=None):
        """
        Initialize the dhanhq class with client ID, access token, and database credentials.
        
        Args:
            client_id (str): The client ID for the trading account
            access_token (str): The access token for API authentication
            disable_ssl (bool): Flag to disable SSL verification
            pool (dict): Optional connection pool settings
            db_password (str): MySQL database password
        """
        try:
            print("[INFO] Initializing DhanHQ client...", flush=True)
            self.client_id = str(client_id)
            self.access_token = access_token
            self.base_url = 'https://api.dhan.co/v2'
            self.timeout = 60
            self.header = {
                'access-token': access_token,
                'Content-type': 'application/json',
                'Accept': 'application/json'
            }
            self.disable_ssl = disable_ssl
            self.db_password = db_password
            
            # Database configuration
            self.db_config = {
                'host': 'localhost',
                'user': 'dhan_hq',
                'password': self.db_password,
                'database': 'dhanhq_db',
                'auth_plugin': 'mysql_native_password',
                'use_pure': True
            }
            
            requests.packages.urllib3.util.connection.HAS_IPV6 = False
            self.session = requests.Session()
            if pool:
                reqadapter = requests.adapters.HTTPAdapter(**pool)
                self.session.mount("https://", reqadapter)
            print("[INFO] DhanHQ client initialized successfully.", flush=True)

            # Add rate limiter
            self.rate_limiter = RateLimiter(calls_per_second=1)
        except Exception as e:
            logging.error('Exception in dhanhq>>__init__: %s', e)
            print(f"[ERROR] Exception during initialization: {e}", flush=True)
    
    def get_existing_symbols(self):
        """
        Retrieve list of existing trading symbols from historical_data table
        """
        try:
            connection = self.get_db_connection()
            if connection:
                cursor = connection.cursor()
                cursor.execute("SELECT DISTINCT trading_symbol FROM historical_data")
                symbols = [row[0] for row in cursor.fetchall()]
                cursor.close()
                connection.close()
                return symbols
        except Error as e:
            print(f"[ERROR] Error retrieving existing symbols: {e}", flush=True)
            return []

    def wait_for_input_with_timeout(self, prompt, timeout, default):
        """
        Wait for user input with a timeout and default value
        
        Args:
            prompt (str): The prompt to display to the user
            timeout (int): Number of seconds to wait for input
            default (str): Default value if no input is received
            
        Returns:
            str: User input or default value
        """
        import threading
        import time
        
        print(f"{prompt} (Default '{default}' in {timeout} seconds): ", end='', flush=True)
        
        answer = []
        def get_input():
            try:
                answer.append(input().strip().lower())
            except:
                pass
        
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Wait for specified timeout
        input_thread.join(timeout)
        
        if answer:
            return answer[0]
        else:
            print(f"\nUsing default value: {default}", flush=True)
            return default
    
    def get_market_cap(self, symbol):
        """
        Fetch market cap for a given symbol using yfinance.
        Tries both .NS and -SM.NS suffixes for NSE symbols.
        Skips symbols with market cap less than 500M.

        Args:
            symbol (str): Trading symbol of the security
            
        Returns:
            float or None: Market capitalization value if available and above threshold, None otherwise
        """
        MARKET_CAP_THRESHOLD = 500000000  # 500M threshold

        try:
            # First try with .NS suffix
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            market_cap = info.get('marketCap')
            
            # If market cap not found, try with -SM.NS suffix
            if market_cap is None:
                print(f"[INFO] Trying alternate symbol format for {symbol}", flush=True)
                yf_symbol = f"{symbol}-SM.NS"
                stock = yf.Ticker(yf_symbol)
                info = stock.info
                market_cap = info.get('marketCap')
                
            if market_cap is not None:
                # Check if market cap is below threshold
                if market_cap < MARKET_CAP_THRESHOLD:
                    print(f"[INFO] Skipping {symbol} - Market cap ({market_cap:,}) below threshold ({MARKET_CAP_THRESHOLD:,})", flush=True)
                    return None
                else:
                    print(f"[INFO] Successfully fetched market cap for {symbol} using {yf_symbol}: {market_cap:,}", flush=True)
                    return market_cap
            else:
                print(f"[WARNING] Market cap not available for {symbol} with either suffix", flush=True)
                return None
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch market cap for {symbol}: {e}", flush=True)
            return None
    
    def _process_volume_filter_for_symbol(self, trading_symbol, connection_pool, volume_threshold):
        """
        Process volume filter for a single symbol using a connection from the pool
        """
        try:
            connection = connection_pool.get()
            cursor = connection.cursor()

            try:
                print(f"\n[INFO] Processing volume filter for: {trading_symbol}", flush=True)

                # Get the last 20 days of data ordered by date
                cursor.execute("""
                    SELECT date, volume 
                    FROM historical_data 
                    WHERE trading_symbol = %s 
                    ORDER BY date DESC 
                    LIMIT 20
                """, (trading_symbol,))
                
                volume_data = cursor.fetchall()
                
                if len(volume_data) >= 20:
                    volumes = [row[1] for row in volume_data]
                    avg_volume = sum(volumes) / 20
                    
                    print(f"[INFO] {trading_symbol} 20-day average volume: {avg_volume:,.0f}", flush=True)
                    
                    if avg_volume < volume_threshold:
                        print(f"[INFO] Removing {trading_symbol} due to insufficient average volume", flush=True)
                        
                        # Delete all records for this symbol
                        cursor.execute("""
                            DELETE FROM historical_data 
                            WHERE trading_symbol = %s
                        """, (trading_symbol,))
                        
                        connection.commit()
                        print(f"[INFO] Removed all records for {trading_symbol}", flush=True)
                    else:
                        print(f"[INFO] {trading_symbol} meets volume criteria - keeping data", flush=True)
                else:
                    print(f"[WARNING] Insufficient data for {trading_symbol} - removing records", flush=True)
                    cursor.execute("""
                        DELETE FROM historical_data 
                        WHERE trading_symbol = %s
                    """, (trading_symbol,))
                    connection.commit()

            finally:
                cursor.close()
                connection_pool.put(connection)

        except Exception as e:
            print(f"[ERROR] Error processing volume filter for {trading_symbol}: {str(e)}", flush=True)

    def filter_low_volume_securities(self, max_workers=3):
        """
        Process the database to remove securities with insufficient average volume using multiple threads
        """
        VOLUME_THRESHOLD = 350000  # 500K threshold
        
        try:
            # Create a connection pool
            connection_pool = Queue()
            active_connections = []
            
            # Initialize connection pool
            for _ in range(max_workers):
                connection = self.get_db_connection()
                if connection:
                    connection_pool.put(connection)
                    active_connections.append(connection)

            # Get list of unique trading symbols
            initial_connection = self.get_db_connection()
            cursor = initial_connection.cursor()
            cursor.execute("SELECT DISTINCT trading_symbol FROM historical_data")
            symbols = [row[0] for row in cursor.fetchall()]
            cursor.close()
            initial_connection.close()

            total_symbols = len(symbols)
            print(f"[INFO] Processing volume filters for {total_symbols} symbols using {max_workers} workers", flush=True)

            # Process symbols with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for symbol in symbols:
                    future = executor.submit(
                        self._process_volume_filter_for_symbol,
                        symbol,
                        connection_pool,
                        VOLUME_THRESHOLD
                    )
                    futures.append(future)

                # Track progress
                completed = 0
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                        if completed % 10 == 0:
                            print(f"[INFO] Progress: {completed}/{total_symbols} symbols processed", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Thread execution failed: {str(e)}", flush=True)

        except Exception as e:
            print(f"[ERROR] Error in filter_low_volume_securities: {str(e)}", flush=True)
        finally:
            # Clean up connections
            print("[INFO] Cleaning up database connections...", flush=True)
            for conn in active_connections:
                try:
                    if conn and conn.is_connected():
                        conn.close()
                except Exception as e:
                    print(f"[WARNING] Error closing connection: {e}", flush=True)
    
    def get_db_connection(self):
        """
        Create and return a database connection
        """
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='dhan_hq',
                password=self.db_password,
                database='dhanhq_db',
                use_pure=True,
                auth_plugin='mysql_native_password'
            )
            if connection.is_connected():
                return connection
        except Error as e:
            print(f"[ERROR] Error while connecting to MySQL: {e}", flush=True)
            return None
    
    # Add this new method to the dhanhq class
    def _process_single_security(self, security_info, start_date, end_date, connection_pool):
        """
        Process a single security's historical data with only market cap filtering
        """
        try:
            security_id = str(int(security_info["SEM_SMST_SECURITY_ID"]))
            exchange_segment = f"{security_info['SEM_EXM_EXCH_ID']}_EQ"
            trading_symbol = security_info["SEM_TRADING_SYMBOL"].strip()
            company_name = security_info.get("SM_SYMBOL_NAME", "").strip()

            print(f"\n[INFO] Processing security: {trading_symbol} ({company_name})", flush=True)

            MARKET_CAP_THRESHOLD = 500000000  # 500M threshold

            # Fetch market cap and check if we should process this security
            market_cap = self.get_market_cap(trading_symbol)
            if market_cap is None or market_cap < MARKET_CAP_THRESHOLD:
                print(f"[INFO] Skipping {trading_symbol} due to insufficient market cap or missing data", flush=True)
                return

            # Get a connection from the pool
            connection = connection_pool.get()
            cursor = connection.cursor()

            try:
                # Check latest data available in database
                cursor.execute("""
                    SELECT MAX(date) as last_date 
                    FROM historical_data 
                    WHERE security_id = %s AND exchange = %s
                """, (security_id, exchange_segment))
                result = cursor.fetchone()
                last_date = result[0] if result[0] else None

                if last_date:
                    # Calculate start date for new data fetch
                    start_fetch_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    # Calculate and log the number of days we need to fetch
                    days_to_fetch = (datetime.strptime(end_date, "%Y-%m-%d").date() - last_date).days
                    print(f"[INFO] Found existing data up to {last_date}. Need to fetch {days_to_fetch} days of new data.", flush=True)
                    
                    # Check if we already have up-to-date data
                    if datetime.strptime(start_fetch_date, "%Y-%m-%d").date() >= datetime.strptime(end_date, "%Y-%m-%d").date():
                        print(f"[INFO] Data already up to date for {trading_symbol}", flush=True)
                        return
                        
                    # Validate that start date isn't after end date
                    if start_fetch_date > end_date:
                        print(f"[WARNING] Start date {start_fetch_date} is after end date {end_date}. Skipping {trading_symbol}", flush=True)
                        return
                        
                    # Check for future dates
                    if datetime.strptime(start_fetch_date, "%Y-%m-%d").date() > datetime.now().date():
                        print(f"[WARNING] Start date {start_fetch_date} is in the future. Adjusting to today's date.", flush=True)
                        start_fetch_date = datetime.now().date().strftime("%Y-%m-%d")
                else:
                    start_fetch_date = start_date

                print(f"[INFO] Fetching data from {start_fetch_date} to {end_date}", flush=True)

                # Apply rate limiting before making API call
                self.rate_limiter.wait()

                response = self.historical_daily_data(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument_type="EQUITY",
                    from_date=start_fetch_date,
                    to_date=end_date,
                    expiry_code=0
                )

                if response and response.get("status") == "success":
                    data = response.get("data", {})
                    if data and all(key in data for key in ["timestamp", "open", "high", "low", "close", "volume"]):
                        records = []
                        for i in range(len(data["timestamp"])):
                            record = {
                                "date": datetime.fromtimestamp(data["timestamp"][i]).strftime("%Y-%m-%d"),
                                "trading_symbol": trading_symbol,
                                "company_name": company_name,
                                "exchange": exchange_segment,
                                "security_id": security_id,
                                "open": data["open"][i],
                                "high": data["high"][i],
                                "low": data["low"][i],
                                "close": data["close"][i],
                                "volume": data["volume"][i],
                                "timestamp": datetime.fromtimestamp(data["timestamp"][i]).strftime("%Y-%m-%d %H:%M:%S"),
                                "market_cap": market_cap
                            }
                            records.append(record)

                        if records:
                            # Insert records in batches
                            batch_size = 1000
                            for i in range(0, len(records), batch_size):
                                batch = records[i:i + batch_size]
                                cursor.executemany("""
                                    INSERT INTO historical_data 
                                    (date, trading_symbol, company_name, exchange, 
                                    security_id, open, high, low, close, volume, 
                                    timestamp, market_cap)
                                    VALUES (%(date)s, %(trading_symbol)s, %(company_name)s, 
                                    %(exchange)s, %(security_id)s, %(open)s, %(high)s, 
                                    %(low)s, %(close)s, %(volume)s, %(timestamp)s, 
                                    %(market_cap)s)
                                """, batch)
                                connection.commit()

                            print(f"[INFO] Added {len(records)} new records for {trading_symbol}", flush=True)
                        else:
                            print(f"[INFO] No new data available for {trading_symbol}", flush=True)

            finally:
                cursor.close()
                connection_pool.put(connection)

        except Exception as e:
            print(f"[ERROR] Error processing {trading_symbol}: {str(e)}", flush=True)
    
    def save_to_mysql(self, df, table_name):
        """
        Save DataFrame to MySQL table
        """
        connection = None
        try:
            connection = self.get_db_connection()
            if connection:
                cursor = connection.cursor()
                
                # Convert DataFrame to list of tuples
                if table_name == 'fil_security_list':
                    for _, row in df.iterrows():
                        sql = """INSERT INTO fil_security_list 
                                (sem_exm_exch_id, sem_segment, sem_smst_security_id, 
                                sem_instrument_name, sem_trading_symbol, 
                                sem_exch_instrument_type, sm_symbol_name)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)"""
                        values = (
                            row['SEM_EXM_EXCH_ID'],
                            row.get('SEM_SEGMENT', ''),
                            row['SEM_SMST_SECURITY_ID'],
                            row['SEM_INSTRUMENT_NAME'],
                            row['SEM_TRADING_SYMBOL'],
                            row['SEM_EXCH_INSTRUMENT_TYPE'],
                            row.get('SM_SYMBOL_NAME', '')
                        )
                        cursor.execute(sql, values)
                
                elif table_name == 'historical_data':
                    for _, row in df.iterrows():
                        sql = """INSERT INTO historical_data 
                                (date, trading_symbol, company_name, exchange, 
                                security_id, open, high, low, close, volume, timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                        values = (
                            row['date'],
                            row['trading_symbol'],
                            row['company_name'],
                            row['exchange'],
                            row['security_id'],
                            row['open'],
                            row['high'],
                            row['low'],
                            row['close'],
                            row['volume'],
                            row['timestamp']
                        )
                        cursor.execute(sql, values)
                
                connection.commit()
                print(f"[INFO] Successfully saved data to {table_name}", flush=True)
                
        except Error as e:
            print(f"[ERROR] Error while saving to MySQL: {e}", flush=True)
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()
                print("[INFO] MySQL connection closed", flush=True)

    def _parse_response(self, response):
        """
        Parse the API response.

        Args:
            response (requests.Response): The response object from the API.

        Returns:
            dict: Parsed response containing status, remarks, and data.
        """
        try:
            status = 'failure'
            remarks = ''
            data = ''
            python_response = json_loads(response.content)
            if response.status_code == 200:
                status = 'success'
                data = python_response
            else:
                error_type = python_response.get('errorType')
                error_code = python_response.get('errorCode')
                error_message = python_response.get('errorMessage')
                remarks = {
                    'error_code': error_code,
                    'error_type': error_type,
                    'error_message': error_message
                }
                data = python_response
        except Exception as e:
            logging.warning('Exception in dhanhq>>_parse_response: %s', e)
            status = 'failure'
            remarks = str(e)
        return {
            'status': status,
            'remarks': remarks,
            'data': data,
        }

    def historical_daily_data(self, security_id, exchange_segment, instrument_type, from_date, to_date, expiry_code=0):
        """
        Retrieve OHLC & Volume of daily candle for desired instrument.

        Args:
            security_id (str): Security ID of the instrument.
            exchange_segment (str): The exchange segment (e.g., NSE, BSE).
            instrument_type (str): The type of instrument (e.g., stock, option).
            expiry_code (int): The expiry code for derivatives.
            from_date (str): The start date for the historical data.
            to_date (str): The end date for the historical data.

        Returns:
            dict: The response containing historical daily data.
        """
        try:
            url = self.base_url + f'/charts/historical'
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument_type,
                "expiryCode": expiry_code,
                "fromDate": from_date,
                "toDate": to_date
            }
            # Validate expiry_code value; it must be 0,1,2, or 3.
            if expiry_code in [0, 1, 2, 3]:
                payload['expiryCode'] = expiry_code
            else:
                raise Exception("expiry_code value must be one of [0, 1, 2, 3].")

            payload = json_dumps(payload)
            response = self.session.post(url, headers=self.header, timeout=self.timeout, data=payload)
            return self._parse_response(response)
        except Exception as e:
            logging.error('Exception in dhanhq>>historical_daily_data: %s', e)
            return {
                'status': 'failure',
                'remarks': str(e),
                'data': '',
            }

    def fetch_security_list(self, mode='compact', filename='security_id_list.csv'):
        """
        Fetch CSV file from DhanHQ based on the specified mode and return as a DataFrame.
        This method saves the original CSV as 'filename'.

        Args:
            mode (str): The mode to fetch the CSV ('compact' or 'detailed').
            filename (str): The name of the file to save the CSV as.

        Returns:
            pd.DataFrame: The DataFrame containing the security list.
        """
        try:
            print(f"[INFO] Fetching security list in '{mode}' mode...", flush=True)

            if mode == 'compact':
                csv_url = self.COMPACT_CSV_URL
            elif mode == 'detailed':
                csv_url = self.DETAILED_CSV_URL
            else:
                raise ValueError("Invalid mode. Choose 'compact' or 'detailed'.")

            response = requests.get(csv_url)
            response.raise_for_status()
            print("[INFO] Successfully downloaded security list.", flush=True)

            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Saved security list as '{os.path.abspath(filename)}'.", flush=True)

            # Load CSV into DataFrame
            df = pd.read_csv(filename)
            print(f"[INFO] CSV loaded into DataFrame with {len(df)} rows.", flush=True)
            return df

        except Exception as e:
            logging.error('Exception in dhanhq>>fetch_security_list: %s', e)
            print(f"[ERROR] Failed to fetch security list: {e}", flush=True)
            return None

    def fetch_and_save_historical_data(self, input_csv, start_date=None, end_date=None, max_workers=3):
        """
        Reads the filtered CSV file and retrieves historical daily data for all securities using multiple threads.
        Includes proper cleanup handling for interruptions.
        """
        active_connections = []  # Keep track of all database connections
        executor = None
        
        try:
            print("[INFO] Starting multithreaded historical data fetch...", flush=True)
            
            # Read and sort the data
            df = pd.read_csv(input_csv)
            df = df.sort_values(by='SEM_TRADING_SYMBOL', ascending=True)
            
            # Log the processing information
            print(f"[INFO] Processing {len(df)} securities in alphabetical order", flush=True)
            print("[INFO] First 5 securities to be processed:", flush=True)
            for idx, row in df.head().iterrows():
                print(f"  - {row['SEM_TRADING_SYMBOL']}", flush=True)
            
            # Set up date range
            if end_date is None:
                end_date = datetime.now().date() + timedelta(days=1)
            else:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                
            if start_date is None:
                start_date = datetime(2015, 1, 1).date()
            else:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            print(f"[INFO] Date range: {from_date} to {to_date}", flush=True)

            # Create connection pool
            connection_pool = Queue()
            for _ in range(max_workers):
                connection = self.get_db_connection()
                if connection:
                    connection_pool.put(connection)
                    active_connections.append(connection)  # Track connections for cleanup

            # Process securities with proper exception handling
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = []
            try:
                for _, row in df.iterrows():
                    future = executor.submit(
                        self._process_single_security,
                        security_info=row.to_dict(),
                        start_date=from_date,
                        end_date=to_date,
                        connection_pool=connection_pool
                    )
                    futures.append(future)

                # Track progress with interruption handling
                total_securities = len(futures)
                completed = 0
                
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                        if completed % 10 == 0:
                            print(f"[INFO] Progress: {completed}/{total_securities} securities processed", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Thread execution failed: {str(e)}", flush=True)

            except KeyboardInterrupt:
                print("\n[INFO] Received interruption signal. Cleaning up...", flush=True)
                # Cancel pending futures
                for future in futures:
                    future.cancel()
                raise  # Re-raise to handle cleanup in outer try block

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down gracefully...", flush=True)
        except Exception as e:
            print(f"[ERROR] Exception in fetch_and_save_historical_data: {str(e)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
        finally:
            # Proper cleanup in finally block
            print("[INFO] Performing cleanup...", flush=True)
            
            # Shutdown executor if it exists
            if executor:
                print("[INFO] Shutting down thread executor...", flush=True)
                executor.shutdown(wait=False)
            
            # Close all database connections
            print("[INFO] Closing database connections...", flush=True)
            for conn in active_connections:
                try:
                    if conn and conn.is_connected():
                        conn.close()
                except Exception as e:
                    print(f"[WARNING] Error closing connection: {e}", flush=True)
            
            print("[INFO] Cleanup completed", flush=True)

if __name__ == "__main__":
    # Get database password
    db_password = 'Passw0rd@098'
    
    # Instantiate the client with database password
    client_id = dhanhq.client_id
    access_token = dhanhq.access_token
    dhan = dhanhq(client_id, access_token, db_password=db_password)

    # Ask user about refreshing symbol list
    refresh_choice = dhan.wait_for_input_with_timeout(
        "Do you want to refresh the symbol list? (yes/no)",
        10,
        "no"
    )

    # Create tables if they don't exist
    connection = dhan.get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create fil_security_list table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fil_security_list (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sem_exm_exch_id VARCHAR(10),
                    sem_segment VARCHAR(20),
                    sem_smst_security_id VARCHAR(20),
                    sem_instrument_name VARCHAR(50),
                    sem_expiry_code INT,
                    sem_trading_symbol VARCHAR(50),
                    sem_lot_units INT,
                    sem_custom_symbol VARCHAR(50),
                    sem_expiry_date DATE,
                    sem_strike_price DECIMAL(10,2),
                    sem_option_type VARCHAR(10),
                    sem_tick_size DECIMAL(10,2),
                    sem_expiry_flag CHAR(1),
                    sem_exch_instrument_type VARCHAR(10),
                    sem_series VARCHAR(10),
                    sm_symbol_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create historical_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    trading_symbol VARCHAR(50),
                    company_name VARCHAR(100),
                    exchange VARCHAR(20),
                    security_id VARCHAR(20),
                    open DECIMAL(10,2),
                    high DECIMAL(10,2),
                    low DECIMAL(10,2),
                    close DECIMAL(10,2),
                    volume DECIMAL(15,2),
                    timestamp DATETIME,
                    market_cap DECIMAL(20,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            connection.commit()
            print("[INFO] Database tables created successfully", flush=True)
            
        except Error as e:
            print(f"[ERROR] Error creating tables: {e}", flush=True)
        finally:
            cursor.close()
            connection.close()

    filtered_filename = "filtered_security_list.csv"
    
    if refresh_choice == "yes":
        print("[INFO] Refreshing symbol list...", flush=True)
        # Fetch and filter security list
        df = dhan.fetch_security_list("compact")
        if df is not None:
            # Filter for equity securities
            df = df[
                (df['SEM_EXM_EXCH_ID'].isin(['BSEEE', 'NSE'])) &
                (df['SEM_INSTRUMENT_NAME'] == 'EQUITY') &
                (df['SEM_EXCH_INSTRUMENT_TYPE'] == 'ES')
            ]
            df.to_csv(filtered_filename, index=False)
    else:
        print("[INFO] Using existing symbols from database...", flush=True)
        # Get detailed symbol information from database
        connection = dhan.get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Fetch existing symbols with security_id and company_name
                cursor.execute("""
                    SELECT DISTINCT trading_symbol, security_id, company_name, exchange 
                    FROM historical_data
                """)
                
                existing_data = cursor.fetchall()
                
                if existing_data:
                    # Create DataFrame with full data from database
                    df = pd.DataFrame({
                        'SEM_TRADING_SYMBOL': [row[0] for row in existing_data],
                        'SEM_SMST_SECURITY_ID': [row[1] for row in existing_data],
                        'SM_SYMBOL_NAME': [row[2] for row in existing_data],
                        'SEM_EXM_EXCH_ID': [row[3].split('_')[0] for row in existing_data],
                        'SEM_INSTRUMENT_NAME': ['EQUITY'] * len(existing_data),
                        'SEM_EXCH_INSTRUMENT_TYPE': ['ES'] * len(existing_data)
                    })
                    df.to_csv(filtered_filename, index=False)
                    print(f"[INFO] Retrieved {len(existing_data)} existing securities from database", flush=True)
                else:
                    print("[ERROR] No existing symbols found in database", flush=True)
                    exit(1)
            except Error as e:
                print(f"[ERROR] Error retrieving data from database: {e}", flush=True)
                exit(1)
            finally:
                cursor.close()
                connection.close()

    # Get date range from user with timeout
    start_date = dhan.wait_for_input_with_timeout(
        "Enter start date (YYYY-MM-DD) or press enter for earliest available",
        10,  # 10 second timeout
        ""   # Empty string as default
    ).strip()

    end_date = dhan.wait_for_input_with_timeout(
        "Enter end date (YYYY-MM-DD) or press enter for today",
        10,  # 10 second timeout
        ""   # Empty string as default
    ).strip()

    start_date = start_date if start_date else None
    end_date = end_date if end_date else None

    # Modify the _process_single_security method to skip market cap check for existing symbols
    if refresh_choice == "no":
        # Store the original method
        original_process_method = dhan._process_single_security
        
        # Define the modified method
        def modified_process_method(security_info, start_date, end_date, connection_pool):
            """Modified version that skips market cap check for existing symbols"""
            try:
                security_id = str(security_info["SEM_SMST_SECURITY_ID"])
                exchange_segment = f"{security_info['SEM_EXM_EXCH_ID']}_EQ"
                trading_symbol = security_info["SEM_TRADING_SYMBOL"].strip()
                company_name = security_info.get("SM_SYMBOL_NAME", "").strip()

                print(f"\n[INFO] Processing security: {trading_symbol} ({company_name})", flush=True)

                # Skip market cap check - we trust existing symbols in database
                # Get a connection from the pool
                connection = connection_pool.get()
                cursor = connection.cursor()

                try:
                    # Check latest data available in database
                    cursor.execute("""
                        SELECT MAX(date) as last_date 
                        FROM historical_data 
                        WHERE trading_symbol = %s
                    """, (trading_symbol,))
                    result = cursor.fetchone()
                    last_date = result[0] if result[0] else None

                    if last_date:
                        # Calculate start date for new data fetch
                        start_fetch_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        # Calculate and log the number of days we need to fetch
                        days_to_fetch = (datetime.strptime(end_date, "%Y-%m-%d").date() - last_date).days
                        print(f"[INFO] Found existing data up to {last_date}. Need to fetch {days_to_fetch} days of new data.", flush=True)
                        
                        # Check if we already have up-to-date data
                        if datetime.strptime(start_fetch_date, "%Y-%m-%d").date() > datetime.strptime(end_date, "%Y-%m-%d").date():
                            print(f"[INFO] Data already up to date for {trading_symbol}", flush=True)
                            return
                            
                        # Validate that start date isn't after end date
                        if start_fetch_date > end_date:
                            print(f"[WARNING] Start date {start_fetch_date} is after end date {end_date}. Skipping {trading_symbol}", flush=True)
                            return
                            
                        # Check for future dates
                        if datetime.strptime(start_fetch_date, "%Y-%m-%d").date() > datetime.now().date():
                            print(f"[WARNING] Start date {start_fetch_date} is in the future. Adjusting to today's date.", flush=True)
                            start_fetch_date = datetime.now().date().strftime("%Y-%m-%d")
                    else:
                        start_fetch_date = start_date

                    print(f"[INFO] Fetching data from {start_fetch_date} to {end_date} for {trading_symbol}", flush=True)

                    # Apply rate limiting before making API call
                    dhan.rate_limiter.wait()

                    response = dhan.historical_daily_data(
                        security_id=security_id,
                        exchange_segment=exchange_segment,
                        instrument_type="EQUITY",
                        from_date=start_fetch_date,
                        to_date=end_date,
                        expiry_code=0
                    )

                    if response and response.get("status") == "success":
                        data = response.get("data", {})
                        if data and all(key in data for key in ["timestamp", "open", "high", "low", "close", "volume"]):
                            records = []
                            for i in range(len(data["timestamp"])):
                                record = {
                                    "date": datetime.fromtimestamp(data["timestamp"][i]).strftime("%Y-%m-%d"),
                                    "trading_symbol": trading_symbol,
                                    "company_name": company_name,
                                    "exchange": exchange_segment,
                                    "security_id": security_id,
                                    "open": data["open"][i],
                                    "high": data["high"][i],
                                    "low": data["low"][i],
                                    "close": data["close"][i],
                                    "volume": data["volume"][i],
                                    "timestamp": datetime.fromtimestamp(data["timestamp"][i]).strftime("%Y-%m-%d %H:%M:%S"),
                                    "market_cap": 0  # Default value since we're skipping market cap check
                                }
                                records.append(record)

                            if records:
                                # Insert records in batches
                                batch_size = 1000
                                for i in range(0, len(records), batch_size):
                                    batch = records[i:i + batch_size]
                                    cursor.executemany("""
                                        INSERT INTO historical_data 
                                        (date, trading_symbol, company_name, exchange, 
                                        security_id, open, high, low, close, volume, 
                                        timestamp, market_cap)
                                        VALUES (%(date)s, %(trading_symbol)s, %(company_name)s, 
                                        %(exchange)s, %(security_id)s, %(open)s, %(high)s, 
                                        %(low)s, %(close)s, %(volume)s, %(timestamp)s, 
                                        %(market_cap)s)
                                    """, batch)
                                    connection.commit()

                                print(f"[INFO] Added {len(records)} new records for {trading_symbol}", flush=True)
                            else:
                                print(f"[INFO] No new data available for {trading_symbol}", flush=True)
                        else:
                            print(f"[WARNING] Invalid data format received for {trading_symbol}", flush=True)
                    else:
                        error_msg = response.get("remarks", "Unknown error")
                        print(f"[ERROR] Failed to fetch data for {trading_symbol}: {error_msg}", flush=True)

                finally:
                    cursor.close()
                    connection_pool.put(connection)

            except Exception as e:
                print(f"[ERROR] Error processing {trading_symbol}: {str(e)}", flush=True)
        
        # Replace the original method with our modified version
        dhan._process_single_security = modified_process_method

    # Fetch historical data
    dhan.fetch_and_save_historical_data(filtered_filename, start_date, end_date, max_workers=6)
    
    # Apply volume filtering only if we refreshed the symbol list
    if refresh_choice == "yes":
        print("\n[INFO] Applying volume filters to downloaded data...", flush=True)
        dhan.filter_low_volume_securities(max_workers=10)
    
    print("\n[INFO] Data processing completed", flush=True)
