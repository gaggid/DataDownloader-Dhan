import mysql.connector
from mysql.connector import Error
from datetime import datetime

DB_CONFIG = {
'host': 'localhost',
'user': 'dhan_hq',
'password': 'Passw0rd@098',
'database': 'dhanhq_db',
'auth_plugin': 'mysql_native_password',
'use_pure': True
}

def log_message(message, level="INFO"):
    """Log messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def clean_database_data():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # 1. Find duplicate dates PER SYMBOL
        duplicate_dates_query = """
        WITH DuplicateDates AS (
            SELECT 
                trading_symbol,
                date,
                COUNT(*) as count,
                GROUP_CONCAT(id ORDER BY id) as duplicate_ids
            FROM historical_data
            GROUP BY trading_symbol, date
            HAVING COUNT(*) > 1
        )
        SELECT 
            trading_symbol,
            date,
            count,
            duplicate_ids
        FROM DuplicateDates
        ORDER BY trading_symbol, date
        """
        
        cursor.execute(duplicate_dates_query)
        duplicates = cursor.fetchall()

        if duplicates:
            log_message(f"Found duplicates in {len(duplicates)} symbol-date combinations:", "WARNING")
            for dup in duplicates[:5]:  # Show first 5 examples
                log_message(f"Symbol: {dup[0]}, Date: {dup[1]}, Count: {dup[2]}", "INFO")
            if len(duplicates) > 5:
                log_message(f"... and {len(duplicates)-5} more", "INFO")
            
            # Create temporary table with correct data (keeping only the latest entry for each symbol-date combination)
            cursor.execute("""
                CREATE TEMPORARY TABLE temp_historical_data AS
                SELECT h.*
                FROM historical_data h
                INNER JOIN (
                    SELECT trading_symbol, date, MAX(id) as max_id
                    FROM historical_data
                    GROUP BY trading_symbol, date
                ) latest 
                ON h.id = latest.max_id
                ORDER BY h.trading_symbol, h.date
            """)

            # Get counts before deletion
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            count_before = cursor.fetchone()[0]

            # Delete all data from original table
            cursor.execute("DELETE FROM historical_data")

            # Insert clean data back
            cursor.execute("""
                INSERT INTO historical_data
                SELECT * FROM temp_historical_data
            """)

            # Get counts after cleaning
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            count_after = cursor.fetchone()[0]

            # Drop temporary table
            cursor.execute("DROP TEMPORARY TABLE temp_historical_data")
            
            log_message(f"Cleaned up duplicates: Removed {count_before - count_after} duplicate records", "INFO")

            # Add unique constraint if not exists
            try:
                cursor.execute("""
                    ALTER TABLE historical_data
                    ADD CONSTRAINT unique_symbol_date UNIQUE (trading_symbol, date)
                """)
                log_message("Added unique constraint for trading_symbol and date", "INFO")
            except Error as e:
                if "Duplicate entry" not in str(e):
                    raise e

        else:
            log_message("No duplicate symbol-date combinations found", "INFO")

        # After cleaning duplicates, identify and remove symbols with fewer than 365 days of data
        log_message("Identifying symbols with less than 365 days of data...", "INFO")
        cursor.execute("""
            SELECT 
                trading_symbol,
                COUNT(*) as record_count
            FROM historical_data
            GROUP BY trading_symbol
            HAVING record_count < 365
        """)
        
        short_history_symbols = cursor.fetchall()
        
        if short_history_symbols:
            symbol_count = len(short_history_symbols)
            log_message(f"Found {symbol_count} symbols with less than 365 days of data", "WARNING")
            
            # Display some examples
            for symbol in short_history_symbols[:5]:
                log_message(f"Symbol: {symbol[0]}, Records: {symbol[1]}", "INFO")
            if symbol_count > 5:
                log_message(f"... and {symbol_count-5} more", "INFO")
            
            # Get count before deletion
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            total_before = cursor.fetchone()[0]
            
            # Delete records for symbols with insufficient data
            symbol_list = [symbol[0] for symbol in short_history_symbols]
            format_strings = ','.join(['%s'] * len(symbol_list))
            delete_query = f"""
                DELETE FROM historical_data 
                WHERE trading_symbol IN ({format_strings})
            """
            cursor.execute(delete_query, symbol_list)
            
            # Get count after deletion
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            total_after = cursor.fetchone()[0]
            
            deleted_count = total_before - total_after
            log_message(f"Removed {deleted_count} records for {symbol_count} symbols with insufficient history", "INFO")
        else:
            log_message("No symbols found with less than 365 days of data", "INFO")

        # Commit changes
        connection.commit()

        # Print final summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT trading_symbol) as total_symbols,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM historical_data
        """)
        stats = cursor.fetchone()
        
        log_message("\nDatabase Summary:", "INFO")
        log_message(f"Total records: {stats[0]:,}", "INFO")
        log_message(f"Total symbols: {stats[1]}", "INFO")
        log_message(f"Date range: {stats[2]} to {stats[3]}", "INFO")

    except Error as e:
        log_message(f"Database error: {str(e)}", "ERROR")
        if connection.is_connected():
            connection.rollback()

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            log_message("Database connection closed", "INFO")

def verify_data_integrity():
    """Verify data integrity after cleaning."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Check for any remaining duplicates per symbol
        cursor.execute("""
            SELECT 
                trading_symbol,
                date,
                COUNT(*) as count
            FROM historical_data
            GROUP BY trading_symbol, date
            HAVING COUNT(*) > 1
            ORDER BY trading_symbol, date
        """)
        
        remaining_duplicates = cursor.fetchall()
        if remaining_duplicates:
            log_message(f"Warning: Found {len(remaining_duplicates)} remaining duplicate symbol-date combinations", "WARNING")
            for dup in remaining_duplicates[:5]:
                log_message(f"Symbol: {dup[0]}, Date: {dup[1]}, Count: {dup[2]}", "WARNING")
            return False

        # Additional integrity checks
        cursor.execute("""
            SELECT 
                trading_symbol,
                COUNT(*) as record_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM historical_data
            GROUP BY trading_symbol
            HAVING record_count < 365
        """)
        
        short_history = cursor.fetchall()
        if short_history:
            log_message(f"\nSymbols with less than 365 days of data:", "WARNING")
            for record in short_history:
                log_message(f"Symbol: {record[0]}, Records: {record[1]}, Range: {record[2]} to {record[3]}", "WARNING")

        log_message("Data integrity verification completed", "INFO")
        return True

    except Error as e:
        log_message(f"Verification error: {str(e)}", "ERROR")
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    log_message("Starting database cleaning process...")
    clean_database_data()
    verify_data_integrity()