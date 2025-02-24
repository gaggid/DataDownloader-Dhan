import mysql.connector

def update_table_structure():
    # Connect to the database
    conn = mysql.connector.connect(
        host='localhost',
        user='dhan_hq',
        password='Password@098',
        database='dhanhq_db',
        auth_plugin='mysql_native_password',
        use_pure=True
    )
    cursor = conn.cursor()

    # Check if the custom indicator columns already exist
    cursor.execute("""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dhanhq_db'
          AND TABLE_NAME = 'technical_indicators'
          AND COLUMN_NAME IN ('custom_sma_20', 'custom_ema_20', 'custom_ema_50');
    """)
    existing_columns = {row[0] for row in cursor.fetchall()}

    # Build ALTER TABLE statements for missing columns
    alter_statements = []
    if 'custom_sma_20' not in existing_columns:
        alter_statements.append("ADD COLUMN custom_sma_20 DECIMAL(10,2) NULL")
    if 'custom_ema_20' not in existing_columns:
        alter_statements.append("ADD COLUMN custom_ema_20 DECIMAL(10,2) NULL")
    if 'custom_ema_50' not in existing_columns:
        alter_statements.append("ADD COLUMN custom_ema_50 DECIMAL(10,2) NULL")

    if alter_statements:
        alter_query = "ALTER TABLE technical_indicators " + ", ".join(alter_statements) + ";"
        cursor.execute(alter_query)
        conn.commit()
        print("Table structure updated with new custom indicator columns.")
    else:
        print("Custom indicator columns already exist.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    update_table_structure()
