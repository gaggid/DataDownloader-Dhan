import mysql.connector

try:
    db_config = {
        'host': 'localhost',
        'user': 'dhan_hq',
        'password': 'Passw0rd@098',
        'database': 'dhanhq_db',
        'auth_plugin': 'mysql_native_password',
        'use_pure': True
    }

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    print(f"Result: {result}")

    conn.close()
    print("Connection closed successfully")

except Exception as e:
    print(f"Error: {e}")