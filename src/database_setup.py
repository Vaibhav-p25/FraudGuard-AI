import sqlite3
import os

def setup_database():
    # This puts the database file inside the 'data' folder
    db_path = os.path.join('..', 'data', 'fraud_guard.db')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            home_lat REAL,
            home_lon REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount REAL,
            lat REAL,
            lon REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    # Dummy Data
    test_users = [
        (101, 'Aditya', 12.9716, 77.5946),
        (102, 'Rahul', 28.6139, 77.2090)
    ]
    cursor.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)", test_users)

    connection.commit()
    connection.close()
    print("Database created in /data/ folder!")

if __name__ == "__main__":
    setup_database()