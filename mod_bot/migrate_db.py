import sqlite3

db_path = 'c:/Users/varsh/OneDrive/Desktop/mod_bot/instance/site.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if column exists first to avoid error if run multiple times
    cursor.execute("PRAGMA table_info(user)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'phone_number' not in columns:
        print("Adding phone_number column...")
        cursor.execute("ALTER TABLE user ADD COLUMN phone_number VARCHAR(20)")
        conn.commit()
        print("Migration successful: Added phone_number column.")
    else:
        print("Column phone_number already exists.")
        
except Exception as e:
    print(f"Migration failed: {e}")
finally:
    if 'conn' in locals():
        conn.close()
