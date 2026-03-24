"""Add sentiment_score column to chat_messages table."""

import sqlite3

# Connect to database
conn = sqlite3.connect('support.db')
cursor = conn.cursor()

try:
    # Add the sentiment_score column
    cursor.execute("ALTER TABLE chat_messages ADD COLUMN sentiment_score FLOAT;")
    conn.commit()
    print("✅ Successfully added sentiment_score column to chat_messages table")
    
    # Verify the column was added
    cursor.execute("PRAGMA table_info(chat_messages);")
    columns = cursor.fetchall()
    print("\nCurrent columns in chat_messages:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("⚠️  Column sentiment_score already exists")
    else:
        print(f"❌ Error: {e}")
finally:
    conn.close()
