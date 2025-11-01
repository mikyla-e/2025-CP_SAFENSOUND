# insert_alert.py
import sqlite3
from datetime import datetime

conn = sqlite3.connect("safensound.db")
now = datetime.now()
conn.execute(
    "INSERT INTO history (action, date, time, room_id) VALUES (?, ?, ?, ?)",
    ("Emergency Detected", now.strftime("%Y-%m-%d"), now.strftime("%I:%M:%S %p"), 1)
)
conn.commit()
conn.close()
print("Inserted Emergency Detected for Room 1")