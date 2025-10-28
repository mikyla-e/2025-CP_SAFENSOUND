# insert_alert.py
import sqlite3
from datetime import datetime

conn = sqlite3.connect("safensound.db")
now = datetime.now()
conn.execute(
    "INSERT INTO history (action, date, time, room_id) VALUES (?, ?, ?, ?)",
    ("Alert Acknowledged", now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), 2)
)
conn.commit()
conn.close()
print("Inserted Alert Acknowledged for Room 1")