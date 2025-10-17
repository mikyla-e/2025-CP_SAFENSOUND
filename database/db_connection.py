import sqlite3
import os
from datetime import datetime

class Database:
    def __init__(self, db_name="safensound.db"):
        self.db_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_name)
        print(f"Using database at: {self.db_name}")
        self.conn = sqlite3.connect(self.db_name)
        self.create_room()
        self.create_history()
        self.initialize_rooms()

    # create tables
    def create_room(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS room (
                    room_id INTEGER PRIMARY KEY CHECK (room_id IN (1, 2, 3)),
                    room_name TEXT NOT NULL
                )
            ''')

    def create_history(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    date DATE,
                    time TIME,
                    room_id INTEGER,
                    FOREIGN KEY (room_id) REFERENCES room (room_id),
                    CHECK (action IN ('Alert Acknowledged', 'Emergency Detected'))
                )
            ''')

    def initialize_rooms(self):
        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM room')
            count = cursor.fetchone()[0]
            
            if count == 0:
                default_rooms = [
                    (1, "Room 1"),
                    (2, "Room 2"), 
                    (3, "Room 3")
                ]
                self.conn.executemany('INSERT INTO room (room_id, room_name) VALUES (?, ?)', default_rooms)
                self.conn.commit()
        except Exception as e:
            print(f"Error initializing rooms: {e}")

    #insert data
    def insert_history(self, action, date, time, room_id):
        date = datetime.now().strftime("%Y-%m-%d")

        with self.conn:
            self.conn.execute('''
                INSERT INTO history (action, date, time, room_id)
                VALUES (?, ?, ?, ?)
            ''', (action, date, time, room_id))


    #update data
    def update_room(self, room_id, new_name):
        with self.conn:
            self.conn.execute('''
                UPDATE room
                SET room_name = ?
                WHERE room_id = ?
            ''', (new_name, room_id))

    #fetch and display data
    def fetch_rooms(self):
        with self.conn:
            cursor = self.conn.execute('SELECT * FROM room')
            return cursor.fetchall()
        
    def fetch_history(self, room_id):
        with self.conn:
            cursor = self.conn.execute('SELECT * FROM history where room_id = ?', (room_id,))
            return cursor.fetchall()


    def close(self):
        self.conn.close()