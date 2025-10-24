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
            cursor = self.conn.execute(
                'SELECT * FROM history WHERE room_id = ? ORDER BY date DESC, time DESC',
                (room_id,)
            )
            return cursor.fetchall()
        
    # monthly emergencies
    def fetch_monthly_emergencies(self, year):
        """Fetch emergency count per month for a specific year"""
        with self.conn:
            cursor = self.conn.execute('''
                SELECT 
                    strftime('%m', date) as month,
                    COUNT(*) as count
                FROM history
                WHERE action = 'Emergency Detected'
                AND strftime('%Y', date) = ?
                GROUP BY month
                ORDER BY month
            ''', (str(year),))
            results = cursor.fetchall()
            
            # Create dict with all 12 months, defaulting to 0
            monthly_data = {str(i).zfill(2): 0 for i in range(1, 13)}
            for month, count in results:
                monthly_data[month] = count
            
            return monthly_data
    
    def fetch_available_years(self):
        """Fetch all years that have emergency data"""
        with self.conn:
            cursor = self.conn.execute('''
                SELECT DISTINCT strftime('%Y', date) as year
                FROM history
                WHERE action = 'Emergency Detected'
                ORDER BY year DESC
            ''')
            return [row[0] for row in cursor.fetchall()]
        
    # top emegency count
    def fetch_top_emergencies(self, year=None, date_range=None, start_date=None, end_date=None):
        """Fetch total emergency count per room with flexible filtering"""
        from datetime import datetime, timedelta
        
        with self.conn:
            query = '''
                SELECT 
                    r.room_id,
                    r.room_name,
                    COUNT(CASE WHEN h.action = 'Emergency Detected' THEN 1 END) as count
                FROM room r
                LEFT JOIN history h ON r.room_id = h.room_id
            '''
            
            conditions = []
            params = []
            
            # Date range filter (primary filter - takes precedence over year)
            if date_range:
                today = datetime.now().date()
                
                if date_range == 'this_year':
                    conditions.append("(h.history_id IS NULL OR strftime('%Y', h.date) = ?)")
                    params.append(str(today.year))
                
                elif date_range == 'this_month':
                    conditions.append("(h.history_id IS NULL OR strftime('%Y-%m', h.date) = ?)")
                    params.append(today.strftime('%Y-%m'))
                
                elif date_range == 'last_30':
                    start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
                    conditions.append("(h.history_id IS NULL OR h.date >= ?)")
                    params.append(start)
                
                elif date_range == 'last_7':
                    start = (today - timedelta(days=7)).strftime('%Y-%m-%d')
                    conditions.append("(h.history_id IS NULL OR h.date >= ?)")
                    params.append(start)
                
                elif date_range == 'custom' and start_date and end_date:
                    conditions.append("(h.history_id IS NULL OR h.date BETWEEN ? AND ?)")
                    params.extend([start_date, end_date])
            
            # Year filter only applies if no date_range is specified
            elif year and year != 'all':
                conditions.append("(h.history_id IS NULL OR strftime('%Y', h.date) = ?)")
                params.append(str(year))
            
            # Add conditions to query
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += '''
                GROUP BY r.room_id, r.room_name
                ORDER BY count DESC
            '''
            
            cursor = self.conn.execute(query, params)
            results = cursor.fetchall()
            
            total = sum(row[2] for row in results)
            room_data = {}
            for room_id, room_name, count in results:
                room_data[room_id] = {
                    "name": room_name,
                    "count": count
                }
            
            return {
                "total": total,
                "rooms": room_data
            }

    def close(self):
        self.conn.close()