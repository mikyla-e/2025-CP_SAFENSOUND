from database.db_connection import Database
from datetime import datetime

def add_emergency_history():
    """Add test emergency history entries to the database"""
    db = Database()
    
    # Get current date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Fetch available rooms
    rooms = db.fetch_rooms()
    
    if not rooms:
        print("No rooms found in database. Please create rooms first.")
        db.close()
        return
    
    print("Available rooms:")
    for room in rooms:
        print(f"Room ID: {room[0]}, Name: {room[1]}")
    
    # Get room_id from user
    room_id = input("\nEnter room ID to add emergency to: ")
    
    try:
        room_id = int(room_id)
        
        # Verify room exists
        room = db.fetch_room(room_id)
        if not room:
            print(f"Room ID {room_id} not found.")
            db.close()
            return
        
        # Add emergency history entry
        db.insert_history(
            action='Emergency Alert Detected',
            sound_type='Emergency',
            date=current_date,
            time=current_time,
            room_id=room_id,
            recording_path=None  # Set to None or provide a path
        )
        
        print(f"\n✓ Emergency alert added successfully for {room[1]}")
        print(f"  Date: {current_date}")
        print(f"  Time: {current_time}")
        
    except ValueError:
        print("Invalid room ID. Please enter a number.")
    except Exception as e:
        print(f"Error adding emergency: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    add_emergency_history()