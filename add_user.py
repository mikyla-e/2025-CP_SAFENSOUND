import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))
from db_connection import Database

def main():
    db = Database()
    
    print("=== SafeNSound User Management ===")
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    if not username or not password:
        print("Error: Username and password cannot be empty")
        return
    
    try:
        db.add_user(username, password)
        print(f"✓ User '{username}' created successfully!")
    except Exception as e:
        print(f"✗ Error creating user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()