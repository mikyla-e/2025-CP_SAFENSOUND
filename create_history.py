import requests
import sqlite3

DB_PATH = "safensound.db"
API_URL = "http://localhost:47845/api/alert"

ALERT_ACTIONS = {
    "1": "Emergency Alert Detected",
    "2": "Alarming Alert Detected",
    "3": "Alert Acknowledged",
}

def get_rooms(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT room_id, room_name FROM room ORDER BY room_name")
        return [{"room_id": row[0], "room_name": row[1]} for row in cur.fetchall()]
    finally:
        conn.close()

def main():
    print("\n=== Safe & Sound — Test Alert ===\n")

    print("Alert Type:")
    for key, label in ALERT_ACTIONS.items():
        print(f"  [{key}] {label}")
    while True:
        choice = input("\nChoose alert type (1/2/3): ").strip()
        if choice in ALERT_ACTIONS:
            action = ALERT_ACTIONS[choice]
            break
        print("  Invalid choice.")

    rooms = get_rooms()
    print("\nRooms:")
    for i, room in enumerate(rooms, start=1):
        print(f"  [{i}] {room['room_name']}")
    while True:
        choice = input("\nChoose room number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(rooms):
            selected_room = rooms[int(choice) - 1]
            break
        print(f"  Invalid. Enter 1–{len(rooms)}.")

    recording_path = input("\nRecording path (press Enter to skip): ").strip() or None

    payload = {
        "room_id": selected_room["room_id"],
        "action":  action,
    }
    if recording_path:
        payload["recording_path"] = recording_path

    print(f"\nSending POST to {API_URL}...")
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        if resp.ok:
            print(f"\n[✓] Alert sent successfully!")
            print(f"    action    : {action}")
            print(f"    room      : {selected_room['room_name']} (id={selected_room['room_id']})")
            print(f"    recording : {recording_path or '—'}")
            print(f"    response  : {resp.json()}")
        else:
            print(f"\n[✗] Server returned {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"\n[✗] Could not connect to {API_URL}")
        print("    Is safensound_app.py running?")

if __name__ == "__main__":
    main()