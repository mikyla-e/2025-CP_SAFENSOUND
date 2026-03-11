import sqlite3
from datetime import datetime

DB_PATH = "safensound.db"

ALERT_ACTIONS = {
    "1": ("emergency",   "Emergency Alert Detected"),
    "2": ("alarming",    "Alarming Alert Detected"),
    "3": ("acknowledge", "Alert Acknowledged"),
}


def get_rooms(db_path: str = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT room_id, room_name FROM room ORDER BY room_name")
        return [{"room_id": row[0], "room_name": row[1]} for row in cur.fetchall()]
    finally:
        conn.close()


def create_history(action, sound_type, room_id, recording_path=None, db_path=DB_PATH):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%I:%M:%S %p")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO history (action, sound_type, date, time, room_id, recording_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action, sound_type, date, time, room_id, recording_path),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def main():
    print("\n=== Safe & Sound — Create History Record ===\n")

    # --- Choose alert type ---
    print("Alert Type:")
    for key, (_, label) in ALERT_ACTIONS.items():
        print(f"  [{key}] {label}")

    while True:
        choice = input("\nChoose alert type (1/2/3): ").strip()
        if choice in ALERT_ACTIONS:
            _, action = ALERT_ACTIONS[choice]
            break
        print("  Invalid choice. Please enter 1, 2, or 3.")

    # --- Choose room ---
    rooms = get_rooms()
    print("\nRooms:")
    for i, room in enumerate(rooms, start=1):
        print(f"  [{i}] {room['room_name']}")

    while True:
        choice = input("\nChoose room number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(rooms):
            selected_room = rooms[int(choice) - 1]
            break
        print(f"  Invalid choice. Please enter a number between 1 and {len(rooms)}.")

    # --- Optional: sound type ---
    sound_type = input("\nSound type (press Enter to skip): ").strip()

    # --- Optional: recording path ---
    recording_path = input("Recording path (press Enter to skip): ").strip()
    if not recording_path:
        recording_path = None

    # --- Save ---
    new_id = create_history(
        action=action,
        sound_type=sound_type,
        room_id=selected_room["room_id"],
        recording_path=recording_path,
    )

    print(f"\n[✓] Record saved!")
    print(f"    history_id   : {new_id}")
    print(f"    action       : {action}")
    print(f"    room         : {selected_room['room_name']}")
    print(f"    sound_type   : {sound_type or '—'}")
    print(f"    recording    : {recording_path or '—'}")
    print()


if __name__ == "__main__":
    main()