from flask import Flask, render_template, request, jsonify
import os
import sys
from threading import Lock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database

app = Flask(__name__, template_folder='templates', static_folder='static')

db_lock = Lock()
db = Database()

room_status = {1: 0, 2: 0, 3: 0}  # 0: Normal, 1: Emergency

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/rooms', methods=['GET'])
def get_rooms():
    try:
        with db_lock:
            rooms = db.fetch_rooms()

        rooms_data = {}
        for room_id, room_name in rooms:
            rooms_data[room_id] = {
                "name": room_name,
                "status": room_status.get(room_id, 0)
            }
        return jsonify(rooms_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/history/<int:room_id>', methods=['GET'])
def get_room_history(room_id):
    try:
        with db_lock:
            history = db.fetch_history(room_id)

        history_data = []
        for record in history:
            history_data.append({
                "history_id": record[0],
                "action": record[1],
                "date": record[2],
                "time": record[3],
                "room_id": record[4]
            })
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/rooms/<int:room_id>/rename', methods=['POST'])
def rename_room(room_id):
    try:
        new_name = request.json.get('new_name')
        if not new_name:
            return jsonify({"error": "New name is required"}), 400

        with db_lock:
            db.update_room(room_id, new_name)

        return jsonify({"success": True, "message": "Room renamed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current room status for real-time updates"""
    return jsonify(room_status)

if __name__ == '__main__':
    print("🌐 Starting SafeNSound Web Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)