from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import sys
import os
import asyncio
import json
import glob
import socket
from typing import List
from datetime import datetime
import uvicorn
import signal
import threading
import traceback
import secrets

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database
from datetime import datetime #for date format (separation)

static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
recordings_dir = os.path.join(os.path.dirname(__file__), "..", "recorded_audio")

class ShutdownRequest(BaseModel):
    target: str
    confirm: bool = False

rpi_ip = None

class RoomRename(BaseModel):
    new_name: str

class NewRoom(BaseModel):
    room_name: str

class DeviceRegister(BaseModel):
    address: str

class DeviceAssign(BaseModel):
    room_id: int

device_room_map: dict[str, int] = {}

class AlertData(BaseModel):
    room_id: int
    action: str
    sound_type: str = None
    recording_path: str = None

class AudioData(BaseModel):
    room_id: int
    timestamp: int
    sample_count: int
    audio_data: List[int]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected.")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()
room_status = {1:0, 2:0, 3:0}

web_port = 63429
stop_event = threading.Event()

class WebDiscoverServer:
    def __init__(self):
        self.running = True
        self.web_ip = self.get_web_ip()

    def get_web_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print("Error getting web IP:", e)
            return "localhost"
        
    def discovery_listener(self):
        global rpi_ip
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', web_port))

        print(f"Discovery server listening on {self.web_ip}:{web_port}")

        while self.running and not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8').strip()
                print(f"Received discovery message from {addr[0]}: {message}")

                if message == "SAFENSOUND RASPBERRY PI HERE":
                    rpi_ip = addr[0]
                    response = f"SAFENSOUND WEB DASHBOARD HERE: {self.web_ip}"
                    sock.sendto(response.encode('utf-8'), addr)
                    print(f"Sent response to {addr[0]}: {self.web_ip}")
            
            except Exception as e:
                if self.running:
                    print(f"Error in discovery listener: {e}")

        sock.close()

    def start(self):
        discovery_thread = threading.Thread(target=self.discovery_listener, daemon=True)
        discovery_thread.start()
        print(f"Discovery server started on {self.web_ip}:{web_port}.")

        return discovery_thread
    
    def stop(self):
        self.running = False
        print("Discovery listener stopped.")

web_discover_server = WebDiscoverServer()

async def periodic_updates():
    while True:
        try:
            await manager.broadcast({
                "type": "status_update",
                "status": room_status
            })
        except Exception as e:
            print(f"Error in periodic update: {e}")
        await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # web_discover_server.start()

    task = asyncio.create_task(periodic_updates())
    print("FastAPI SafeNSound started!")
    # print(f"Homepage available at: http://{web_discover_server.web_ip}:8080")
    print(f"Homepage available at: http://localhost:8080")
    print("API Documentation at: http://localhost:8080/docs")

    yield
    print("Shutting down...")
    # web_discover_server.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Periodic updates cancelled.")

app = FastAPI(title="SafeNSound Homepage", version="1.0", lifespan=lifespan)

# middleware
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.mount("/static", StaticFiles(directory=static_dir), name="static")  
templates = Jinja2Templates(directory=templates_dir)

db = Database()

# no zero starting hour
def strip_leading_zero_hour(t: str) -> str:
    try:
        return t[1:] if t and t.startswith('0') else t
    except Exception:
        return t

connected_websockets: List[WebSocket] = []

# Authentication dependency
def get_current_user(request: Request):
    """Check if user is logged in"""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If already logged in, redirect to dashboard
    if request.session.get("user"):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = db.verify_user(username, password)
    if user:
        request.session["user"] = {"user_id": user[0], "username": user[1]}
        return RedirectResponse(url="/dashboard", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password"
        })

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_alias(request: Request):
    # Check if user is logged in
    user = request.session.get("user")
    if not user:
        # Redirect to login page if not authenticated
        return RedirectResponse(url="/login", status_code=303)
    
    # If authenticated, show dashboard
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": user["username"]
    })

@app.get("/api/rooms")
async def get_rooms(user: dict = Depends(get_current_user)):
    try:
        rooms = db.fetch_rooms()
        devices = db.fetch_devices()
        
        # Create a map of room_id -> has_sensor
        rooms_with_sensors = set(device[2] for device in devices if device[2] != 0)
        
        # Separate and sort rooms
        assigned_rooms = []
        unassigned_rooms = []
        
        for room in rooms:
            room_id = room[0]
            room_name = room[1]
            room_data = {
                "id": room_id,
                "name": room_name,
                "status": room_status.get(room_id, 0),
                "has_sensor": room_id in rooms_with_sensors
            }
            
            if room_id in rooms_with_sensors:
                assigned_rooms.append(room_data)
            else:
                unassigned_rooms.append(room_data)
        
        # Sort unassigned rooms alphabetically by name
        unassigned_rooms.sort(key=lambda x: x["name"].lower())
        
        # Combine: assigned first, then unassigned (alphabetically)
        sorted_rooms = assigned_rooms + unassigned_rooms
        
        # Return as dictionary with room_id as key
        rooms_data = {}
        for room in sorted_rooms:
            rooms_data[room["id"]] = {
                "name": room["name"],
                "status": room["status"],
                "has_sensor": room["has_sensor"]
            }
        
        return rooms_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# MOST RECENT EMERGENCY
@app.get("/api/recent_emergency")
async def get_recent_emergency():
    try:
        query = """
            SELECT action, date, time, room_id
            FROM history
            WHERE action = 'Emergency Alert Detected' or action = 'Alarming Alert Detected'
            ORDER BY date DESC, time DESC
            LIMIT 1
        """
        cursor = db.conn.execute(query)
        row = cursor.fetchone()
        if row:
            formatted_date = datetime.strptime(row[1], "%Y-%m-%d").strftime("%b %d, %Y")
            return {
                "action": row[0],
                "date": formatted_date,
                "time": strip_leading_zero_hour(row[2]),
                "room_id": row[3]
            }
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HISTORY FOR A ROOM 
@app.get("/api/history/{room_id}")
async def get_history(room_id: int, user: dict = Depends(get_current_user)):
    try:
        history = db.fetch_history(room_id)
        formatted_history = []
        for record in history:
            # record structure: (history_id, action, sound_type, date, time, room_id, recording_path)
            history_id = record[0]
            action = record[1]
            sound_type = record[2]
            date_str = record[3]
            time_str = record[4]
            room_id_db = record[5]
            recording_path = record[6] if len(record) > 6 else None
            
            # Format date as MM/DD/YY
            formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m/%d/%y")
            has_recording = False

            if recording_path and os.path.exists(recording_path):
                has_recording = True
            
            formatted_history.append({
                "history_id": history_id,
                "action": action,
                "sound_type": sound_type,
                "date": formatted_date,
                "time": strip_leading_zero_hour(time_str),
                "has_recording": has_recording,
                "recording_path": recording_path        
            })
        return formatted_history
    except Exception as e:
        print(f"Error fetching history: {e}")  # Add logging
        raise HTTPException(status_code=500, detail=str(e))

# ROOM UPDATES
@app.post("/api/rooms")
async def create_room(data: NewRoom, user: dict = Depends(get_current_user)):
    try:
        db.insert_room(data.room_name)
        cursor = db.conn.execute('SELECT last_insert_rowid()')
        room_id = cursor.fetchone()[0]

        room_status[room_id] = room_status.get(room_id, 0)

        await manager.broadcast({
            "type": "room_created",
            "room_id": room_id,
            "room_name": data.room_name
        })
        return {"success": True, "message": "Room created", "room_id": room_id, "room_name": data.room_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rooms/{room_id}/rename")
async def rename_room(room_id: int, data: RoomRename):
    try:
        db.update_room(room_id, data.new_name)

        await manager.broadcast({
            "type": "room_renamed",
            "room_id": room_id,
            "new_name": data.new_name
        })

        return {"success": True, "message": "Room renamed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/rooms/{room_id}")
async def delete_room(room_id: int):
    try:
        # Check if room exists
        rooms = db.fetch_rooms()
        room_ids = [room[0] for room in rooms] 
        
        if room_id not in room_ids:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # Delete room from database
        db.delete_room(room_id)
        
        # Remove from room_status tracking
        if room_id in room_status:
            del room_status[room_id]
        
        # Broadcast room deletion to all connected clients
        await manager.broadcast({
            "type": "room_deleted",
            "room_id": room_id
        })
        
        return {"message": "Room deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# DEVICES
@app.post("/api/devices/register")
async def register_device(data: DeviceRegister):
    address = data.address.strip()
    if not address:
        raise HTTPException(status_code=400, detail="Invalid address")
    try:
        record = db.fetch_device(address)  
        
        if record is None:
            db.register_device(address)
            record = db.fetch_device(address)
        
        room_id = record[2] if record else 0
        return {"success": True, "address": address, "room_id": room_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices")
async def list_devices():
    devices = db.fetch_devices()
    return [{"address": device[1], "room_id": device[2]} for device in devices]

@app.post("/api/devices/{address}/assign_room")
async def assign_device(address: str, data: DeviceAssign):
    if not address or address == "undefined" or address == "null":
        raise HTTPException(status_code=400, detail="Invalid device address")
    try:
        record = db.fetch_device(address)
        db.assign_device(address, data.room_id)

        await manager.broadcast({
            "type": "device_assigned",
            "address": address,
            "room_id": data.room_id
        })
        return {"success": True, "address": address, "room_id": data.room_id}
    except Exception as e:
            print(f"Error registering device: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices/config")
async def device_config(address: str):
    record = db.fetch_device(address)
    if record is None:
        return {"registered": False, "room_id": 0}
    room_id = record[2]
    return {"registered": True, "room_id": room_id}

# RECORDINGS
@app.get("/api/recordings/{filename}")
async def get_recording(filename: str):
    b_filename = os.path.basename(filename)
    file_path = os.path.join(recordings_dir, b_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return FileResponse(file_path, media_type="audio/wav", filename=b_filename)

@app.get("/api/history/{history_id}/recording")
async def get_recording_by_id(history_id: int):
    try:
        recording_path = db.fetch_recording(history_id)
        if recording_path:
            print(recording_path)

        if not recording_path or not os.path.exists(recording_path):
            raise HTTPException(status_code=404, detail="Recording not found")
        
        filename = os.path.basename(recording_path)
        return FileResponse(recording_path, media_type="audio/wav", filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/recordings")
async def list_recordings():
    try:
        if not os.path.exists(recordings_dir):
            return[]
        
        recordings = []
        for filepath in glob.glob(os.path.join(recordings_dir, "*.wav")):
            filename = os.path.basename(filepath)
            stat = os.stat(filename)
            room_id = None
            room_name = "Unknown"

            try:
                if filename.startswith("ID"):
                    room_id = int(filename.split("_")[0][2:])
                    rooms = db.fetch_rooms()
                    for room in rooms:
                        if room[0] == room_id:
                            room_name = room[1]
                            break
            except:
                pass

            recordings.append({
                "filename": filename,
                "room_id": room_id,
                "room_name": room_name,
                "size": round(stat.st_size / 1024, 1),
                "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%m/%d/%y %I:%M %p")
            })

        recordings.sort(key=lambda x: x["created_at"], reverse=True)
        return recordings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/recordings/{filename}")
async def delete_recording(filename: str):
    b_filename = os.path.basename(filename)
    file_path = os.path.join(recordings_dir, b_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Recording not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": "Recording deleted successfully."}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

    
# MONTHLY EMERGENCIES
@app.get("/api/monthly_emergencies/{year}")
async def get_monthly_emergencies(year: int):
    try:
        monthly_data = db.fetch_monthly_emergencies(year)
        return monthly_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available_years")
async def get_available_years():
    try:
        years = db.fetch_available_years()
        if not years:
            # Return current year if no data exists
            return [datetime.now().year]
        return years
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# TOP EMERGENCY COUNT
@app.get("/api/top_emergencies")
async def get_top_emergencies(year: str = None, range: str = None, start_date: str = None, end_date: str = None):
    try:
        top_data = db.fetch_top_emergencies(year, range, start_date, end_date)
        return top_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ALERT
@app.post("/api/alert")
async def handle_alert(data: AlertData):
    try:
        current_time = datetime.now().strftime("%I:%M:%S %p")
        current_date = datetime.now().strftime("%Y-%m-%d")
        formatted_date = datetime.now().strftime("%m/%d/%y")

        if data.action == "Emergency Alert Detected" or data.action == "Alarming Alert Detected":
            # Set status to 1 (emergency active) - bell will blink
            room_status[data.room_id] = 1
        elif data.action == "Alert Acknowledged":
            # Set status to 0 (normal) - bell will stop blinking
            room_status[data.room_id] = 0
        else:
            raise HTTPException(status_code=400, detail="Invalid action.")

        db.insert_history(
            action=data.action,
            sound_type=data.sound_type,
            date=current_date,
            time=current_time,
            room_id=data.room_id,
            recording_path=data.recording_path
        )
        
        print(f"Alert processed: Room {data.room_id}, Action: {data.action}, Sound Type: {data.sound_type}, Recording: {data.recording_path}, Status: {room_status[data.room_id]}")

        await manager.broadcast({
            "type": "alert_update",
            "room_id": data.room_id,
            "status": room_status[data.room_id],
            "action": data.action,
            "sound_type": data.sound_type,
            "date": formatted_date,
            "time": strip_leading_zero_hour(current_time),
            "has_recording": data.recording_path is not None
        })
        
        return {"success": True, "message": "Alert processed successfully."}
    except Exception as e:
        print(f"Error processing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/status")
async def get_status():
    return room_status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status_update",
            "status": room_status
        })
        print(f"Sent initial status to client: {room_status}")
        
        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for messages from client (if any)
                data = await websocket.receive_text()
                print(f"Received from client: {data}")
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Error in WebSocket loop: {e}")
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/api/shutdown")
async def shutdown_system(data: ShutdownRequest):
    if not data.confirm:
        raise HTTPException(status_code=400, detail="Shutdown not confirmed")
    
    try:
        if data.target in ["all"]:
            # if rpi_ip:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://localhost:58081/shutdown",
                        json={"confirm": True},
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            print("Shutdown signal sent to RPI")
            except Exception as e:
                print(f"Failed to send shutdown to RPI: {e}")
            # else:
            #     print("RPI IP not known, cannot send shutdown signal")
        
        asyncio.create_task(shutdown_web_server())
        return {"success": True, "message": "Web server shutting down..."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def shutdown_web_server():
    await asyncio.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)

# ========================================================
# DASHBOARD PRINT
@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request):
    return templates.TemplateResponse("statistics_report.html", {"request": request})

@app.get("/api/report_data")
async def get_report_data(
    room: int = None,
    range: str = None,
    start_date: str = None,
    end_date: str = None
):
    """Fetch filtered emergency data for the report"""
    from datetime import datetime, timedelta
    
    try:
        conn = db.conn
        query = """
            SELECT h.action, h.date, h.time, h.room_id, r.room_name
            FROM history h
            JOIN room r ON h.room_id = r.room_id
            WHERE h.action = 'Emergency Alert Detected' or h.action = 'Alarming Alert Detected'
        """
        
        conditions = []
        params = []
        
        # Room filter
        if room:
            conditions.append("h.room_id = ?")
            params.append(room)
        
        # Date range filter
        if range:
            today = datetime.now().date()
            
            if range == 'this_year':
                conditions.append("strftime('%Y', h.date) = ?")
                params.append(str(today.year))
            
            elif range == 'this_month':
                conditions.append("strftime('%Y-%m', h.date) = ?")
                params.append(today.strftime('%Y-%m'))
            
            elif range == 'last_30':
                start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
                conditions.append("h.date >= ?")
                params.append(start)
            
            elif range == 'last_7':
                start = (today - timedelta(days=7)).strftime('%Y-%m-%d')
                conditions.append("h.date >= ?")
                params.append(start)
            
            elif range == 'custom' and start_date and end_date:
                conditions.append("h.date BETWEEN ? AND ?")
                params.extend([start_date, end_date])
        
        # Add conditions to query
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY h.date DESC, h.time DESC"
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        # Format data
        emergencies = []
        by_room = {1: 0, 2: 0, 3: 0}
        
        for row in rows:
            action, date_str, time_str, room_id, room_name = row
            
            # Format date as MM/DD/YY
            formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m/%d/%y")
            
            emergencies.append({
                "action": action,
                "date": formatted_date,
                "time": strip_leading_zero_hour(time_str),
                "room_id": room_id,
                "room_name": room_name
            })
            
            by_room[room_id] = by_room.get(room_id, 0) + 1
        
        return {
            "total": len(emergencies),
            "emergencies": emergencies,
            "by_room": by_room
        }
    
    except Exception as e:
        print(f"Error fetching report data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("safensound_app:app", host="0.0.0.0", port=8080, reload=True, log_level="info")
    print("SafeNSound FastAPI is running on...")