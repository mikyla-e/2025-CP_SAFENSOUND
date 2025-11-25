from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pydantic import BaseModel
import sys
import os
import asyncio
import json
from typing import List
from datetime import datetime
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database
from datetime import datetime #for date format (separation)

static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

class RoomRename(BaseModel):
    new_name: str

class NewRoom(BaseModel):
    room_name: str

class DeviceRegister(BaseModel):
    device_id: str

class DeviceAssign(BaseModel):
    room_id: int

device_room_map: dict[str, int] = {}


class AlertData(BaseModel):
    room_id: int
    action: str

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
    task = asyncio.create_task(periodic_updates())
    print("FastAPI SafeNSound Dashboard started!")
    print("Dashboard available at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")

    yield
    print("Shutting down...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Periodic updates cancelled.")

app = FastAPI(title="SafeNSound Dashboard", version="1.0", lifespan=lifespan)
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

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_alias(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/rooms")
async def get_rooms():
    try:
        rooms = db.fetch_rooms()
        rooms_data = {}
        for room_id, room_name in rooms:
            rooms_data[room_id] = {
                "name": room_name,
                "status": room_status.get(room_id, 0)
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
            WHERE action = 'Emergency Detected'
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
async def get_history(room_id: int):
    try:
        history = db.fetch_history(room_id)
        formatted_history = []
        for record in history:
            # Format date as MM/DD/YY
            formatted_date = datetime.strptime(record[2], "%Y-%m-%d").strftime("%m/%d/%y")
            formatted_history.append({
                "action": record[1],
                "date": formatted_date,
                "time": strip_leading_zero_hour(record[3])            
            })
        return formatted_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ROOM UPDATES
@app.post("/api/rooms")
async def create_room(data: NewRoom):
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
    
@app.post("/api/devices/register")
async def register_device(data: DeviceRegister):
    device_id = data.device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="Invalid device_id")
    if device_id not in device_room_map:
        device_room_map[device_id] = 0 # unassigned
    return {"success": True, "device_id": device_id, "room_id": device_room_map[device_id]}

@app.get("/api/devices")
async def list_devices():
    return [{"device_id": d, "room_id": r} for d, r in device_room_map.items()]

@app.post("/api/devices/{device_id}/assign_room")
async def assign_device(device_id: str, data: DeviceAssign):
    if device_id not in device_room_map:
        raise HTTPException(status_code=404, detail="Device not registered")
    device_room_map[device_id] = data.room_id
    
    await manager.broadcast({
        "type": "device_assigned",
        "device_id": device_id,
        "room_id": data.room_id
    })
    return {"success": True, "device_id": device_id, "room_id": data.room_id}

@app.get("/api/devices/config")
async def device_config(device_id: str):
    if device_id not in device_room_map:
        return {"registered": False, "room_id": 0}
    return {"registered": True, "room_id": device_room_map[device_id]}
    
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

        if data.action == "Emergency Detected":
            # Set status to 1 (emergency active) - bell will blink
            room_status[data.room_id] = 1
            
        elif data.action == "Alert Acknowledged":
            # Set status to 0 (normal) - bell will stop blinking
            room_status[data.room_id] = 0
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action.")

        db.insert_history(data.action, current_date, current_time, data.room_id)
        
        print(f"Alert processed: Room {data.room_id}, Action: {data.action}, Status: {room_status[data.room_id]}")

        await manager.broadcast({
            "type": "alert_update",
            "room_id": data.room_id,
            "status": room_status[data.room_id],
            "action": data.action,
            "date": formatted_date,
            "time": strip_leading_zero_hour(current_time)
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
            WHERE h.action = 'Emergency Detected'
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
    uvicorn.run("safensound_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")