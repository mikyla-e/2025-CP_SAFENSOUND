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
from typing import List, Dict
from datetime import datetime
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database

static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

class RoomRename(BaseModel):
    new_name: str

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
    print("🚀 FastAPI SafeNSound Dashboard started!")
    print("📊 Dashboard available at: http://localhost:8000")
    print("📖 API Documentation at: http://localhost:8000/docs")

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

connected_websockets: List[WebSocket] = []

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/rooms")
async def get_rooms():
    try:
        rooms = db.get_all_rooms()
        rooms_data = {}
        for room_id, room_name in rooms:
            rooms_data[room_id] = {
                "name": room_name,
                "status": room_status.get(room_id, 0)
            }
        return rooms_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/history/{room_id}")
async def get_history(room_id: int):
    try:
        history = db.fetch_history(room_id)
        formatted_history = []
        for record in history:
            formatted_history.append({
                "action": record[1],
                "date": record[2],
                "time": record[3]
            })
        return formatted_history
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
    
@app.post("/api/alert")
async def handle_alert(data: AlertData):
    try:
        current_time = datetime.now().strftime("%H:%M %p")
        current_date = datetime.now().strftime("%m/%d/%y")

        db.insert_history(data.action, current_date, current_time, data.room_id)

        if data.action == "Emergency Detected":
            room_status[data.room_id] = 1
        elif data.action == "Alarm Reset":
            room_status[data.room_id] = 0

        await manager.broadcast({
            "type": "alert_update",
            "room_id": data.room_id,
            "status": room_status[data.room_id],
            "action": data.action,
            "date": current_date,
            "time": current_time
        })
        
        return {"success": True, "message": "Alert processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/audio")
async def receive_audio(data: AudioData):
    """Receive audio data from ESP32 devices"""
    try:
        print(f"🎵 Received audio from Room {data.room_id}: {data.sample_count} samples")
        
        # Here you would process the audio data with your ML model
        # For now, just broadcast that audio was received
        await manager.broadcast({
            "type": "audio_received",
            "room_id": data.room_id,
            "timestamp": data.timestamp,
            "sample_count": data.sample_count
        })
        
        return {"success": True, "message": "Audio received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/status")
async def get_status():
    return room_status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "initial_status",
            "room_status": room_status
        }))
        while True:
            data = await websocket.receive_text()
            print(f"Received message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("safensound_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")