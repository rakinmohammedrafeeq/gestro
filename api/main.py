"""
FastAPI Main Application for Gestro
Author: Rakin Mohammed Rafeeq
Description: REST API and WebSocket server for real-time hand detection monitoring
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List, Dict
import json
import asyncio
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector_utils

app = FastAPI(
    title="Gestro API",
    description="Real-Time Hand Gesture Detection API",
    version="1.0.0"
)

# CORS middleware for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
detection_graph = None
sess = None
metrics = {
    "total_detections": 0,
    "avg_inference_time": 0,
    "active_connections": 0,
    "uptime_start": datetime.now()
}

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        metrics["active_connections"] = len(self.active_connections)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        metrics["active_connections"] = len(self.active_connections)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize TensorFlow model on startup"""
    global detection_graph, sess
    print("Loading detection model...")
    detection_graph, sess = detector_utils.load_inference_graph()
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gestro API - Real-Time Hand Detection",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "websocket": "/ws/stream"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detection_graph is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_hands(file: UploadFile = File(...)):
    """
    Detect hands in uploaded image
    Returns bounding boxes and confidence scores
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        start_time = datetime.now()
        boxes, scores = detector_utils.detect_objects(image_rgb, detection_graph, sess)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Filter by confidence threshold
        threshold = 0.27
        detections = []
        for i, score in enumerate(scores):
            if score > threshold:
                box = boxes[i]
                detections.append({
                    "box": {
                        "ymin": float(box[0]),
                        "xmin": float(box[1]),
                        "ymax": float(box[2]),
                        "xmax": float(box[3])
                    },
                    "confidence": float(score),
                    "class": "hand"
                })
        
        # Update metrics
        metrics["total_detections"] += len(detections)
        
        return {
            "success": True,
            "detections": detections,
            "num_hands": len(detections),
            "inference_time_ms": round(inference_time, 2),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    uptime = (datetime.now() - metrics["uptime_start"]).total_seconds()
    return {
        "total_detections": metrics["total_detections"],
        "active_websocket_connections": metrics["active_connections"],
        "uptime_seconds": round(uptime, 2),
        "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_name": "SSD MobileNet V2 FPNLite",
        "training_iterations": 22500,
        "dataset": "EgoHands",
        "num_classes": 1,
        "confidence_threshold": 0.27,
        "input_size": "300x300",
        "framework": "TensorFlow",
        "model_loaded": detection_graph is not None
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection streaming
    Clients can send frames and receive detection results
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            if "frame" in data:
                # Decode base64 frame
                import base64
                frame_data = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Detect hands
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, scores = detector_utils.detect_objects(
                        frame_rgb, detection_graph, sess
                    )
                    
                    # Filter detections
                    threshold = 0.27
                    detections = []
                    for i, score in enumerate(scores):
                        if score > threshold:
                            box = boxes[i]
                            detections.append({
                                "box": box.tolist(),
                                "confidence": float(score)
                            })
                    
                    # Send results back
                    await websocket.send_json({
                        "detections": detections,
                        "num_hands": len(detections),
                        "timestamp": datetime.now().isoformat()
                    })
            
            await asyncio.sleep(0.01)  # Small delay to prevent overload
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming
    """
    await manager.connect(websocket)
    try:
        while True:
            # Send metrics every second
            await websocket.send_json(await get_metrics())
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Metrics WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
