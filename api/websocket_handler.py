"""
WebSocket Handler for Gestro API
Author: Rakin Mohammed Rafeeq
Description: Manages WebSocket connections for real-time detection streaming
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    """
    Manages WebSocket connections for real-time communication
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        """
        Accept and register new WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if client_info:
            self.connection_metadata[websocket] = {
                **client_info,
                'connected_at': datetime.now().isoformat(),
                'messages_sent': 0,
                'messages_received': 0
            }
        
        print(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata.pop(websocket)
            print(f"Connection closed. Duration: {metadata.get('connected_at')}")
        
        print(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """
        Send message to specific client
        """
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['messages_sent'] += 1
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict, exclude: WebSocket = None):
        """
        Broadcast message to all connected clients
        """
        disconnected = []
        
        for connection in self.active_connections:
            if connection == exclude:
                continue
            
            try:
                await connection.send_json(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]['messages_sent'] += 1
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def get_connection_count(self) -> int:
        """
        Get number of active connections
        """
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict:
        """
        Get statistics for all connections
        """
        return {
            'total_connections': len(self.active_connections),
            'connections': [
                {
                    'id': i,
                    'connected_at': meta.get('connected_at'),
                    'messages_sent': meta.get('messages_sent', 0),
                    'messages_received': meta.get('messages_received', 0)
                }
                for i, meta in enumerate(self.connection_metadata.values())
            ]
        }

class DetectionStreamHandler:
    """
    Handles real-time detection streaming over WebSocket
    """
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.stream_active = False
    
    async def start_stream(self, websocket: WebSocket):
        """
        Start detection stream for a client
        """
        self.stream_active = True
        
        try:
            while self.stream_active:
                # Receive frame data from client
                data = await websocket.receive_json()
                
                if 'command' in data:
                    if data['command'] == 'stop':
                        self.stream_active = False
                        break
                
                # Process and send back results
                # (actual detection logic handled in main.py)
                
                await asyncio.sleep(0.01)  # Small delay to prevent overload
                
        except WebSocketDisconnect:
            print("Client disconnected from detection stream")
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            self.stream_active = False
    
    def stop_stream(self):
        """
        Stop the detection stream
        """
        self.stream_active = False

class MetricsStreamHandler:
    """
    Handles real-time metrics streaming over WebSocket
    """
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.update_interval = 1.0  # seconds
    
    async def stream_metrics(self, websocket: WebSocket, metrics_callback):
        """
        Stream metrics to client at regular intervals
        """
        try:
            while True:
                # Get latest metrics
                metrics = metrics_callback()
                
                # Send to client
                await self.manager.send_personal_message(metrics, websocket)
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
        except WebSocketDisconnect:
            print("Client disconnected from metrics stream")
        except Exception as e:
            print(f"Metrics stream error: {e}")

# Global connection manager instance
manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    """
    Get the global connection manager instance
    """
    return manager
