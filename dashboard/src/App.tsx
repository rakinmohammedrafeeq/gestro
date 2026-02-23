/**
 * Gestro Dashboard - Main Application Component
 * Author: Rakin Mohammed Rafeeq
 * Description: Real-time monitoring dashboard for hand detection inference
 */

import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import DetectionView from './components/DetectionView';
import MetricsPanel from './components/MetricsPanel';
import ControlPanel from './components/ControlPanel';

interface Metrics {
  total_detections: number;
  active_websocket_connections: number;
  uptime_formatted: string;
  timestamp: string;
}

interface Detection {
  box: {
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
  };
  confidence: number;
  class: string;
}

const App: React.FC = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to metrics WebSocket
    const metricsWs = new WebSocket('ws://localhost:8000/ws/metrics');
    
    metricsWs.onopen = () => {
      console.log('Connected to metrics WebSocket');
      setIsConnected(true);
    };
    
    metricsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMetrics(data);
    };
    
    metricsWs.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    metricsWs.onclose = () => {
      console.log('Disconnected from metrics WebSocket');
      setIsConnected(false);
    };
    
    return () => {
      metricsWs.close();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      // Connect to detection WebSocket
      const detectionWs = new WebSocket('ws://localhost:8000/ws/stream');
      wsRef.current = detectionWs;
      
      detectionWs.onopen = () => {
        console.log('Connected to detection WebSocket');
      };
      
      detectionWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setDetections(data.detections || []);
      };
      
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤚 Gestro Dashboard</h1>
        <p>Real-Time Hand Detection Monitoring</p>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Connected' : '○ Disconnected'}
        </div>
      </header>
      
      <div className="dashboard-container">
        <div className="main-content">
          <DetectionView 
            detections={detections}
            fps={fps}
          />
          <ControlPanel 
            onStartCamera={startCamera}
            isConnected={isConnected}
          />
        </div>
        
        <div className="sidebar">
          <MetricsPanel metrics={metrics} />
        </div>
      </div>
      
      <footer className="App-footer">
        <p>Built by Rakin Mohammed Rafeeq | Gestro v1.0.0</p>
      </footer>
    </div>
  );
};

export default App;
