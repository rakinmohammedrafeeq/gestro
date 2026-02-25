/**
 * Metrics Panel Component
 * Author: Rakin Mohammed Rafeeq
 * Displays real-time performance metrics
 */

import React from 'react';

interface Metrics {
  total_detections: number;
  active_websocket_connections: number;
  uptime_formatted: string;
  timestamp: string;
}

interface MetricsPanelProps {
  metrics: Metrics | null;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics }) => {
  return (
    <div className="metrics-panel">
      <h2>Performance Metrics</h2>
      
      {metrics ? (
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Total Detections</div>
            <div className="metric-value">{metrics.total_detections}</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Active Connections</div>
            <div className="metric-value">{metrics.active_websocket_connections}</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Uptime</div>
            <div className="metric-value">{metrics.uptime_formatted}</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Last Update</div>
            <div className="metric-value">
              {new Date(metrics.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      ) : (
        <div className="no-data">Waiting for metrics...</div>
      )}
    </div>
  );
};

export default MetricsPanel;
