/**
 * Detection View Component
 * Author: Rakin Mohammed Rafeeq
 * Displays real-time hand detection results
 */

import React from 'react';

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

interface DetectionViewProps {
  detections: Detection[];
  fps: number;
}

const DetectionView: React.FC<DetectionViewProps> = ({ detections, fps }) => {
  return (
    <div className="detection-view">
      <h2>Detection View</h2>
      <div className="video-container">
        <canvas id="detection-canvas" width="640" height="480" />
        <div className="fps-counter">FPS: {fps.toFixed(1)}</div>
      </div>
      
      <div className="detection-info">
        <h3>Detections: {detections.length}</h3>
        {detections.map((det, idx) => (
          <div key={idx} className="detection-item">
            <span className="detection-class">{det.class}</span>
            <span className="detection-confidence">
              {(det.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DetectionView;
