/**
 * Control Panel Component
 * Author: Rakin Mohammed Rafeeq
 * Controls for starting/stopping detection
 */

import React from 'react';

interface ControlPanelProps {
  onStartCamera: () => void;
  isConnected: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
  onStartCamera, 
  isConnected 
}) => {
  return (
    <div className="control-panel">
      <h2>Controls</h2>
      <div className="controls">
        <button 
          onClick={onStartCamera}
          disabled={!isConnected}
          className="btn btn-primary"
        >
          Start Camera
        </button>
        
        <button 
          className="btn btn-secondary"
          disabled={!isConnected}
        >
          Stop Detection
        </button>
        
        <div className="control-settings">
          <label>
            Confidence Threshold:
            <input type="range" min="0" max="100" defaultValue="27" />
            <span>0.27</span>
          </label>
          
          <label>
            Max Hands:
            <select defaultValue="2">
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
