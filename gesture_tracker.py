"""
Gesture Tracking Module for Gestro
Author: Rakin Mohammed Rafeeq
Description: Implements hand tracking with temporal smoothing and gesture recognition
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional

class HandTracker:
    """
    Tracks hand positions over time with smoothing
    """
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.hand_positions = deque(maxlen=history_length)
        self.hand_velocities = deque(maxlen=history_length)
        
    def update(self, detections: List[Dict]):
        """
        Update tracker with new detections
        """
        if len(detections) > 0:
            # Get center positions
            centers = []
            for det in detections:
                box = det['box']
                center_x = (box['xmin'] + box['xmax']) / 2
                center_y = (box['ymin'] + box['ymax']) / 2
                centers.append((center_x, center_y))
            
            self.hand_positions.append(centers)
            
            # Calculate velocity if we have history
            if len(self.hand_positions) >= 2:
                prev_centers = self.hand_positions[-2]
                curr_centers = self.hand_positions[-1]
                
                velocities = []
                for i in range(min(len(prev_centers), len(curr_centers))):
                    vx = curr_centers[i][0] - prev_centers[i][0]
                    vy = curr_centers[i][1] - prev_centers[i][1]
                    velocities.append((vx, vy))
                
                self.hand_velocities.append(velocities)
    
    def get_smoothed_positions(self) -> List[Tuple[float, float]]:
        """
        Get temporally smoothed hand positions
        """
        if len(self.hand_positions) == 0:
            return []
        
        # Average positions over history
        num_hands = len(self.hand_positions[-1])
        smoothed = []
        
        for hand_idx in range(num_hands):
            sum_x, sum_y = 0, 0
            count = 0
            
            for positions in self.hand_positions:
                if hand_idx < len(positions):
                    sum_x += positions[hand_idx][0]
                    sum_y += positions[hand_idx][1]
                    count += 1
            
            if count > 0:
                smoothed.append((sum_x / count, sum_y / count))
        
        return smoothed
    
    def get_average_velocity(self) -> List[Tuple[float, float]]:
        """
        Get average velocity for each tracked hand
        """
        if len(self.hand_velocities) == 0:
            return []
        
        num_hands = max(len(v) for v in self.hand_velocities)
        avg_velocities = []
        
        for hand_idx in range(num_hands):
            sum_vx, sum_vy = 0, 0
            count = 0
            
            for velocities in self.hand_velocities:
                if hand_idx < len(velocities):
                    sum_vx += velocities[hand_idx][0]
                    sum_vy += velocities[hand_idx][1]
                    count += 1
            
            if count > 0:
                avg_velocities.append((sum_vx / count, sum_vy / count))
        
        return avg_velocities

class GestureRecognizer:
    """
    Recognizes gestures from hand tracking data
    """
    def __init__(self, tracker: HandTracker):
        self.tracker = tracker
        self.gesture_threshold_velocity = 0.05  # Threshold for motion gestures
        
    def recognize_gesture(self) -> Optional[str]:
        """
        Recognize current gesture based on hand movement
        """
        velocities = self.tracker.get_average_velocity()
        
        if len(velocities) == 0:
            return None
        
        # Simple gesture recognition based on velocity
        for vx, vy in velocities:
            speed = np.sqrt(vx**2 + vy**2)
            
            if speed < self.gesture_threshold_velocity:
                return "STATIONARY"
            
            # Determine direction
            angle = np.arctan2(vy, vx)
            
            if -np.pi/4 < angle <= np.pi/4:
                return "SWIPE_RIGHT"
            elif np.pi/4 < angle <= 3*np.pi/4:
                return "SWIPE_DOWN"
            elif angle > 3*np.pi/4 or angle <= -3*np.pi/4:
                return "SWIPE_LEFT"
            else:
                return "SWIPE_UP"
        
        return None
    
    def map_gesture_to_action(self, gesture: str) -> Optional[str]:
        """
        Map recognized gesture to system action
        """
        gesture_actions = {
            "SWIPE_RIGHT": "NEXT",
            "SWIPE_LEFT": "PREVIOUS",
            "SWIPE_UP": "SCROLL_UP",
            "SWIPE_DOWN": "SCROLL_DOWN",
            "STATIONARY": "CLICK",
        }
        
        return gesture_actions.get(gesture)

def create_gesture_pipeline():
    """
    Create a complete gesture recognition pipeline
    """
    tracker = HandTracker(history_length=10)
    recognizer = GestureRecognizer(tracker)
    return tracker, recognizer

if __name__ == '__main__':
    # Example usage
    tracker, recognizer = create_gesture_pipeline()
    
    # Simulate detections
    test_detections = [
        {'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}}
    ]
    
    tracker.update(test_detections)
    smoothed = tracker.get_smoothed_positions()
    gesture = recognizer.recognize_gesture()
    
    print(f"Smoothed positions: {smoothed}")
    print(f"Recognized gesture: {gesture}")
