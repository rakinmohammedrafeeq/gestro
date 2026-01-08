"""
Data Preprocessing Utilities for Gestro
Author: Rakin Mohammed Rafeeq
Description: Helper functions for image preprocessing and augmentation
"""

import cv2
import numpy as np
from typing import Tuple, List

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size maintaining aspect ratio"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range"""
    return image.astype(np.float32) / 255.0

def augment_brightness(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """Randomly adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] *= (1.0 + np.random.uniform(-factor, factor))
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def flip_horizontal(image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flip image and bounding boxes horizontally"""
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = boxes.copy()
    flipped_boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]  # xmin, xmax
    return flipped_image, flipped_boxes

def random_crop(image: np.ndarray, boxes: np.ndarray, 
                min_crop: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop image with bounding boxes"""
    height, width = image.shape[:2]
    crop_factor = np.random.uniform(min_crop, 1.0)
    
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)
    
    x = np.random.randint(0, width - new_width + 1)
    y = np.random.randint(0, height - new_height + 1)
    
    cropped_image = image[y:y+new_height, x:x+new_width]
    
    # Adjust bounding boxes
    adjusted_boxes = boxes.copy()
    adjusted_boxes[:, [1, 3]] = (boxes[:, [1, 3]] * width - x) / new_width
    adjusted_boxes[:, [0, 2]] = (boxes[:, [0, 2]] * height - y) / new_height
    
    # Clip boxes to valid range
    adjusted_boxes = np.clip(adjusted_boxes, 0, 1)
    
    return cropped_image, adjusted_boxes

def preprocess_for_inference(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image
