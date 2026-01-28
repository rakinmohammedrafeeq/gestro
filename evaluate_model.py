"""
Model Evaluation Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Evaluates trained model on test set and generates metrics
"""

import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import argparse
from typing import Dict, List
import detector_utils

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_model(model_path: str, test_csv: str, image_dir: str, 
                   iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate model on test set
    """
    # Load model
    detection_graph, sess = detector_utils.load_inference_graph()
    
    # Load test annotations
    df = pd.read_csv(test_csv)
    
    metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_images': df['filename'].nunique(),
        'inference_times': []
    }
    
    print(f"Evaluating on {metrics['total_images']} test images...")
    
    for filename in df['filename'].unique():
        img_path = f"{image_dir}/{filename}"
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Get ground truth boxes for this image
        gt_boxes = df[df['filename'] == filename][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Run inference
        import time
        start = time.time()
        boxes, scores = detector_utils.detect_objects(img, detection_graph, sess)
        inference_time = (time.time() - start) * 1000
        metrics['inference_times'].append(inference_time)
        
        # Filter predictions by confidence
        pred_boxes = []
        for i, score in enumerate(scores):
            if score > 0.27:
                pred_boxes.append(boxes[i])
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_box in pred_boxes:
            matched = False
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    metrics['true_positives'] += 1
                    matched_gt.add(idx)
                    matched = True
                    break
            if not matched:
                metrics['false_positives'] += 1
        
        # Count unmatched ground truth as false negatives
        metrics['false_negatives'] += len(gt_boxes) - len(matched_gt)
    
    # Calculate final metrics
    precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) \
                if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) \
             if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1_score
    metrics['avg_inference_time'] = np.mean(metrics['inference_times'])
    
    sess.close()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to frozen inference graph')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV')
    parser.add_argument('--image_dir', required=True, help='Path to test images')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    metrics = evaluate_model(args.model_path, args.test_csv, args.image_dir, args.iou_threshold)
    
    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Avg Inference Time: {metrics['avg_inference_time']:.2f}ms")
