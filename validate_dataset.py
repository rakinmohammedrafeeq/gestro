"""
Dataset Validation Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Validates dataset integrity and visualizes annotations
"""

import cv2
import pandas as pd
import os
import argparse
from typing import List

def validate_annotations(csv_path: str, image_dir: str) -> dict:
    """
    Validate annotation file and check image integrity
    """
    df = pd.read_csv(csv_path)
    
    stats = {
        'total_annotations': len(df),
        'unique_images': df['filename'].nunique(),
        'missing_images': [],
        'invalid_boxes': [],
        'class_distribution': df['class'].value_counts().to_dict()
    }
    
    print(f"Validating {stats['total_annotations']} annotations...")
    
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        
        # Check if image exists
        if not os.path.exists(img_path):
            stats['missing_images'].append(row['filename'])
            continue
        
        # Check bounding box validity
        if row['xmin'] >= row['xmax'] or row['ymin'] >= row['ymax']:
            stats['invalid_boxes'].append(row['filename'])
        
        if row['xmin'] < 0 or row['ymin'] < 0:
            stats['invalid_boxes'].append(row['filename'])
    
    return stats

def visualize_annotations(csv_path: str, image_dir: str, num_samples: int = 5):
    """
    Visualize random samples with bounding boxes
    """
    df = pd.read_csv(csv_path)
    samples = df.sample(min(num_samples, len(df)))
    
    for index, row in samples.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Draw bounding box
        cv2.rectangle(
            img,
            (int(row['xmin']), int(row['ymin'])),
            (int(row['xmax']), int(row['ymax'])),
            (0, 255, 0),
            2
        )
        
        # Add label
        cv2.putText(
            img,
            row['class'],
            (int(row['xmin']), int(row['ymin']) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('Annotation Validation', img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, help='Path to CSV annotation file')
    parser.add_argument('--image_dir', required=True, help='Path to images directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize samples')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Validate dataset
    stats = validate_annotations(args.csv_path, args.image_dir)
    
    print("\n=== Validation Results ===")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Unique images: {stats['unique_images']}")
    print(f"Missing images: {len(stats['missing_images'])}")
    print(f"Invalid boxes: {len(stats['invalid_boxes'])}")
    print(f"Class distribution: {stats['class_distribution']}")
    
    if args.visualize:
        print("\nVisualizing samples...")
        visualize_annotations(args.csv_path, args.image_dir, args.num_samples)
