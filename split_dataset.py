"""
Dataset Splitting Utility for Gestro
Author: Rakin Mohammed Rafeeq
Description: Splits dataset into train/test sets with stratification
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import os

def split_dataset(csv_path: str, train_ratio: float = 0.9, random_seed: int = 42):
    """
    Split dataset into train and test sets
    """
    df = pd.read_csv(csv_path)
    
    # Get unique image filenames
    unique_images = df['filename'].unique()
    
    # Split images
    train_images, test_images = train_test_split(
        unique_images,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    # Create train and test dataframes
    train_df = df[df['filename'].isin(train_images)]
    test_df = df[df['filename'].isin(test_images)]
    
    print(f"Total images: {len(unique_images)}")
    print(f"Train images: {len(train_images)} ({len(train_df)} annotations)")
    print(f"Test images: {len(test_images)} ({len(test_df)} annotations)")
    
    return train_df, test_df

def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """
    Save train and test splits to CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_labels.csv')
    test_path = os.path.join(output_dir, 'test_labels.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train set to: {train_path}")
    print(f"Saved test set to: {test_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, help='Path to full CSV annotation file')
    parser.add_argument('--output_dir', required=True, help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Train set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_df, test_df = split_dataset(args.csv_path, args.train_ratio, args.seed)
    save_splits(train_df, test_df, args.output_dir)
    
    print("\nDataset split complete!")
