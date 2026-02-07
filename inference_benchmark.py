"""
Inference Benchmarking Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Benchmarks model inference performance
"""

import time
import cv2
import numpy as np
import argparse
import detector_utils
from typing import List, Dict

def benchmark_inference(model_path: str, num_iterations: int = 100, 
                       image_size: tuple = (640, 480)) -> Dict:
    """
    Benchmark model inference performance
    """
    print(f"Loading model from: {model_path}")
    detection_graph, sess = detector_utils.load_inference_graph()
    
    # Create test images
    test_images = []
    for _ in range(10):
        img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        test_images.append(img)
    
    print(f"Running {num_iterations} inference iterations...")
    
    inference_times = []
    
    # Warmup
    for i in range(10):
        _ = detector_utils.detect_objects(test_images[0], detection_graph, sess)
    
    # Benchmark
    for i in range(num_iterations):
        img = test_images[i % len(test_images)]
        
        start_time = time.time()
        boxes, scores = detector_utils.detect_objects(img, detection_graph, sess)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    sess.close()
    
    # Calculate statistics
    metrics = {
        'mean_time_ms': np.mean(inference_times),
        'median_time_ms': np.median(inference_times),
        'std_time_ms': np.std(inference_times),
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times),
        'p95_time_ms': np.percentile(inference_times, 95),
        'p99_time_ms': np.percentile(inference_times, 99),
        'fps': 1000 / np.mean(inference_times)
    }
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to frozen inference graph')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    
    args = parser.parse_args()
    
    metrics = benchmark_inference(
        args.model_path,
        args.iterations,
        (args.height, args.width)
    )
    
    print("\n=== Benchmark Results ===")
    print(f"Mean inference time: {metrics['mean_time_ms']:.2f} ms")
    print(f"Median inference time: {metrics['median_time_ms']:.2f} ms")
    print(f"Std deviation: {metrics['std_time_ms']:.2f} ms")
    print(f"Min time: {metrics['min_time_ms']:.2f} ms")
    print(f"Max time: {metrics['max_time_ms']:.2f} ms")
    print(f"95th percentile: {metrics['p95_time_ms']:.2f} ms")
    print(f"99th percentile: {metrics['p99_time_ms']:.2f} ms")
    print(f"Average FPS: {metrics['fps']:.2f}")
