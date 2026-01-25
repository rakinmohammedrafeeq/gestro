"""
Model Export Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Exports trained checkpoint to frozen inference graph
"""

import tensorflow as tf
import os
import argparse
from object_detection import exporter_lib_v2

def export_inference_graph(pipeline_config_path: str, 
                          trained_checkpoint_dir: str,
                          output_directory: str):
    """
    Export trained model to inference graph
    """
    print(f"Exporting model from: {trained_checkpoint_dir}")
    print(f"Output directory: {output_directory}")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Export the model
    exporter_lib_v2.export_inference_graph(
        input_type='image_tensor',
        pipeline_config_path=pipeline_config_path,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_directory
    )
    
    print(f"Model exported successfully to {output_directory}")
    print("Saved artifacts:")
    for item in os.listdir(output_directory):
        print(f"  - {item}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_config', required=True, 
                       help='Path to pipeline config file')
    parser.add_argument('--checkpoint_dir', required=True,
                       help='Path to trained checkpoint directory')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for inference graph')
    
    args = parser.parse_args()
    
    export_inference_graph(
        args.pipeline_config,
        args.checkpoint_dir,
        args.output_dir
    )
