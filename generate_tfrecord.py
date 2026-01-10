"""
TFRecord Generation Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Converts CSV annotations to TFRecord format for TensorFlow training
"""

import tensorflow as tf
import pandas as pd
import io
import os
from PIL import Image
from object_detection.utils import dataset_util
import argparse

def create_tf_example(row, image_dir):
    """
    Creates a TF Example from a single annotation row
    """
    img_path = os.path.join(image_dir, row['filename'])
    
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image = fid.read()
    
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    
    width, height = image.size
    filename = row['filename'].encode('utf8')
    image_format = b'jpg'
    
    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    
    classes_text = [row['class'].encode('utf8')]
    classes = [1]  # Hand class
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example

def generate_tfrecords(csv_path, image_dir, output_path):
    """
    Generate TFRecord file from CSV annotations
    """
    writer = tf.io.TFRecordWriter(output_path)
    df = pd.read_csv(csv_path)
    
    print(f"Processing {len(df)} annotations...")
    
    for index, row in df.iterrows():
        tf_example = create_tf_example(row, image_dir)
        writer.write(tf_example.SerializeToString())
        
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} images")
    
    writer.close()
    print(f"Successfully created TFRecord: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, help='Path to CSV annotation file')
    parser.add_argument('--image_dir', required=True, help='Path to images directory')
    parser.add_argument('--output_path', required=True, help='Output TFRecord path')
    
    args = parser.parse_args()
    
    generate_tfrecords(args.csv_path, args.image_dir, args.output_path)
