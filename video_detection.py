"""
Video Detection Script for Gestro
Author: Rakin Mohammed Rafeeq
Description: Process video files with hand detection
"""

import cv2
import argparse
import detector_utils
import time

def process_video(video_path: str, output_path: str, display: bool = True):
    """
    Process video file with hand detection
    """
    # Load model
    print("Loading detection model...")
    detection_graph, sess = detector_utils.load_inference_graph()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores = detector_utils.detect_objects(frame_rgb, detection_graph, sess)
        
        # Draw detections
        detector_utils.draw_box_on_image(
            num_hands_detect=2,
            score_thresh=0.27,
            scores=scores,
            boxes=boxes,
            im_width=width,
            im_height=height,
            image_np=frame
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        detector_utils.draw_fps_on_image(f"FPS: {current_fps:.1f}", frame)
        
        # Write frame
        out.write(frame)
        
        if display:
            cv2.imshow('Video Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    sess.close()
    
    print(f"\nProcessing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--display', action='store_true', help='Display video while processing')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.display)
