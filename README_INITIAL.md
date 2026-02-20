# Gestro

Real-time hand gesture detection using computer vision and deep learning.

## Project Goals
- Implement hand detection using transfer learning
- Build real-time inference pipeline
- Create monitoring dashboard

## Progress
✅ Dataset preparation pipeline complete
✅ EgoHands dataset integration
✅ Data preprocessing and augmentation
✅ TFRecord generation
✅ Model training complete (22,500 steps)
✅ Transfer learning on SSD MobileNet V2 FPNLite
✅ Model evaluation: mAP@0.5 = 0.887
✅ Real-time inference engine with multi-threading
✅ Hand tracking with temporal smoothing
✅ Gesture recognition pipeline
✅ Video processing capabilities
✅ FastAPI REST API with multiple endpoints
✅ WebSocket support for real-time streaming
✅ API middleware (logging, rate limiting, error handling)
✅ Comprehensive test suite (pytest)
🚧 Dashboard UI development in progress

## Model Performance
- Training Steps: 22,500
- Final mAP: 0.887
- Inference Time: ~35ms per image (GPU)
- Real-time FPS: ~28 FPS (webcam)
- Model Size: 19.2 MB

## API Features
- Image detection endpoint
- Batch processing
- Base64 image support
- WebSocket real-time streaming
- Metrics streaming
- Rate limiting
- Request logging

## Status
🚧 Under Development

Author: Rakin Mohammed Rafeeq
