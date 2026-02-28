# Gestro - Real-Time Intelligent Gesture Interaction Platform

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time computer vision platform that detects and recognizes hand gestures from live camera input using deep learning and object detection. Built with transfer learning on SSD MobileNet V2 FPNLite for responsive, intelligent gesture-based interaction.

## 🎯 Project Overview

Gestro is a complete end-to-end computer vision pipeline that combines transfer learning, real-time object detection, and interactive monitoring to enable gesture-based human-computer interaction. The platform includes:

- **Real-time hand detection** from webcam input using OpenCV and TensorFlow
- **Fine-tuned SSD MobileNet V2 FPNLite** model using TensorFlow Object Detection API
- **REST and WebSocket APIs** built with FastAPI for inference monitoring
- **Interactive React/TypeScript dashboard** for performance metrics and visualization
- **Production-ready deployment** with Docker and CI/CD pipelines

## 🚀 Key Features

- **Transfer Learning Pipeline**: Fine-tuned pre-trained SSD MobileNet V2 FPNLite on EgoHands dataset
- **Real-Time Inference**: Multi-threaded webcam processing with TensorFlow and OpenCV
- **Hand Tracking**: Confidence filtering, temporal smoothing, and gesture-to-command mapping
- **API Services**: FastAPI-based REST and WebSocket endpoints for real-time monitoring
- **Interactive Dashboard**: React/TypeScript frontend for inference visualization
- **Performance Optimization**: Multi-threading and optimized inference for low-latency detection
- **CI/CD Integration**: GitHub Actions for automated testing and deployment

## 🛠️ Technology Stack

**Core Technologies:**
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- TensorFlow Object Detection API

**Backend Services:**
- FastAPI
- WebSockets
- REST APIs

**Frontend:**
- React
- TypeScript

**DevOps & Tools:**
- Docker
- pytest
- GitHub Actions
- Linux

## 📊 Model Performance

The final model was trained on **22,500 iterations** using transfer learning on the SSD MobileNet V2 FPNLite architecture:

- **Dataset**: EgoHands (4,000+ annotated hand images)
- **Architecture**: SSD MobileNet V2 FPNLite
- **Training Method**: Transfer Learning
- **Confidence Threshold**: 0.27
- **Detection Classes**: 1 (Hand)

### Sample Results

Real-time hand detection with bounding boxes, confidence scores, and center point tracking:

```
FPS: ~25-30 on CPU
Detection Latency: <50ms
Multi-hand Support: Up to 2 hands simultaneously
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gestro Platform                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Webcam     │─────▶│   OpenCV     │─────▶│ TensorFlow│ │
│  │   Input      │      │   Capture    │      │  Inference│ │
│  └──────────────┘      └──────────────┘      └─────┬─────┘ │
│                                                      │       │
│                        ┌─────────────────────────────┘       │
│                        ▼                                      │
│              ┌──────────────────┐                            │
│              │  SSD MobileNet   │                            │
│              │  V2 FPNLite      │                            │
│              │  (Fine-tuned)    │                            │
│              └────────┬─────────┘                            │
│                       │                                       │
│         ┌─────────────┴─────────────┐                        │
│         ▼                           ▼                        │
│  ┌─────────────┐           ┌──────────────┐                 │
│  │   FastAPI   │           │   WebSocket  │                 │
│  │  REST API   │           │   Server     │                 │
│  └──────┬──────┘           └──────┬───────┘                 │
│         │                         │                          │
│         └────────────┬────────────┘                          │
│                      ▼                                        │
│            ┌──────────────────┐                              │
│            │  React Dashboard │                              │
│            │   (TypeScript)   │                              │
│            └──────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Gestro/
├── core/
│   ├── dataset_preparation.py    # Dataset download and preprocessing
│   ├── detector_utils.py          # Core detection utilities
│   ├── label_map_util.py          # Label mapping utilities
│   └── inference_engine.py        # Real-time inference engine
├── models/
│   ├── training/                  # Training scripts and notebooks
│   └── frozen_inference_graphs/   # Trained model checkpoints
├── api/
│   ├── main.py                    # FastAPI application
│   ├── websocket_handler.py       # WebSocket connections
│   └── routes/                    # API endpoints
├── dashboard/
│   ├── src/                       # React/TypeScript source
│   ├── components/                # UI components
│   └── services/                  # API integration
├── tests/
│   ├── test_detection.py          # Unit tests
│   └── test_api.py                # Integration tests
├── docker/
│   ├── Dockerfile                 # Container configuration
│   └── docker-compose.yml         # Service orchestration
├── .github/
│   └── workflows/                 # CI/CD pipelines
├── configs/
│   └── model_config.yaml          # Model configuration
└── requirements.txt               # Python dependencies
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Webcam or video input device
- (Optional) CUDA-compatible GPU for faster inference

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rakinmohammedrafeeq/gestro.git
cd gestro
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained model:**
The fine-tuned models are included in `frozen_inference_graphs/`. The recommended model is `frozen_inference_graph_v2_22.5k.pb`.

## 🎮 Usage

### Real-Time Hand Detection

Run the multi-threaded detection with webcam:

```bash
python multi_threaded_detection.py --source 0 --num_hands 2 --display 1
```

**Parameters:**
- `--source`: Camera device index (default: 0)
- `--num_hands`: Maximum number of hands to detect (default: 2)
- `--width`: Frame width (default: 400)
- `--height`: Frame height (default: 225)
- `--display`: Show detection window (1 for yes, 0 for no)
- `--num-workers`: Number of worker threads (default: 4)
- `--fps`: Display FPS counter (default: 1)

### Dataset Preparation

To prepare the EgoHands dataset for training:

```bash
python create_dataset.py
```

This will:
1. Download the EgoHands dataset (~1.3GB)
2. Extract and process annotations
3. Generate CSV files for training
4. Split data into train/test sets (90/10)

### API Server

Start the FastAPI server:

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`

### Dashboard

Launch the React dashboard:

```bash
cd dashboard
npm install
npm start
```

Access the dashboard at `http://localhost:3000`

## 🧪 Testing

Run unit and integration tests:

```bash
pytest tests/ -v
```

Run specific test suites:

```bash
# Test detection engine
pytest tests/test_detection.py

# Test API endpoints
pytest tests/test_api.py
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t gestro:latest -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 -p 3000:3000 --device /dev/video0 gestro:latest
```

Or use docker-compose:

```bash
docker-compose -f docker/docker-compose.yml up
```

## 📈 Training Your Own Model

To train a custom model:

1. Prepare your dataset in the required format
2. Update `hand_label_map.pbtxt` with your classes
3. Open `training_ssd_mobilenet_v2.ipynb` in Google Colab
4. Follow the notebook instructions to train on GPU
5. Export the frozen inference graph
6. Place the `.pb` file in `frozen_inference_graphs/`

## 🎯 Use Cases

- **Touchless Computer Control**: Navigate interfaces without physical contact
- **Accessibility Tools**: Enable computer interaction for users with mobility limitations
- **Gaming**: Gesture-based game controls
- **Sign Language Recognition**: Foundation for ASL/gesture language translation
- **Smart Home Control**: Gesture-based IoT device control
- **Interactive Presentations**: Control slides and media with hand gestures

## 🔬 Technical Details

### Object Detection Model

- **Base Model**: SSD MobileNet V2 FPNLite (COCO pre-trained)
- **Transfer Learning**: Fine-tuned on EgoHands dataset
- **Input Size**: 300x300 RGB images
- **Output**: Bounding boxes with confidence scores

### Performance Optimization

- **Multi-threading**: Separate threads for capture and inference
- **Frame Queuing**: Input/output queues for smooth processing
- **Temporal Smoothing**: Reduces detection jitter
- **Confidence Filtering**: Eliminates false positives (threshold: 0.27)

### API Endpoints

**REST API:**
- `GET /health` - Service health check
- `POST /detect` - Single image detection
- `GET /metrics` - Performance metrics
- `GET /model/info` - Model information

**WebSocket:**
- `/ws/stream` - Real-time detection stream
- `/ws/metrics` - Live performance metrics

## 📚 Dataset

**EgoHands Dataset** (Indiana University)
- 4,800 labeled frames
- 48 Google Glass videos
- 15,000+ hand annotations
- Diverse environments and hand poses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rakin Mohammed Rafeeq**
- GitHub: [@rakinmohammedrafeeq](https://github.com/rakinmohammedrafeeq)
- Project Link: [https://github.com/rakinmohammedrafeeq/gestro](https://github.com/rakinmohammedrafeeq/gestro)

## 🙏 Acknowledgments

- **TensorFlow Object Detection API** - Framework for training
- **EgoHands Dataset** - Indiana University for the hand detection dataset
- **SSD Paper** by Liu et al. - Single Shot MultiBox Detector architecture
- **Victor Dibia** - Initial hand tracking implementation inspiration
- **OpenCV Community** - Computer vision tools and libraries

## 📊 Project Timeline

**Development Period**: January 2026 - February 2026

- ✅ Dataset preparation and annotation pipeline
- ✅ Transfer learning and model training
- ✅ Real-time inference engine
- ✅ Multi-threading optimization
- ✅ REST API with FastAPI
- ✅ WebSocket real-time streaming
- ✅ React/TypeScript dashboard
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ Integration testing and optimization

## 📞 Support

For questions, issues, or suggestions, please open an issue on GitHub or contact me directly.

---

**Built with ❤️ using Python, TensorFlow, and OpenCV**
