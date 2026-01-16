# Robotics Vision System

A real-time computer vision system for robotics applications using YOLO26 object detection, BoTSORT tracking, and monocular depth estimation. This system enables robots to perceive, track, and estimate distances to objects in their environment with GPU acceleration.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Output & Logging](#output--logging)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **Real-Time Object Detection**: Uses YOLO26 (2026 standard) for fast and accurate object detection
- **Multi-Object Tracking**: BoTSORT tracker maintains consistent object IDs across frames
- **Monocular Depth Estimation**: Calculates distance to objects using triangle similarity
- **GPU Acceleration**: Optimized for NVIDIA GPUs with FP16 precision support (RTX 3050+)
- **Live Performance Metrics**: Real-time FPS and latency monitoring with visual overlay
- **JSON Logging**: Comprehensive scene data logging for post-processing and analysis
- **Flexible Model Selection**: Choose between Nano (speed) and Small (accuracy) variants

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│     Camera Input (Webcam/USB Camera)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    YOLO26 Object Detector               │
│    (NMS-Free, Layer Fused)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    BoTSORT Tracker                      │
│    (Persistent Object IDs)              │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐  ┌──────────────────┐
│ 2D Boxes & │  │ Monocular Depth  │
│ Class IDs  │  │ Estimation       │
└────────────┘  └──────────────────┘
       │                │
       └────────┬───────┘
                ▼
    ┌──────────────────────┐
    │  Scene JSON Payload  │
    │  + Visual Overlay    │
    │  + Logging           │
    └──────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Object Detection | YOLO26 (Ultralytics) | 8.3.0+ |
| Tracking | BoTSORT | Built-in |
| Computer Vision | OpenCV | 4.10.0+ |
| Deep Learning | PyTorch | Auto (with CUDA) |
| Optimization | ONNX Runtime (GPU) | Latest |

## 📦 Prerequisites

- **Hardware**:
  - NVIDIA GPU (RTX 3050 or better recommended)
  - Minimum 4GB VRAM
  - USB Webcam or integrated camera
  
- **Software**:
  - Python 3.8+
  - CUDA 11.8+ (for GPU acceleration)
  - cuDNN 8.0+ (for GPU acceleration)

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd d:\AIML-Projects\Robotics_Vision_System
```

### 2. Create a Python Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

The system requires YOLO26 pre-trained weights. Run the model download script:

```bash
python download_models.py
```

This will download:
- `yolo26n.pt` - Nano variant (fastest, optimized for RTX 3050)
- `yolo26s.pt` - Small variant (more accurate)

Models are saved to: `models/`

## 📁 Project Structure

```
Robotics_Vision_System/
├── main.py                          # Entry point - Run this to start the system
├── perception_core.py               # Core VisualCortex class with all perception logic
├── download_models.py               # Download YOLO26 models to ./models/
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── models/                          # Pre-trained YOLO26 weights
│   ├── yolo26n.pt                   # Nano model (fastest)
│   └── yolo26s.pt                   # Small model (more accurate)
├── logs/                            # JSON scene logs (auto-created)
│   └── perception_log_*.json        # Timestamped perception data
└── runs/                            # YOLO inference artifacts (auto-created)
    └── detect/
        └── track/
```

## 🎯 Usage

### Basic Startup

```bash
python main.py
```

The system will:
1. Initialize the GPU and load the YOLO26 model
2. Open your default webcam (camera index 0)
3. Display real-time object detection and tracking
4. Show latency and FPS metrics
5. Log all scene data to `logs/perception_log_*.json`

### Controls

- **Q Key**: Gracefully shutdown the system and save logs

### Output Display

The live video display shows:

```
┌─────────────────────────────────────────────┐
│ Model: YOLO26-Nano (NMS-Free)               │
│ Latency: 45.2 ms                            │
│ FPS: 22.1                                   │
├─────────────────────────────────────────────┤
│                                             │
│  [Person #1]                                │
│  ┌─────────────┐  ← Bounding Box           │
│  │ person      │     Z: 1.45m ← Depth     │
│  │ conf: 0.92  │                           │
│  └─────────────┘     ID: #1 (Track ID)    │
│                                             │
│  [Cup #2]                                   │
│  ┌─────────────┐                           │
│  │ cup         │     Z: 0.85m              │
│  │ conf: 0.87  │                           │
│  └─────────────┘     ID: #2                │
│                                             │
└─────────────────────────────────────────────┘
```

## ⚙️ Configuration

### Model Selection

Edit `main.py` to switch between models:

```python
MODEL_NAME = "yolo26n.pt"  # Use 'n' for max speed on RTX 3050
# MODEL_NAME = "yolo26s.pt"  # Use 's' for higher accuracy
```

**Recommendations**:
- **RTX 3050 (Mobile)**: Use `yolo26n.pt` for ~20+ FPS
- **RTX 3070+**: Use `yolo26s.pt` for better accuracy with good FPS

### Camera Selection

Modify `capture_index` in `main.py`:

```python
robot_eye = VisualCortex(model_path=model_path, capture_index=0)
```

- `0` = Default webcam
- `1` = First external USB camera
- `2` = Second external USB camera (etc.)

### Depth Estimation Calibration

In `perception_core.py`, adjust known heights and focal length:

```python
self.KNOWN_HEIGHTS = {
    0: 1700,      # Person (~1.7m)
    67: 150,      # Cell phone (~15cm)
    39: 500,      # Bottle (~50cm)
    41: 300,      # Cup (~30cm)
}
self.FOCAL_LENGTH = 800  # Calibrate for your specific camera
```

**To calibrate `FOCAL_LENGTH`**:
1. Place an object of known height at a known distance
2. Measure its bounding box height in pixels
3. Use formula: `focal_length = (bbox_h * distance_mm) / real_height_mm`
4. Update `FOCAL_LENGTH` with the result

### Performance Tuning

In `perception_core.py`, modify inference parameters:

```python
results = self.model.track(
    frame, 
    persist=True,        # Keep tracking memory
    tracker="botsort.yaml",
    verbose=False,
    half=True            # FP16 precision (faster on RTX 3050)
)
```

## 🔍 Technical Details

### YOLO26 Model

- **Architecture**: Latest Ultralytics YOLOv8 derivative (2026 standard)
- **NMS-Free**: Uses end-to-end detection without traditional Non-Maximum Suppression
- **Layer Fusion**: Automatically fuses layers for faster inference
- **FP16 Support**: Leverages half-precision for RTX 3050 optimization

### BoTSORT Tracker

- **Approach**: Combines appearance features with motion-based matching
- **Persistent IDs**: Each object maintains a unique ID across frames
- **Memory**: Tracks objects even when temporarily occluded

### Monocular Depth Estimation

Uses **triangle similarity principle**:

$$\text{Distance} = \frac{\text{Focal Length} \times \text{Real Height}}{\text{Bounding Box Height (pixels)}}$$

**Assumptions**:
- Single camera (monocular)
- Objects have known real-world heights
- Objects are roughly perpendicular to camera plane

**Limitations**:
- No depth for unknown object classes
- Accuracy depends on camera calibration
- Not suitable for highly tilted objects

## 📊 Output & Logging

### Console Output

```
🚀 Initializing Visual Cortex on: NVIDIA GeForce RTX 3050
📂 Loading Model: D:\AIML-Projects\Robotics_Vision_System\models\yolo26n.pt...
🟢 Perception System Online. Press 'Q' to shutdown.
🛑 System Shutdown. Logs saved to logs/perception_log_1768543133.json
```

### JSON Log Format

Each frame is logged as a JSON object:

```json
{
  "timestamp": "2025-01-16T10:45:33.123456",
  "objects": [
    {
      "id": 1,
      "class": "person",
      "confidence": 0.92,
      "position_2d": [640.5, 360.2],
      "depth_est": 1.45
    },
    {
      "id": 2,
      "class": "cup",
      "confidence": 0.87,
      "position_2d": [320.1, 480.3],
      "depth_est": 0.85
    }
  ]
}
```

**Log Location**: `logs/perception_log_<timestamp>.json`

**Fields**:
- `timestamp`: ISO 8601 formatted timestamp
- `objects[].id`: Persistent tracking ID
- `objects[].class`: Detected object class name
- `objects[].confidence`: Detection confidence (0-1)
- `objects[].position_2d`: [x_center, y_center] in pixels
- `objects[].depth_est`: Estimated distance in meters (-1 if unknown)

## 🐛 Troubleshooting

### **Issue: Model not found error**

```
❌ Error: Model not found at D:\AIML-Projects\Robotics_Vision_System\models\yolo26n.pt
```

**Solution**: Run the download script:
```bash
python download_models.py
```

### **Issue: Low FPS (<10)**

**Possible causes**:
- GPU not being used (check device: should show GPU name)
- Using `yolo26s.pt` instead of `yolo26n.pt`
- `half=True` not enabled
- Insufficient VRAM

**Solution**:
1. Switch to nano model: `MODEL_NAME = "yolo26n.pt"`
2. Verify GPU usage: Check console output for GPU name
3. Enable FP16: Ensure `half=True` in `perception_core.py`

### **Issue: Webcam not opening**

```
Traceback: Cannot open camera
```

**Solutions**:
1. Try different camera indices:
   ```python
   capture_index=0   # Built-in camera
   capture_index=1   # External USB camera
   ```

2. Check camera permissions (Windows may ask for camera access)
3. Ensure no other application is using the camera

### **Issue: CUDA out of memory**

```
torch.cuda.OutOfMemoryError
```

**Solutions**:
1. Reduce frame resolution in `perception_core.py`:
   ```python
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Down from 1280
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Down from 720
   ```

2. Use nano model instead of small
3. Close other GPU-intensive applications

### **Issue: Inaccurate depth estimates**

**Cause**: `FOCAL_LENGTH` not calibrated for your camera

**Solution**: Follow calibration steps in [Depth Estimation Calibration](#depth-estimation-calibration)

## 🔮 Future Enhancements

Potential improvements for future versions:

- [ ] **Stereo Depth**: Implement stereo vision for more accurate 3D localization
- [ ] **3D Bounding Boxes**: Extend to full 3D pose estimation
- [ ] **Multi-Camera Support**: Fuse data from multiple cameras for better coverage
- [ ] **ROS Integration**: Connect to Robot Operating System for robotic control
- [ ] **Persistent Memory**: Store historical tracking data for behavior analysis
- [ ] **Custom Model Training**: Fine-tune YOLO26 on domain-specific objects
- [ ] **Segmentation**: Add instance segmentation for pixel-level object masks
- [ ] **Export Formats**: Support ONNX, TensorRT for deployment on edge devices
- [ ] **Web Dashboard**: Real-time web interface for remote monitoring
- [ ] **Optimization Profiles**: Auto-select model/settings based on hardware

## 📄 License

This project uses Ultralytics YOLOv8, which is available under the AGPL-3.0 license for open-source projects.

## 🤝 Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Update logs and documentation
3. Submit a pull request with detailed descriptions

## 📞 Support

For issues, questions, or suggestions:
- Check the [Troubleshooting](#troubleshooting) section
- Review [Ultralytics Documentation](https://docs.ultralytics.com)
- Check [OpenCV Documentation](https://docs.opencv.org)

---

**Last Updated**: January 2026  
**YOLO Model**: YOLO26 (2026 Standard)  
**Compatible Hardware**: NVIDIA GPU (RTX 3050+)
