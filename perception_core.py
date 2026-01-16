import cv2
import torch
import time
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from collections import deque

class VisualCortex:
    def __init__(self, model_path, capture_index=0, target_fps=30):
        """
        Initialize the Robotics Perception Core.
        """
        # 1. Device Setup (Force GPU)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Visual Cortex on: {torch.cuda.get_device_name(0) if self.device != 'cpu' else 'CPU'}")

        # 2. Load YOLO26 Model (NMS-Free)
        # fusing=True merges layers for faster inference
        print(f"📂 Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.model.fuse() 
        
        # 3. Camera Setup
        self.cap = cv2.VideoCapture(capture_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        # 4. Perception Constants (Approximations for Single Camera)
        # Real-world heights (in mm) for distance estimation
        self.KNOWN_HEIGHTS = {
            0: 1700,  # Person (~1.7m)
            67: 150,  # Cell phone (~15cm)
            39: 500,  # Bottle (~50cm)
            41: 300,  # Cup (~30cm)
            # Add more class IDs as needed
        }
        self.FOCAL_LENGTH = 800  # Calibrate this for your specific webcam if needed

        # 5. Logging & Metrics
        self.fps_buffer = deque(maxlen=30)
        self.log_file = f"logs/perception_log_{int(time.time())}.json"
        self._init_log()

    def _init_log(self):
        """Creates the JSON log file."""
        import os
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open(self.log_file, 'w') as f:
            f.write("[\n") # Start JSON array

    def estimate_depth(self, bbox_h, class_id):
        """
        Calculate distance using Monocular Triangle Similarity.
        Distance = (Focal_Length * Real_Object_Height) / Object_Pixel_Height
        """
        if class_id in self.KNOWN_HEIGHTS:
            real_height = self.KNOWN_HEIGHTS[class_id]
            # Avoid division by zero
            if bbox_h < 1: bbox_h = 1
            distance_mm = (self.FOCAL_LENGTH * real_height) / bbox_h
            return distance_mm / 1000.0  # Convert to meters
        return -1.0  # Unknown depth

    def run(self):
        print("🟢 Perception System Online. Press 'Q' to shutdown.")
        
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            # --- INFERENCE STEP ---
            # YOLO26 + BoTSORT Tracker
            # persist=True keeps track of IDs (memory)
            # verbose=False reduces console spam
            # half=True uses FP16 (Faster on RTX 3050)
            results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, half=True)

            # --- DATA PROCESSING ---
            scene_data = {
                "timestamp": datetime.now().isoformat(),
                "objects": []
            }
            
            # Annotator provided by Ultralytics is fast
            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                # Extract data: Bounding Boxes, Class IDs, Tracking IDs
                boxes = results[0].boxes.xywh.cpu().numpy()  # x_center, y_center, width, height
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    x, y, w, h = box
                    
                    # 1. Estimate Depth
                    dist = self.estimate_depth(h, cls_id)
                    dist_str = f"{dist:.2f}m" if dist > 0 else "N/A"

                    # 2. Update Scene Data (JSON Payload)
                    obj_data = {
                        "id": track_id,
                        "class": results[0].names[cls_id],
                        "confidence": round(conf, 2),
                        "position_2d": [float(x), float(y)],
                        "depth_est": float(dist) if dist > 0 else None
                    }
                    scene_data["objects"].append(obj_data)

                    # 3. Draw Depth on Screen (Custom Overlay)
                    # We draw strictly 'depth' since YOLO draws the box/label
                    text_pos = (int(x - w/2), int(y - h/2) - 10)
                    cv2.putText(annotated_frame, f"Z: {dist_str}", text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- METRICS & DISPLAY ---
            inference_time = (time.time() - start_time) * 1000
            fps = 1000 / inference_time if inference_time > 0 else 0
            self.fps_buffer.append(fps)
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

            # Dashboard Overlay
            cv2.rectangle(annotated_frame, (0, 0), (280, 90), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Model: YOLO26-Nano (NMS-Free)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Latency: {inference_time:.1f} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Robotics Visual Perception System", annotated_frame)
            
            # Write to log (Append mode simulation)
            with open(self.log_file, 'a') as f:
                json.dump(scene_data, f)
                f.write(",\n")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        # Close the JSON array properly
        with open(self.log_file, 'a') as f:
            f.write("{}]") 
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"🛑 System Shutdown. Logs saved to {self.log_file}")
