from perception_core import VisualCortex
import os

# --- CONFIGURATION ---
MODEL_DIR = r"D:\AIML-Projects\Robotics_Vision_System\models"
MODEL_NAME = "yolo26n.pt"  # Use 'n' for max speed on RTX 3050, 's' for accuracy

def main():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   -> Did you run the download script from the previous step?")
        return

    # Initialize System
    try:
        # capture_index=0 is usually the default webcam. 
        # If you have an external USB camera, try 1.
        robot_eye = VisualCortex(model_path=model_path, capture_index=0)
        robot_eye.run()
    except Exception as e:
        print(f"💥 Critical Failure: {e}")

if __name__ == "__main__":
    main()
