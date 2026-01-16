import os
from ultralytics import YOLO

# 1. Define your strict target directory
target_dir = r"D:\AIML-Projects\Robotics_Vision_System\models"

# 2. Create the directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"📂 Created directory: {target_dir}")

# 3. Change the current working directory to the target
# Ultralytics downloads to the current working directory by default
os.chdir(target_dir)
print(f"📍 Set download path to: {os.getcwd()}")

# 4. Trigger downloads for YOLO26 (The 2026 Standard)
# We download 'Nano' (n) for speed and 'Small' (s) for higher accuracy
print("⬇️ Downloading YOLO26 Nano (Fastest)...")
model_n = YOLO("yolo26n.pt") 

print("⬇️ Downloading YOLO26 Small (More Accurate)...")
model_s = YOLO("yolo26s.pt")

print("\n✅ Success! Models are saved in:")
print(f"   - {os.path.join(target_dir, 'yolo26n.pt')}")
print(f"   - {os.path.join(target_dir, 'yolo26s.pt')}")
