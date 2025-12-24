import os
import cv2
import time
import datetime
from ultralytics import YOLO
from picamera2 import Picamera2


# --- 1. CONFIGURATION & CONSTANTS ---
WINNING_FOCUS = 500  # The focus value you found sharpest
TARGET_ANIMALS = ['bird', 'cat', 'dog', 'squirrel']
COOLDOWN_SECONDS = 3.0
RESOLUTION = (1280, 720) # 720p is safer for Pi 4 memory limits


# --- 2. HARDWARE CONTROL FUNCTIONS ---
def set_manual_focus(val):
    """Bypasses kernel drivers to move the Arducam motor directly."""
    try:
        value = (val << 4) & 0x3ff0
        dat1 = (value >> 8) & 0x3f
        dat2 = value & 0xf0
        # Command for Arducam motor on Bus 10, Address 0x0c
        os.system(f"i2cset -y 10 0x0c {dat1} {dat2}")
    except Exception as e:
        print(f"Focus Hardware Error: {e}")

def setup_camera():
    """Initializes and configures the Picamera2 instance."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={'size': RESOLUTION})
    picam2.configure(config)
    picam2.start()
    return picam2


# --- 3. DETECTION & RECORDING LOGIC ---
def get_detections(model, frame):
    """Runs YOLO and returns the label of the first target animal found."""
    # imgsz=320 makes detection much faster on a Raspberry Pi 4
    results = model(frame, verbose=False, imgsz=320)
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label in TARGET_ANIMALS:
                return label
    return None


def main():
    # Setup
    print("Initializing system...")
    set_manual_focus(WINNING_FOCUS)
    picam = setup_camera()
    model = YOLO('yolov8n.pt')
    
    is_recording = False
    last_seen_time = 0

    print(f"System Ready. Monitoring for: {', '.join(TARGET_ANIMALS)}")

    try:
        while True:
            # Capture and prepare frame
            raw_frame = picam.capture_array()
            # Convert RGBA to BGR for YOLO/OpenCV
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
            
            # Identify animals
            animal = get_detections(model, frame)
            
            if animal:
                last_seen_time = time.time()
                if not is_recording:
                    now = datetime.datetime.now()
                    filename = now.strftime(f"%Y%m%d-%H%M%S-{animal}.mp4")
                    print(f"DETECTION: {animal} | Starting record: {filename}")
                    picam.start_recording(filename)
                    is_recording = True
            
            # Handle stopping
            if is_recording and (time.time() - last_seen_time > COOLDOWN_SECONDS):
                print("Target left. Stopping recording.")
                picam.stop_recording()
                is_recording = False

    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        if is_recording:
            picam.stop_recording()
        picam.stop()


if __name__ == "__main__":
    main()