import cv2
import datetime
import time
import smbus
from ultralytics import YOLO
from picamera2 import Picamera2

# --- ARDUCAM MOTORIZED FOCUS HELPER ---
class ArducamFocus:
    def __init__(self, bus_num=10):
        # Bus 10 is standard for Pi 4 camera i2c
        try:
            self.bus = smbus.SMBus(bus_num)
        except:
            self.bus = None
            print("Warning: i2c bus not found. Motorized focus disabled.")

    def set_focus(self, val):
        # Value should be 0 (infinity) to 1023 (near)
        if self.bus:
            val = max(0, min(1023, val))
            data = [(val >> 4) & 0xff, (val << 4) & 0xff]
            try:
                self.bus.write_i2c_block_data(0x0c, data[0], [data[1]])
            except Exception as e:
                print(f"Focus Error: {e}")

# --- MAIN PROJECT LOGIC ---
# Initialize YOLO (Nano version for Pi performance)
model = YOLO('yolov8n.pt')

# Initialize Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={'size': (1280, 720)})
picam2.configure(config)
picam2.start()

# Initialize Focus (Set to a middle value like 500 for testing)
focus_ctrl = ArducamFocus()
focus_ctrl.set_focus(500) 

TARGET_CLASSES = ['bird', 'cat', 'dog', 'squirrel'] 
COOLDOWN_TIME = 3.0  # Seconds to wait after animal leaves before stopping
is_recording = False
last_seen_time = 0

print("Monitoring balcony. System Ready.")

try:
    while True:
        # 1. Capture frame
        raw_frame = picam2.capture_array()
        
        # 2. Convert RGBA -> BGR (Fixes the 4-channel error and color swap)
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
        
        # 3. AI Inference
        results = model(frame, verbose=False, imgsz=320) # Low imgsz = faster Pi performance
        
        detected_animal = None
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in TARGET_CLASSES:
                    detected_animal = label
                    break
        
        # 4. Recording Logic
        if detected_animal:
            last_seen_time = time.time()
            if not is_recording:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{timestamp}-{detected_animal}.mp4"
                print(f"!!! {detected_animal.upper()} DETECTED !!! Recording to {filename}")
                picam2.start_recording(filename, format="mp4")
                is_recording = True
        
        # 5. Stop Logic
        if is_recording and (time.time() - last_seen_time > COOLDOWN_TIME):
            print("Target left field of view. Stopping recording.")
            picam2.stop_recording()
            is_recording = False

except KeyboardInterrupt:
    if is_recording:
        picam2.stop_recording()
    picam2.stop()
    print("\nSystem shut down safely.")