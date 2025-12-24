import os
import cv2
import time
import datetime
from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


# --- 1. CONFIGURATION ---
WINNING_FOCUS = 500
TARGET_ANIMALS = ['bird', 'cat', 'dog', 'squirrel']
COOLDOWN_SECONDS = 3.0
RESOLUTION = (1280, 720)
YOLO_IMG_SIZE = 320
H264_BITRATE = 8_000_000
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)


# --- 2. HARDWARE CONTROL FUNCTIONS ---
def set_manual_focus(val: int) -> None:
    """Sets focus using system-level I2C commands via sudo."""
    try:
        time.sleep(0.5)
        value = (val << 4) & 0x3ff0
        dat1 = (value >> 8) & 0x3f
        dat2 = value & 0xf0
        ret = os.system(f"sudo i2cset -y 10 0x0c {dat1} {dat2}")
        if ret != 0:
            print("Focus Hardware Warning: i2cset returned non-zero (bus/address may be wrong).")
    except Exception as e:
        print(f"Focus Hardware Error: {e}")


def setup_camera() -> Picamera2:
    """Initializes Picamera2 with the specified resolution."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={'size': RESOLUTION, 'format': 'YUV420'}  # lighter than XBGR8888
    )
    picam2.configure(config)
    picam2.start()
    return picam2


# --- 3. CAPTURE / CONVERT ---
def capture_bgr_frame(picam: Picamera2):
    """
    Capture a frame from Picamera2 and return a BGR image for OpenCV/YOLO.
    Handles both RGBA and YUV camera outputs.
    """
    raw_frame = picam.capture_array()

    # RGBA -> BGR
    if raw_frame.ndim == 3 and raw_frame.shape[2] == 4:
        return cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)

    # YUV420 -> BGR (try I420 then NV12)
    try:
        return cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_I420)
    except cv2.error:
        return cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_NV12)


# --- 4. DETECTION ---
def get_detections(model: YOLO, frame_bgr):
    """Processes frame with YOLO and returns the first detected target label (or None)."""
    results = model(frame_bgr, verbose=False, imgsz=YOLO_IMG_SIZE)
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label in TARGET_ANIMALS:
                return label
    return None


# --- 5. RECORDING HELPERS ---
def make_filename(animal: str) -> str:
    now = datetime.datetime.now()
    base = now.strftime(f"%Y%m%d-%H%M%S-{animal}.mp4")
    return os.path.join(CAPTURE_DIR, base)


def start_recording(picam: Picamera2, encoder: H264Encoder, filename: str) -> FfmpegOutput:
    output = FfmpegOutput(filename)
    picam.start_recording(encoder, output)
    return output


def stop_recording(picam: Picamera2) -> None:
    picam.stop_recording()


def should_stop_recording(is_recording: bool, last_seen_time: float) -> bool:
    if not is_recording:
        return False
    return (time.time() - last_seen_time) > COOLDOWN_SECONDS


# --- 6. MAIN LOOP ---
def run_event_loop(picam: Picamera2, model: YOLO, encoder: H264Encoder) -> None:
    is_recording = False
    last_seen_time = 0.0

    print(f"System Ready. Monitoring for: {', '.join(TARGET_ANIMALS)}")

    while True:
        frame = capture_bgr_frame(picam)
        animal = get_detections(model, frame)

        if animal:
            last_seen_time = time.time()
            if not is_recording:
                filename = make_filename(animal)
                print(f"DETECTION: {animal.upper()} | Starting record: {filename}")
                start_recording(picam, encoder, filename)
                is_recording = True

        if should_stop_recording(is_recording, last_seen_time):
            print("Target left field of view. Stopping recording.")
            stop_recording(picam)
            is_recording = False


def main():
    print("Initializing birding system...")
    set_manual_focus(WINNING_FOCUS)

    picam = setup_camera()
    encoder = H264Encoder(bitrate=H264_BITRATE)
    model = YOLO('yolov8n.pt')

    try:
        run_event_loop(picam, model, encoder)
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        # If recording is active, stop_recording() would be needed,
        # but Picamera2 is generally safe to stop; keep it conservative:
        try:
            picam.stop_recording()
        except Exception:
            pass
        picam.stop()


if __name__ == "__main__":
    main()
