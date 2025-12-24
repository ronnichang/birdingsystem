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

# Recording control
START_CONFIRM_FRAMES = 3          # N consecutive detections required to START recording
STOP_CONFIRM_FRAMES = 10          # N consecutive misses required to STOP recording (hysteresis)
MIN_RECORD_SECONDS = 3.0          # never stop before this many seconds after starting

# Confidence thresholds
START_MIN_CONF = 0.15             # must be >= this to count toward START_CONFIRM_FRAMES
KEEP_MIN_CONF = 0.08              # while running YOLO, keep boxes down to this conf to avoid flicker-stops

# Camera / performance
ENABLE_MANUAL_FOCUS = False
RESOLUTION = (1280, 720)          # main stream (recording)
DETECT_RESOLUTION = (640, 360)    # lores stream (detection)
DETECT_FPS = 8                    # throttle detection loop
YOLO_IMG_SIZE = 320
H264_BITRATE = 8_000_000

# Debug
PRINT_DETECTION_PROGRESS = True
PRINT_PROGRESS_EVERY_N_FRAMES = 8

# Output
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
    """Initializes Picamera2 with main(for recording) + lores(for detection)."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={'size': RESOLUTION, 'format': 'YUV420'},
        lores={'size': DETECT_RESOLUTION, 'format': 'YUV420'},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    return picam2


# --- 3. CAPTURE / CONVERT ---
def capture_bgr_frame(picam: Picamera2):
    """Capture from lores stream and convert to BGR for YOLO."""
    raw = picam.capture_array("lores")

    # Rare case: already RGB
    if raw.ndim == 3 and raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    try:
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_I420)
    except cv2.error:
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12)


# --- 4. DETECTION ---
def get_best_detection(model: YOLO, frame_bgr):
    """
    Run YOLO and return (label, conf) for the best target animal.
    We run YOLO with KEEP_MIN_CONF to reduce flicker while recording.
    """
    results = model(frame_bgr, verbose=False, imgsz=YOLO_IMG_SIZE, conf=KEEP_MIN_CONF)

    best_label = None
    best_conf = 0.0

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None, 0.0

    for box in r0.boxes:
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0]) if box.conf is not None else 0.0

        if label in TARGET_ANIMALS and conf > best_conf:
            best_label = label
            best_conf = conf

    return best_label, best_conf


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


# --- 6. MAIN LOOP ---
def run_event_loop(picam: Picamera2, model: YOLO, encoder: H264Encoder) -> None:
    is_recording = False

    # Start gating (N consecutive hits)
    consecutive_hits = 0
    pending_label = None

    # Stop hysteresis (N consecutive misses while recording)
    misses_while_recording = 0
    record_start_time = 0.0

    frame_count = 0

    print(f"System Ready. Monitoring for: {', '.join(TARGET_ANIMALS)}")
    print(
        f"Start: {START_CONFIRM_FRAMES} hits @ conf>={START_MIN_CONF} | "
        f"Stop: {STOP_CONFIRM_FRAMES} misses | "
        f"Keep conf>={KEEP_MIN_CONF} | "
        f"MIN_RECORD_SECONDS={MIN_RECORD_SECONDS} | DETECT_FPS={DETECT_FPS}"
    )

    while True:
        frame_count += 1
        frame = capture_bgr_frame(picam)

        animal, conf = get_best_detection(model, frame)

        if animal:
            # Any valid detection (>= KEEP_MIN_CONF because of model(conf=...)) keeps recording alive
            if is_recording:
                misses_while_recording = 0

            # Starting logic only counts hits when conf is strong enough
            if not is_recording and conf >= START_MIN_CONF:
                if pending_label == animal:
                    consecutive_hits += 1
                else:
                    pending_label = animal
                    consecutive_hits = 1

                if PRINT_DETECTION_PROGRESS and (frame_count % PRINT_PROGRESS_EVERY_N_FRAMES == 0):
                    print(f"[detect] {animal} conf={conf:.2f} hits={consecutive_hits}/{START_CONFIRM_FRAMES}")

                if consecutive_hits >= START_CONFIRM_FRAMES:
                    filename = make_filename(animal)
                    print(f"DETECTION CONFIRMED: {animal.upper()} x{consecutive_hits} | Starting record: {filename}")
                    start_recording(picam, encoder, filename)
                    is_recording = True
                    record_start_time = time.time()
                    misses_while_recording = 0
                    consecutive_hits = 0
                    pending_label = None

            elif not is_recording:
                # Weak/conf-flicker detection: do not count toward start
                if PRINT_DETECTION_PROGRESS and (frame_count % PRINT_PROGRESS_EVERY_N_FRAMES == 0):
                    print(f"[weak] {animal} conf={conf:.2f} (needs >= {START_MIN_CONF} to start)")
                # reset start gating so we require stable confident detections
                consecutive_hits = 0
                pending_label = None

        else:
            if is_recording:
                misses_while_recording += 1
            else:
                consecutive_hits = 0
                pending_label = None

        # Stop condition: misses, but never stop too early
        if is_recording:
            elapsed = time.time() - record_start_time
            if elapsed >= MIN_RECORD_SECONDS and misses_while_recording >= STOP_CONFIRM_FRAMES:
                print(f"Target likely gone (misses={misses_while_recording}). Stopping recording.")
                stop_recording(picam)
                is_recording = False
                misses_while_recording = 0
                consecutive_hits = 0
                pending_label = None

        time.sleep(1.0 / DETECT_FPS)


def main():
    print("Initializing birding system...")
    if ENABLE_MANUAL_FOCUS:
        set_manual_focus(WINNING_FOCUS)

    picam = setup_camera()
    encoder = H264Encoder(bitrate=H264_BITRATE)
    model = YOLO('yolov8n.pt')

    try:
        run_event_loop(picam, model, encoder)
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        try:
            picam.stop_recording()
        except Exception:
            pass
        picam.stop()


if __name__ == "__main__":
    main()
