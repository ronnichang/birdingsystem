import os
import cv2
import time
import datetime
import threading
from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


# --- 1. CONFIGURATION ---
WINNING_FOCUS = 500
TARGET_ANIMALS = ['bird', 'cat', 'dog', 'squirrel']

# Recording control
START_CONFIRM_FRAMES = 3            # N consecutive detections required to START recording
STOP_CONFIRM_FRAMES = 10           # N consecutive misses required to STOP recording (hysteresis)
MIN_RECORD_SECONDS = 3.0           # never stop before this many seconds after starting
POST_RECORD_COOLDOWN_SECONDS = 0.0 # cooldown after STOP before allowing next START

# Confidence thresholds
START_MIN_CONF = 0.15              # must be >= this to count toward START_CONFIRM_FRAMES
KEEP_MIN_CONF = 0.08               # YOLO runs with this to reduce flicker while recording

# Camera / performance
ENABLE_MANUAL_FOCUS = False
RESOLUTION = (1280, 720)           # main stream (recording)
DETECT_RESOLUTION = (640, 360)     # lores stream (detection)
DETECT_FPS = 8                     # throttle detection loop
YOLO_IMG_SIZE = 320
H264_BITRATE = 8_000_000
BUFFER_COUNT = 6                   # slightly higher can reduce stalls

# Debug / visibility
PRINT_DETECTION_PROGRESS = True
PRINT_PROGRESS_EVERY_N_FRAMES = 1
HEARTBEAT_SECONDS = 1.0

# Watchdog: if capture/inference blocks and no progress is made, restart camera
PROGRESS_TIMEOUT_SECONDS = 20.0

# Output
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


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
            log("Focus Hardware Warning: i2cset returned non-zero (bus/address may be wrong).")
    except Exception as e:
        log(f"Focus Hardware Error: {e}")


def create_and_start_camera() -> Picamera2:
    """Create + configure + start Picamera2 with main+lores streams."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={'size': RESOLUTION, 'format': 'YUV420'},
        lores={'size': DETECT_RESOLUTION, 'format': 'YUV420'},
        buffer_count=BUFFER_COUNT,
    )
    picam2.configure(config)
    picam2.start()
    return picam2


def safe_stop_close_camera(picam: Picamera2) -> None:
    """Best-effort stop/close without throwing."""
    try:
        try:
            picam.stop_recording()
        except Exception:
            pass
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass
    except Exception:
        pass


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
    YOLO runs with KEEP_MIN_CONF to reduce flicker while recording.
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


def stop_recording(picam: Picamera2, output: FfmpegOutput | None) -> None:
    """Stop recording and close ffmpeg output."""
    picam.stop_recording()
    if output is not None:
        try:
            output.close()
        except Exception:
            pass


# --- 6. MAIN LOOP ---
def run_event_loop(model: YOLO) -> None:
    cooldown_announced = False
    is_recording = False

    # Start gating (N consecutive strong hits)
    consecutive_hits = 0
    pending_label = None

    # Stop hysteresis (N consecutive misses while recording)
    misses_while_recording = 0
    record_start_time = 0.0

    # Recording output handle (close cleanly)
    current_output: FfmpegOutput | None = None

    # Standby cooldown after stop
    next_start_allowed_time = 0.0

    # Create camera + encoder
    picam = create_and_start_camera()
    encoder = H264Encoder(bitrate=H264_BITRATE)

    # Heartbeat + watchdog progress
    frame_count = 0
    last_heartbeat = time.monotonic()

    last_progress = time.monotonic()
    progress_lock = threading.Lock()
    restart_camera_flag = threading.Event()

    def touch_progress():
        nonlocal last_progress
        with progress_lock:
            last_progress = time.monotonic()

    def get_last_progress():
        with progress_lock:
            return last_progress

    def watchdog():
        while True:
            time.sleep(1.0)
            since = time.monotonic() - get_last_progress()
            if since > PROGRESS_TIMEOUT_SECONDS:
                log(f"[watchdog] No progress for {since:.1f}s. Requesting camera restart...")
                restart_camera_flag.set()

    threading.Thread(target=watchdog, daemon=True).start()

    def restart_camera():
        """Hard reset camera pipeline (fixes rare post-record deadlock)."""
        nonlocal picam, encoder, is_recording, current_output
        log("[recovery] Restarting camera pipeline...")
        try:
            if is_recording:
                try:
                    stop_recording(picam, current_output)
                except Exception:
                    pass
            current_output = None
            is_recording = False
        except Exception:
            pass

        safe_stop_close_camera(picam)
        time.sleep(0.3)

        picam = create_and_start_camera()
        encoder = H264Encoder(bitrate=H264_BITRATE)
        log("[recovery] Camera restarted.")
        restart_camera_flag.clear()

    log(f"System Ready. Monitoring for: {', '.join(TARGET_ANIMALS)}")
    log(
        f"Start: {START_CONFIRM_FRAMES} hits @ conf>={START_MIN_CONF} | "
        f"Stop: {STOP_CONFIRM_FRAMES} misses | "
        f"Keep conf>={KEEP_MIN_CONF} | "
        f"MIN_RECORD_SECONDS={MIN_RECORD_SECONDS} | "
        f"POST_RECORD_COOLDOWN_SECONDS={POST_RECORD_COOLDOWN_SECONDS} | "
        f"DETECT_FPS={DETECT_FPS}"
    )
    log("[standby] Waiting for animals...")

    while True:
        # If watchdog asked for restart, do it here (main thread)
        if restart_camera_flag.is_set():
            restart_camera()

        touch_progress()
        frame_count += 1
        now_m = time.monotonic()
        now = time.time()

        # Heartbeat
        if (now_m - last_heartbeat) >= HEARTBEAT_SECONDS and not is_recording:
            wait_left = max(0.0, next_start_allowed_time - now)
            if wait_left > 0:
                log(f"[standby] Alive. Cooldown {wait_left:.1f}s remaining...")
                cooldown_announced = True
            else:
                if cooldown_announced:
                    log("[standby] Cooldown ended. Ready to trigger.")
                    cooldown_announced = False
                else:
                    log("[standby] Alive. Ready to trigger.")
            last_heartbeat = now_m

        # Capture + detection (can block => watchdog will request restart)
        frame = capture_bgr_frame(picam)
        touch_progress()
        animal, conf = get_best_detection(model, frame)
        touch_progress()

        # Print detections
        if PRINT_DETECTION_PROGRESS and (animal is not None) and (frame_count % PRINT_PROGRESS_EVERY_N_FRAMES == 0):
            if not is_recording:
                gate = "strong" if conf >= START_MIN_CONF else "weak"
                log(f"[detect-{gate}] {animal} conf={conf:.2f}")
            else:
                log(f"[keep] {animal} conf={conf:.2f} (recording)")

        # --- While recording ---
        if is_recording:
            if animal:
                misses_while_recording = 0
            else:
                misses_while_recording += 1

            elapsed = now - record_start_time
            if elapsed >= MIN_RECORD_SECONDS and misses_while_recording >= STOP_CONFIRM_FRAMES:
                log(f"Target likely gone (misses={misses_while_recording}). Stopping recording.")
                stop_recording(picam, current_output)
                current_output = None

                # Re-arm standby
                is_recording = False
                misses_while_recording = 0
                consecutive_hits = 0
                pending_label = None
                next_start_allowed_time = time.time() + POST_RECORD_COOLDOWN_SECONDS

                log("[standby] Ready for next animal...")

                # IMPORTANT: hard-reset camera after each recording stop (prevents deadlock)
                restart_camera()

            time.sleep(1.0 / DETECT_FPS)
            continue

        # --- Not recording: cooldown gate ---
        if now < next_start_allowed_time:
            consecutive_hits = 0
            pending_label = None
            time.sleep(1.0 / DETECT_FPS)
            continue

        # --- Not recording: start gating with strong detections ---
        if animal and conf >= START_MIN_CONF:
            if pending_label == animal:
                consecutive_hits += 1
            else:
                pending_label = animal
                consecutive_hits = 1

            if consecutive_hits >= START_CONFIRM_FRAMES:
                filename = make_filename(animal)
                log(f"DETECTION CONFIRMED: {animal.upper()} x{consecutive_hits} | Starting record: {filename}")
                current_output = start_recording(picam, encoder, filename)
                is_recording = True
                record_start_time = time.time()
                misses_while_recording = 0
                consecutive_hits = 0
                pending_label = None
        else:
            consecutive_hits = 0
            pending_label = None

        time.sleep(1.0 / DETECT_FPS)


def main():
    log("Initializing birding system...")
    if ENABLE_MANUAL_FOCUS:
        set_manual_focus(WINNING_FOCUS)

    model = YOLO('yolov8n.pt')

    try:
        run_event_loop(model)
    except KeyboardInterrupt:
        log("\nShutting down safely...")


if __name__ == "__main__":
    main()
