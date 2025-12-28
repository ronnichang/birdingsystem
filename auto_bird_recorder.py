#!/usr/bin/env python3
"""
auto_bird_recorder.py

- Uses Picamera2 dual-stream:
  - main: recording (YUV420 @ RESOLUTION)
  - lores: detection (YUV420 @ DETECT_RESOLUTION)
- Runs YOLOv8n on lores frames.
- Starts recording after START_CONFIRM_FRAMES consecutive strong detections.
- Stops after STOP_CONFIRM_FRAMES consecutive misses AND after MIN_RECORD_SECONDS.
- Restarts the camera pipeline after every stop (reliability on Pi/libcamera).
- Watchdog requests a camera restart if capture/inference blocks too long.
"""

import os
import cv2
import time
import datetime
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


# ---------------------------
# 1) CONFIG
# ---------------------------

@dataclass(frozen=True)
class Settings:
    # Focus
    enable_manual_focus: bool = False
    winning_focus: int = 500

    # Targets
    target_animals: Tuple[str, ...] = ("bird", "cat", "dog", "squirrel")

    # Recording control
    start_confirm_frames: int = 3
    stop_confirm_frames: int = 10
    min_record_seconds: float = 3.0
    post_record_cooldown_seconds: float = 0.0  # set >0 if you want a deadband after stopping

    # Confidence thresholds
    start_min_conf: float = 0.15
    keep_min_conf: float = 0.08

    # Camera / performance
    resolution: Tuple[int, int] = (1280, 720)       # main (recording)
    detect_resolution: Tuple[int, int] = (640, 360) # lores (detection)
    detect_fps: float = 8.0
    yolo_img_size: int = 320
    h264_bitrate: int = 8_000_000
    buffer_count: int = 6

    # Audio 
    audio_device: str = "plughw:1,0"   # from arecord -l => card 1, device 0
    audio_gain: float = 3.0            # boost quiet mic; set 1.0 to disable

    # Debug
    print_detection_progress: bool = True
    print_progress_every_n_frames: int = 8
    heartbeat_seconds: float = 1.0

    # Watchdog
    progress_timeout_seconds: float = 20.0

    # Output
    capture_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    yolo_weights: str = "yolov8n.pt"


CFG = Settings()
os.makedirs(CFG.capture_dir, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------
# 2) CAMERA FOCUS CONTROL (if camera is motorized)
# ---------------------------

def set_manual_focus(val: int) -> None:
    """Sets focus using system-level I2C commands via sudo (Arducam motor on bus 10, addr 0x0c)."""
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


# ---------------------------
# 3) CAMERA PIPELINE (CREATE/RESET)
# ---------------------------

class CameraManager:
    """Owns Picamera2 + encoder and can hard-restart the pipeline for reliability."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.picam: Optional[Picamera2] = None
        self.encoder: Optional[H264Encoder] = None

    def start(self) -> None:
        """Create + configure + start the camera and encoder."""
        self.picam = Picamera2()
        config = self.picam.create_video_configuration(
            main={"size": self.cfg.resolution, "format": "YUV420"},
            lores={"size": self.cfg.detect_resolution, "format": "YUV420"},
            buffer_count=self.cfg.buffer_count,
        )
        self.picam.configure(config)
        self.picam.start()
        self.encoder = H264Encoder(bitrate=self.cfg.h264_bitrate)

    def stop_close(self) -> None:
        """Best-effort stop/close without throwing."""
        if not self.picam:
            return
        try:
            try:
                self.picam.stop_recording()
            except Exception:
                pass
            try:
                self.picam.stop()
            except Exception:
                pass
            try:
                self.picam.close()
            except Exception:
                pass
        finally:
            self.picam = None
            self.encoder = None

    def restart(self, reason: str = "") -> None:
        """Hard reset camera pipeline (fixes rare post-record deadlock)."""
        if reason:
            log(f"[recovery] Restarting camera pipeline ({reason})...")
        else:
            log("[recovery] Restarting camera pipeline...")
        self.stop_close()
        time.sleep(0.3)
        self.start()
        log("[recovery] Camera restarted.")

    def capture_bgr_from_lores(self):
        """Capture from lores stream and convert to BGR for YOLO."""
        assert self.picam is not None
        raw = self.picam.capture_array("lores")

        # Rare case: already RGB
        if raw.ndim == 3 and raw.shape[2] == 3:
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        try:
            return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_I420)
        except cv2.error:
            return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12)


# ---------------------------
# 4) DETECTOR
# ---------------------------

class AnimalDetector:
    """Wrap YOLO + target filtering and returns best (label, conf)."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_weights)

    def best_target(self, frame_bgr) -> Tuple[Optional[str], float]:
        """
        Run YOLO and return (label, conf) for best target animal.
        YOLO runs with KEEP_MIN_CONF to reduce flicker while recording.
        """
        results = self.model(
            frame_bgr,
            verbose=False,
            imgsz=self.cfg.yolo_img_size,
            conf=self.cfg.keep_min_conf,
        )

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None, 0.0

        best_label = None
        best_conf = 0.0

        for box in r0.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            if label in self.cfg.target_animals and conf > best_conf:
                best_label = label
                best_conf = conf

        return best_label, best_conf


# ---------------------------
# 5) RECORDER
# ---------------------------

class Recorder:
    """
    Records video via Picamera2 and audio via arecord, then muxes them into one MP4.
    """
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.output: Optional[FfmpegOutput] = None
        self.record_start_time: float = 0.0

        self._arecord: Optional[subprocess.Popen] = None
        self._video_tmp: Optional[str] = None
        self._audio_tmp: Optional[str] = None
        self._final_path: Optional[str] = None

    def _start_audio(self, wav_path: str) -> None:
        # -f cd => 44100 Hz, S16_LE, stereo. Can change to -r 48000 if preferred.
        cmd = [
            "arecord",
            "-D", self.cfg.audio_device,
            "-f", "cd",
            "-t", "wav",
            wav_path,
        ]
        self._arecord = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop_audio(self) -> None:
        if self._arecord is None:
            return
        try:
            self._arecord.terminate()
            self._arecord.wait(timeout=2.0)
        except Exception:
            try:
                self._arecord.kill()
            except Exception:
                pass
        finally:
            self._arecord = None

    def make_filename(self, animal: str) -> str:
        now = datetime.datetime.now()
        base = now.strftime(f"%Y%m%d-%H%M%S-{animal}.mp4")
        return os.path.join(self.cfg.capture_dir, base)

    def start(self, cam: CameraManager, final_filename: str) -> None:
        """Start video and audio. Video goes to a temp mp4; final mp4 created at stop."""
        assert cam.picam is not None and cam.encoder is not None

        final_path = Path(final_filename)
        stem = final_path.stem  # YYYYMMDD-HHMMSS-animal

        video_tmp = str(final_path.with_name(stem + ".video.mp4"))
        audio_tmp = str(final_path.with_name(stem + ".audio.wav"))

        self._final_path = str(final_path)
        self._video_tmp = video_tmp
        self._audio_tmp = audio_tmp

        # Start audio first (so we don't miss the beginning)
        self._start_audio(audio_tmp)

        # Start video
        self.output = FfmpegOutput(video_tmp)
        cam.picam.start_recording(cam.encoder, self.output)

        self.record_start_time = time.time()

    def stop(self, cam: CameraManager) -> None:
        """Stop video+audio and mux into final mp4."""
        assert cam.picam is not None

        # Stop video
        cam.picam.stop_recording()
        if self.output is not None:
            try:
                self.output.close()
            except Exception:
                pass
        self.output = None

        # Stop audio
        self._stop_audio()

        if not (self._video_tmp and self._audio_tmp and self._final_path):
            return

        # Mux video + audio
        # -c:v copy => no re-encode video (fast)
        # -c:a aac => encode audio into mp4
        # -shortest => stop at the shorter stream
        # optional: boost audio volume if the mic is quiet
        af = f"volume={self.cfg.audio_gain}" if self.cfg.audio_gain and self.cfg.audio_gain != 1.0 else "anull"

        cmd = [
            "ffmpeg", "-y",
            "-i", self._video_tmp,
            "-i", self._audio_tmp,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", af,
            "-shortest",
            self._final_path,
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Cleanup temp files
        for p in (self._video_tmp, self._audio_tmp):
            try:
                os.remove(p)
            except Exception:
                pass

        self._video_tmp = None
        self._audio_tmp = None
        self._final_path = None


# ---------------------------
# 6) WATCHDOG
# ---------------------------

class ProgressWatchdog:
    """
    Watchdog thread:
    - If loop progress stalls, request a camera restart (via event).
    """

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._last_progress = time.monotonic()
        self.restart_flag = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def touch(self) -> None:
        with self._lock:
            self._last_progress = time.monotonic()

    def _seconds_since_progress(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_progress

    def _run(self) -> None:
        while True:
            time.sleep(1.0)
            since = self._seconds_since_progress()
            if since > self.timeout_seconds:
                log(f"[watchdog] No progress for {since:.1f}s. Requesting camera restart...")
                self.restart_flag.set()


# ---------------------------
# 7) STATE MACHINE
# ---------------------------

@dataclass
class RuntimeState:
    is_recording: bool = False

    # Start gating
    consecutive_hits: int = 0
    pending_label: Optional[str] = None

    # Stop gating
    misses_while_recording: int = 0

    # Cooldown
    next_start_allowed_time: float = 0.0

    # Heartbeat
    last_heartbeat: float = time.monotonic()
    cooldown_announced: bool = False

    # Debug frame counter
    frame_count: int = 0


def should_print_progress(cfg: Settings, state: RuntimeState) -> bool:
    return cfg.print_detection_progress and (state.frame_count % max(1, cfg.print_progress_every_n_frames) == 0)


def handle_heartbeat(cfg: Settings, state: RuntimeState, now_m: float, now: float) -> None:
    if state.is_recording:
        return

    if (now_m - state.last_heartbeat) < cfg.heartbeat_seconds:
        return

    wait_left = max(0.0, state.next_start_allowed_time - now)
    if wait_left > 0:
        log(f"[standby] Alive. Cooldown {wait_left:.1f}s remaining...")
        state.cooldown_announced = True
    else:
        if state.cooldown_announced:
            log("[standby] Cooldown ended. Ready to trigger.")
            state.cooldown_announced = False
        else:
            log("[standby] Alive. Ready to trigger.")

    state.last_heartbeat = now_m


def in_post_stop_cooldown(state: RuntimeState, now: float) -> bool:
    return now < state.next_start_allowed_time


def start_logic(cfg: Settings, state: RuntimeState, animal: str, conf: float) -> bool:
    """
    Update gating counters and return True if we should start recording now.
    Only strong detections count toward start.
    """
    if conf < cfg.start_min_conf:
        state.consecutive_hits = 0
        state.pending_label = None
        return False

    if state.pending_label == animal:
        state.consecutive_hits += 1
    else:
        state.pending_label = animal
        state.consecutive_hits = 1

    return state.consecutive_hits >= cfg.start_confirm_frames


def stop_logic(cfg: Settings, recorder: Recorder, state: RuntimeState, now: float, animal: Optional[str]) -> bool:
    """
    Update miss counter and return True if we should stop now.
    Any detection keeps recording alive. Stop after N misses and min duration elapsed.
    """
    if animal:
        state.misses_while_recording = 0
        return False

    state.misses_while_recording += 1

    elapsed = now - recorder.record_start_time
    if elapsed < cfg.min_record_seconds:
        return False

    return state.misses_while_recording >= cfg.stop_confirm_frames


# ---------------------------
# 8) APP / MAIN LOOP
# ---------------------------

class BirdingApp:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.cam = CameraManager(cfg)
        self.detector = AnimalDetector(cfg)
        self.recorder = Recorder(cfg)
        self.state = RuntimeState()
        self.watchdog = ProgressWatchdog(cfg.progress_timeout_seconds)

    def setup(self) -> None:
        log("Initializing birding system...")
        if self.cfg.enable_manual_focus:
            set_manual_focus(self.cfg.winning_focus)

        self.cam.start()
        self.watchdog.start()

        log(f"System Ready. Monitoring for: {', '.join(self.cfg.target_animals)}")
        log(
            f"Start: {self.cfg.start_confirm_frames} hits @ conf>={self.cfg.start_min_conf} | "
            f"Stop: {self.cfg.stop_confirm_frames} misses | "
            f"Keep conf>={self.cfg.keep_min_conf} | "
            f"MIN_RECORD_SECONDS={self.cfg.min_record_seconds} | "
            f"POST_RECORD_COOLDOWN_SECONDS={self.cfg.post_record_cooldown_seconds} | "
            f"DETECT_FPS={self.cfg.detect_fps}"
        )
        log("[standby] Waiting for animals...")

    def restart_camera_now(self, reason: str) -> None:
        # Ensure we are not recording
        if self.state.is_recording:
            try:
                self.recorder.stop(self.cam)
            except Exception:
                pass
        self.state.is_recording = False
        self.recorder.output = None
        self.state.misses_while_recording = 0
        self.state.consecutive_hits = 0
        self.state.pending_label = None

        self.cam.restart(reason=reason)
        self.watchdog.restart_flag.clear()

    def start_recording(self, animal: str) -> None:
        filename = self.recorder.make_filename(animal)
        log(f"DETECTION CONFIRMED: {animal.upper()} x{self.state.consecutive_hits} | Starting record: {filename}")
        self.recorder.start(self.cam, filename)
        self.state.is_recording = True
        self.state.misses_while_recording = 0
        self.state.consecutive_hits = 0
        self.state.pending_label = None

    def stop_recording(self) -> None:
        log(f"Target likely gone (misses={self.state.misses_while_recording}). Stopping recording.")
        self.recorder.stop(self.cam)

        # Re-arm standby
        self.state.is_recording = False
        self.state.misses_while_recording = 0
        self.state.consecutive_hits = 0
        self.state.pending_label = None
        self.state.next_start_allowed_time = time.time() + self.cfg.post_record_cooldown_seconds

        log("[standby] Ready for next animal...")

        # IMPORTANT: hard reset after every stop (your preference)
        self.cam.restart(reason="post-record restart")

    def tick(self) -> None:
        """One iteration of the event loop."""
        # Watchdog requested a restart?
        if self.watchdog.restart_flag.is_set():
            self.restart_camera_now("watchdog stall")

        self.watchdog.touch()
        self.state.frame_count += 1

        now_m = time.monotonic()
        now = time.time()

        handle_heartbeat(self.cfg, self.state, now_m, now)

        # Capture + detect (common stall point)
        frame = self.cam.capture_bgr_from_lores()
        self.watchdog.touch()
        animal, conf = self.detector.best_target(frame)
        self.watchdog.touch()

        # Debug print
        if animal and should_print_progress(self.cfg, self.state):
            if self.state.is_recording:
                log(f"[keep] {animal} conf={conf:.2f} (recording)")
            else:
                gate = "strong" if conf >= self.cfg.start_min_conf else "weak"
                log(f"[detect-{gate}] {animal} conf={conf:.2f}")

        # Recording state machine
        if self.state.is_recording:
            if stop_logic(self.cfg, self.recorder, self.state, now, animal):
                self.stop_recording()
        else:
            if in_post_stop_cooldown(self.state, now):
                # Don't accumulate hits in cooldown
                self.state.consecutive_hits = 0
                self.state.pending_label = None
            else:
                if animal:
                    if start_logic(self.cfg, self.state, animal, conf):
                        self.start_recording(animal)
                else:
                    self.state.consecutive_hits = 0
                    self.state.pending_label = None

        time.sleep(1.0 / self.cfg.detect_fps)

    def run_forever(self) -> None:
        self.setup()
        while True:
            self.tick()

    def shutdown(self) -> None:
        log("\nShutting down safely...")
        try:
            if self.state.is_recording:
                self.recorder.stop(self.cam)
        except Exception:
            pass
        self.cam.stop_close()


def main():
    app = BirdingApp(CFG)
    try:
        app.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
