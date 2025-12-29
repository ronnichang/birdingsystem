#!/usr/bin/env python3
"""
demo_headless_overlay.py

Headless demo recorder that:
- imports and reuses most of auto_bird_recorder.py
- runs the same detection/recording state machine
- produces ONE continuous demo MP4 with overlays burned in:
  - heartbeat/status text
  - detection label/confidence
  - flashing REC indicator while recording
"""

import os
import time
import datetime
import subprocess

import cv2
from sympy import ff

# Reuse your existing implementation
import auto_bird_recorder as abr


# ---------------------------
# Demo-specific config (only)
# ---------------------------

# Output demo video resolution (for nicer viewing); frames are upscaled from lores.
DEMO_OUTPUT_RESOLUTION = (1280, 720)

# Demo output fps: usually match detection fps.
DEMO_FPS = int(round(abr.CFG.detect_fps))

# Flashing indicator period (seconds)
REC_FLASH_PERIOD = 0.5

# Heartbeat overlay update rate (seconds) – independent of abr.heartbeat_seconds printing
OVERLAY_HEARTBEAT_SECONDS = 1.0


def start_ffmpeg_writer(out_path: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """
    Start ffmpeg process to accept raw BGR frames on stdin and write an MP4.
    IMPORTANT: Don't PIPE stderr unless you read it (ffmpeg writes progress to stderr).
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-nostats",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",

        # Better for “recording while running”; still fine after Ctrl+C
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",

        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def overlay_hud(frame_bgr, lines, is_recording: bool, rec_flash_on: bool):
    """
    Draw HUD overlay onto a BGR image.
    Uses a semi-transparent panel instead of an opaque black strip.
    Also draws State line in red when recording.
    """
    h, w = frame_bgr.shape[:2]

    # Semi-transparent HUD panel
    panel_h = 165
    x1, y1 = 10, 10
    x2, y2 = w - 10, min(h - 10, y1 + panel_h)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # alpha: 0 = fully transparent, 1 = fully opaque
    alpha = 0.35
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    # Draw text (State line color changes)
    y = y1 + 32
    for s in lines:
        if s.startswith("State:"):
            color = (0, 0, 255) if is_recording else (255, 255, 255)  # red if recording
            thickness = 3 if is_recording else 2
        else:
            color = (255, 255, 255)
            thickness = 2

        cv2.putText(frame_bgr, s, (x1 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
        y += 30

    # Flashing REC indicator
    if is_recording and rec_flash_on:
        cv2.putText(frame_bgr, "REC", (w - 160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.circle(frame_bgr, (w - 55, 45), 12, (0, 0, 255), -1)

    return frame_bgr


def build_overlay_lines(cfg: abr.Settings, state: abr.RuntimeState, animal, conf, overlay_heartbeat: str) -> list[str]:
    return [
        f"Heartbeat: {overlay_heartbeat}",
        f"Detect: {animal or 'none'}   conf={conf:.2f}",
        f"State: {'RECORDING' if state.is_recording else 'STANDBY'}",
        f"Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Start gate: {cfg.start_confirm_frames} @ conf>={cfg.start_min_conf:.2f} | "
        f"Stop gate: {cfg.stop_confirm_frames} misses | MinRec: {cfg.min_record_seconds:.1f}s",
    ]


def close_ffmpeg(ff: subprocess.Popen, timeout: float = 8.0) -> None:
    """Close ffmpeg cleanly so MP4 is finalized (moov written)."""
    if ff is None:
        return

    try:
        if ff.stdin:
            try:
                ff.stdin.flush()
            except Exception:
                pass
            try:
                ff.stdin.close()
            except Exception:
                pass

        try:
            ff.wait(timeout=timeout)
            return
        except Exception:
            pass

        # If it didn't exit, terminate then kill.
        try:
            ff.terminate()
            ff.wait(timeout=3.0)
            return
        except Exception:
            pass

        try:
            ff.kill()
        except Exception:
            pass
    finally:
        pass


def demo_loop(cfg: abr.Settings) -> None:
    # Demo output path
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    demo_path = os.path.join(cfg.capture_dir, f"demo-{stamp}.mp4")
    os.makedirs(cfg.capture_dir, exist_ok=True)

    abr.log("Starting headless overlay demo...")
    abr.log(f"Demo output: {demo_path}")
    abr.log("(This creates ONE continuous MP4 with HUD overlays burned in.)")

    # Reuse your components
    cam = abr.CameraManager(cfg)
    detector = abr.AnimalDetector(cfg)
    recorder = abr.Recorder(cfg)
    watchdog = abr.ProgressWatchdog(cfg.progress_timeout_seconds)
    state = abr.RuntimeState()

    if cfg.enable_manual_focus:
        abr.set_manual_focus(cfg.winning_focus)

    cam.start()
    watchdog.start()

    # ffmpeg writer for demo MP4
    out_w, out_h = DEMO_OUTPUT_RESOLUTION
    ff = start_ffmpeg_writer(demo_path, out_w, out_h, DEMO_FPS)
    if ff.stdin is None:
        raise RuntimeError("Failed to start ffmpeg (stdin is None).")

    # Overlay timing
    last_overlay_heartbeat = time.time()
    overlay_heartbeat = "Standby: ready"
    last_flash = time.time()
    rec_flash_on = True

    try:
        while True:
            loop_start = time.time()

            # Watchdog requested restart?
            if watchdog.restart_flag.is_set():
                abr.log("[demo] Watchdog requested restart.")
                # Ensure not recording
                if state.is_recording:
                    try:
                        recorder.stop(cam)
                    except Exception:
                        pass
                state.is_recording = False
                recorder.output = None
                state.misses_while_recording = 0
                state.consecutive_hits = 0
                state.pending_label = None
                cam.restart(reason="watchdog stall")
                watchdog.restart_flag.clear()

            watchdog.touch()
            state.frame_count += 1
            now_m = time.monotonic()
            now = time.time()

            # Keep your terminal heartbeat behavior (optional, uses abr.handle_heartbeat)
            abr.handle_heartbeat(cfg, state, now_m, now)

            # Capture + detect using your exact code paths
            frame_lores_bgr = cam.capture_bgr_from_lores()
            watchdog.touch()
            animal, conf = detector.best_target(frame_lores_bgr)
            watchdog.touch()

            # Your existing terminal debug prints (optional)
            if animal and abr.should_print_progress(cfg, state):
                if state.is_recording:
                    abr.log(f"[keep] {animal} conf={conf:.2f} (recording)")
                else:
                    gate = "strong" if conf >= cfg.start_min_conf else "weak"
                    abr.log(f"[detect-{gate}] {animal} conf={conf:.2f}")

            # Update overlay heartbeat once/sec
            if (time.time() - last_overlay_heartbeat) >= OVERLAY_HEARTBEAT_SECONDS:
                overlay_heartbeat = "Recording..." if state.is_recording else "Standby: ready"
                last_overlay_heartbeat = time.time()

            # Flash toggle
            if (time.time() - last_flash) >= REC_FLASH_PERIOD:
                rec_flash_on = not rec_flash_on
                last_flash = time.time()

            # ---- Recording state machine (REUSE your helpers) ----
            if state.is_recording:
                if abr.stop_logic(cfg, recorder, state, now, animal):
                    abr.log(f"Target likely gone (misses={state.misses_while_recording}). Stopping recording.")
                    recorder.stop(cam)

                    # Re-arm standby
                    state.is_recording = False
                    state.misses_while_recording = 0
                    state.consecutive_hits = 0
                    state.pending_label = None
                    state.next_start_allowed_time = time.time() + cfg.post_record_cooldown_seconds

                    abr.log("[standby] Ready for next animal...")

                    # IMPORTANT: restart after every stop (your preference)
                    cam.restart(reason="post-record restart")
            else:
                if abr.in_post_stop_cooldown(state, now):
                    state.consecutive_hits = 0
                    state.pending_label = None
                else:
                    if animal:
                        if abr.start_logic(cfg, state, animal, conf):
                            filename = recorder.make_filename(animal)
                            abr.log(f"DETECTION CONFIRMED: {animal.upper()} x{state.consecutive_hits} | Starting record: {filename}")
                            recorder.start(cam, filename)
                            state.is_recording = True
                            state.misses_while_recording = 0
                            state.consecutive_hits = 0
                            state.pending_label = None
                    else:
                        state.consecutive_hits = 0
                        state.pending_label = None

            # ---- Create demo overlay frame and write to mp4 ----
            vis = cv2.resize(frame_lores_bgr, DEMO_OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            lines = build_overlay_lines(cfg, state, animal, conf, overlay_heartbeat)
            vis = overlay_hud(vis, lines, state.is_recording, rec_flash_on)
            try:
                ff.stdin.write(vis.tobytes())
            except BrokenPipeError:
                abr.log("[demo] ffmpeg pipe broke (ffmpeg exited). Stopping demo loop.")
                break

            # Throttle to demo fps
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / DEMO_FPS) - elapsed))

    except KeyboardInterrupt:
        abr.log("\nStopping demo (Ctrl+C).")

    finally:
        # Close ffmpeg cleanly
        close_ffmpeg(ff)


def main():
    demo_loop(abr.CFG)


if __name__ == "__main__":
    main()
