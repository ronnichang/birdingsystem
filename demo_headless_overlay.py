#!/usr/bin/env python3
"""
demo_headless_overlay.py (overlay + real recordings)

Goal:
- Keep the on-screen proof HUD (heartbeat + state + flashing REC).
- ALSO behave like auto_bird_recorder.py: when an animal is detected,
  start a real recording (video+audio muxed) and stop it when gone.
- IMPORTANT: We do NOT record continuous demo audio, because the USB mic
  can't be opened by two recorders at once. The *real per-animal clips*
  will have audio (the class Recorder uses arecord).
"""

import os
import time
import datetime
import subprocess
import cv2
import signal
import threading

import auto_bird_recorder as abr


# ---------------------------
# Demo-only settings
# ---------------------------

DEMO_OUTPUT_RESOLUTION = (1280, 720)  # HUD video size
DEMO_FPS = int(round(abr.CFG.detect_fps))
REC_FLASH_PERIOD = 0.5
OVERLAY_HEARTBEAT_SECONDS = 1.0
STOP_EVENT = threading.Event()


def _request_stop(signum, frame):
    abr.log(f"[signal] Received signal {signum}; stopping demo...")
    STOP_EVENT.set()


def start_ffmpeg_writer(out_path: str, width: int, height: int, fps: int) -> subprocess.Popen:
    """
    Write a continuous demo MP4 from raw BGR frames via stdin.
    (Video-only; audio is reserved for real per-animal recordings.)
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
        # Helps keep file playable if you Ctrl+C
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def close_ffmpeg(ff: subprocess.Popen, timeout: float = 8.0) -> None:
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


def overlay_hud(frame_bgr, lines, is_recording: bool, rec_flash_on: bool):
    """
    Semi-transparent overlay panel; State line turns red when recording.
    No black “strip”.
    """
    h, w = frame_bgr.shape[:2]

    panel_h = 165
    x1, y1 = 10, 10
    x2, y2 = w - 10, min(h - 10, y1 + panel_h)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.35
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    y = y1 + 32
    for s in lines:
        if s.startswith("State:"):
            color = (0, 0, 255) if is_recording else (255, 255, 255)
            thickness = 3 if is_recording else 2
        else:
            color = (255, 255, 255)
            thickness = 2

        cv2.putText(frame_bgr, s, (x1 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
        y += 30

    if is_recording and rec_flash_on:
        cv2.putText(frame_bgr, "REC", (w - 160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.circle(frame_bgr, (w - 55, 45), 12, (0, 0, 255), -1)

    return frame_bgr


def build_overlay_lines(cfg: abr.Settings, state: abr.RuntimeState, animal, conf, hb: str) -> list[str]:
    return [
        f"Heartbeat: {hb}",
        f"Detect: {animal or 'none'}   conf={conf:.2f}",
        f"State: {'RECORDING' if state.is_recording else 'STANDBY'}",
        f"Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Start gate: {cfg.start_confirm_frames} @ conf>={cfg.start_min_conf:.2f} | "
        f"Stop gate: {cfg.stop_confirm_frames} misses | MinRec: {cfg.min_record_seconds:.1f}s",
    ]


def demo_loop(cfg: abr.Settings) -> None:
    # Demo file (continuous, overlayed)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    demo_path = os.path.join(cfg.capture_dir, f"demo-{stamp}.mp4")
    os.makedirs(cfg.capture_dir, exist_ok=True)

    abr.log("Starting demo overlay + real recorder...")
    abr.log(f"Demo overlay file (video-only): {demo_path}")
    abr.log("Real detections will be saved as normal clips WITH audio under captures/")

    # Reuse your existing building blocks
    cam = abr.CameraManager(cfg)
    detector = abr.AnimalDetector(cfg)
    recorder = abr.Recorder(cfg)          # <-- REAL recorder (video+audio mux per clip)
    watchdog = abr.ProgressWatchdog(cfg.progress_timeout_seconds)
    state = abr.RuntimeState()

    if cfg.enable_manual_focus:
        abr.set_manual_focus(cfg.winning_focus)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cam.start()
    watchdog.start()

    # Start demo writer
    out_w, out_h = DEMO_OUTPUT_RESOLUTION
    ff = start_ffmpeg_writer(demo_path, out_w, out_h, DEMO_FPS)
    if ff.stdin is None:
        raise RuntimeError("Failed to start ffmpeg (stdin is None).")

    # Overlay timers
    last_overlay_hb = time.time()
    overlay_hb = "Standby: ready"
    last_flash = time.time()
    rec_flash_on = True

    abr.log("[standby] Waiting for animals...")

    try:
        while not STOP_EVENT.is_set():
            loop_start = time.time()

            # Watchdog restart request?
            if watchdog.restart_flag.is_set():
                abr.log("[demo] Watchdog requested camera restart.")
                # If we were recording, stop cleanly first
                if state.is_recording:
                    try:
                        recorder.stop(cam)
                    except Exception:
                        pass
                state.is_recording = False
                state.misses_while_recording = 0
                state.consecutive_hits = 0
                state.pending_label = None
                cam.restart(reason="watchdog stall (demo)")
                watchdog.restart_flag.clear()

            watchdog.touch()
            state.frame_count += 1
            now_m = time.monotonic()
            now = time.time()

            abr.handle_heartbeat(cfg, state, now_m, now)

            # Capture + detect
            frame_lores_bgr = cam.capture_bgr_from_lores()
            watchdog.touch()
            animal, conf = detector.best_target(frame_lores_bgr)
            watchdog.touch()

            # Console debug
            if animal and abr.should_print_progress(cfg, state):
                if state.is_recording:
                    abr.log(f"[keep] {animal} conf={conf:.2f} (recording)")
                else:
                    gate = "strong" if conf >= cfg.start_min_conf else "weak"
                    abr.log(f"[detect-{gate}] {animal} conf={conf:.2f}")

            # Overlay HB
            if (time.time() - last_overlay_hb) >= OVERLAY_HEARTBEAT_SECONDS:
                overlay_hb = "Recording..." if state.is_recording else "Standby: ready"
                last_overlay_hb = time.time()

            # Flashing REC
            if (time.time() - last_flash) >= REC_FLASH_PERIOD:
                rec_flash_on = not rec_flash_on
                last_flash = time.time()

            # -------------------------
            # REAL state machine:
            # start/stop actual recordings using abr.Recorder
            # -------------------------
            if state.is_recording:
                if abr.stop_logic(cfg, recorder, state, now, animal):
                    abr.log(f"Target likely gone (misses={state.misses_while_recording}). Stopping recording.")
                    recorder.stop(cam)  # <-- creates final MP4 with audio
                    state.is_recording = False
                    state.misses_while_recording = 0
                    state.consecutive_hits = 0
                    state.pending_label = None
                    state.next_start_allowed_time = time.time() + cfg.post_record_cooldown_seconds
                    abr.log("[standby] Ready for next animal...")
                    cam.restart(reason="post-record restart (demo)")
            else:
                if abr.in_post_stop_cooldown(state, now):
                    state.consecutive_hits = 0
                    state.pending_label = None
                else:
                    if animal and abr.start_logic(cfg, state, animal, conf):
                        # Start a REAL clip (with audio)
                        final_path = recorder.make_filename(animal)
                        abr.log(f"DETECTION CONFIRMED: {animal.upper()} x{state.consecutive_hits} | Starting clip: {final_path}")
                        recorder.start(cam, final_path)
                        state.is_recording = True
                        state.misses_while_recording = 0
                        state.consecutive_hits = 0
                        state.pending_label = None
                    elif not animal:
                        state.consecutive_hits = 0
                        state.pending_label = None

            # -------------------------
            # Write overlay frame into demo mp4
            # -------------------------
            vis = cv2.resize(frame_lores_bgr, DEMO_OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            lines = build_overlay_lines(cfg, state, animal, conf, overlay_hb)
            vis = overlay_hud(vis, lines, state.is_recording, rec_flash_on)

            try:
                ff.stdin.write(vis.tobytes())
            except BrokenPipeError:
                abr.log("[demo] ffmpeg pipe broke (ffmpeg exited).")
                break

            # Throttle loop
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / DEMO_FPS) - elapsed))

    except KeyboardInterrupt:
        abr.log("\nStopping demo (Ctrl+C).")

    finally:
        # Stop any active real recording cleanly
        try:
            if state.is_recording:
                recorder.stop(cam)
        except Exception:
            pass

        close_ffmpeg(ff)

        try:
            cam.stop_close()
        except Exception:
            pass

        abr.log(f"[demo] Done. Overlay demo saved: {demo_path}")
        abr.log("[demo] Check captures/ for your normal bird clips (they should include audio).")


def main():
    demo_loop(abr.CFG)


if __name__ == "__main__":
    main()
