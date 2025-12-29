#!/usr/bin/env python3
"""
demo_headless_overlay.py

Headless demo recorder that:
- imports and reuses most of auto_bird_recorder.py
- runs the same detection/recording state machine
- produces ONE continuous demo MP4 with overlays burned in
- records USB mic audio and muxes into the final MP4

Output files:
- demo-YYYYMMDD-HHMMSS.video.mp4   (temp)
- demo-YYYYMMDD-HHMMSS.audio.wav   (temp)
- demo-YYYYMMDD-HHMMSS.mp4         (final, video+audio)
"""

import os
import time
import datetime
import subprocess

import cv2

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

# Heartbeat overlay update rate (seconds)
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

        # More resilient if you Ctrl+C; keeps files playable
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",

        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def close_ffmpeg(ff: subprocess.Popen, timeout: float = 8.0) -> None:
    """Close ffmpeg cleanly so MP4 is finalized."""
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


def overlay_hud(frame_bgr, lines, is_recording: bool, rec_flash_on: bool):
    """
    Draw HUD overlay onto a BGR image.
    Semi-transparent panel; State line is red when recording.
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


def build_overlay_lines(cfg: abr.Settings, state: abr.RuntimeState, animal, conf, overlay_heartbeat: str) -> list[str]:
    return [
        f"Heartbeat: {overlay_heartbeat}",
        f"Detect: {animal or 'none'}   conf={conf:.2f}",
        f"State: {'RECORDING' if state.is_recording else 'STANDBY'}",
        f"Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Start gate: {cfg.start_confirm_frames} @ conf>={cfg.start_min_conf:.2f} | "
        f"Stop gate: {cfg.stop_confirm_frames} misses | MinRec: {cfg.min_record_seconds:.1f}s",
    ]


def mux_video_audio(cfg: abr.Settings, video_path: str, audio_path: str, out_path: str) -> bool:
    """
    Mux demo video+audio into a final MP4.
    Uses explicit -map so audio is not dropped.
    Applies cfg.audio_gain if != 1.0.
    """
    af = f"volume={cfg.audio_gain}" if cfg.audio_gain and cfg.audio_gain != 1.0 else "anull"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-af", af,
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mux_log = os.path.join(cfg.av_log_dir, f"demo-ffmpeg-mux-{ts}.log")

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        try:
            with open(mux_log, "w", encoding="utf-8") as f:
                f.write(result.stderr or "")
        except Exception:
            pass
        abr.log(f"[demo-audio] ffmpeg mux FAILED. See log: {mux_log}")
        return False

    if cfg.debug_av:
        try:
            with open(mux_log, "w", encoding="utf-8") as f:
                f.write(result.stderr or "")
        except Exception:
            pass

    return True


def demo_loop(cfg: abr.Settings) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(cfg.capture_dir, f"demo-{stamp}")
    demo_final = base + ".mp4"
    demo_video = base + ".video.mp4"
    demo_audio = base + ".audio.wav"

    os.makedirs(cfg.capture_dir, exist_ok=True)
    os.makedirs(cfg.av_log_dir, exist_ok=True)

    abr.log("Starting headless overlay demo (video+audio)...")
    abr.log(f"Demo final: {demo_final}")
    abr.log(f"Temp video:  {demo_video}")
    abr.log(f"Temp audio:  {demo_audio}")
    abr.log("(One continuous MP4 with HUD overlays + USB mic audio.)")

    # Reuse your components
    cam = abr.CameraManager(cfg)
    detector = abr.AnimalDetector(cfg)
    recorder_for_audio = abr.Recorder(cfg)  # we will reuse its audio start/stop logic
    watchdog = abr.ProgressWatchdog(cfg.progress_timeout_seconds)
    state = abr.RuntimeState()

    if cfg.enable_manual_focus:
        abr.set_manual_focus(cfg.winning_focus)

    cam.start()
    watchdog.start()

    # Start audio using the same SIGINT-safe approach as auto_bird_recorder
    recorder_for_audio._start_audio(demo_audio)

    # Start demo video writer (HUD-burned)
    out_w, out_h = DEMO_OUTPUT_RESOLUTION
    ff = start_ffmpeg_writer(demo_video, out_w, out_h, DEMO_FPS)
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
                abr.log("[demo] Watchdog requested camera restart.")
                # Ensure not recording (in state machine terms)
                state.is_recording = False
                state.misses_while_recording = 0
                state.consecutive_hits = 0
                state.pending_label = None
                cam.restart(reason="watchdog stall")
                watchdog.restart_flag.clear()

            watchdog.touch()
            state.frame_count += 1
            now_m = time.monotonic()
            now = time.time()

            # Terminal heartbeat (optional)
            abr.handle_heartbeat(cfg, state, now_m, now)

            # Capture + detect
            frame_lores_bgr = cam.capture_bgr_from_lores()
            watchdog.touch()
            animal, conf = detector.best_target(frame_lores_bgr)
            watchdog.touch()

            # Terminal debug prints (optional)
            if animal and abr.should_print_progress(cfg, state):
                if state.is_recording:
                    abr.log(f"[keep] {animal} conf={conf:.2f} (recording)")
                else:
                    gate = "strong" if conf >= cfg.start_min_conf else "weak"
                    abr.log(f"[detect-{gate}] {animal} conf={conf:.2f}")

            # Overlay heartbeat once/sec
            if (time.time() - last_overlay_heartbeat) >= OVERLAY_HEARTBEAT_SECONDS:
                overlay_heartbeat = "Recording..." if state.is_recording else "Standby: ready"
                last_overlay_heartbeat = time.time()

            # REC flash toggle
            if (time.time() - last_flash) >= REC_FLASH_PERIOD:
                rec_flash_on = not rec_flash_on
                last_flash = time.time()

            # ---- Recording state machine (for demo overlays only) ----
            if state.is_recording:
                if abr.stop_logic(cfg, recorder_for_audio, state, now, animal):
                    abr.log(f"Target likely gone (misses={state.misses_while_recording}). (demo state -> STANDBY)")
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
                        abr.log(f"DETECTION CONFIRMED: {animal.upper()} x{state.consecutive_hits} (demo state -> RECORDING)")
                        state.is_recording = True
                        state.misses_while_recording = 0
                        state.consecutive_hits = 0
                        state.pending_label = None
                    elif not animal:
                        state.consecutive_hits = 0
                        state.pending_label = None

            # ---- Build overlay frame and write to demo video ----
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
        # Stop video writer first (finalize mp4)
        close_ffmpeg(ff)

        # Stop audio cleanly (finalize wav header)
        recorder_for_audio._stop_audio()

        # Mux into final mp4
        abr.log("[demo] Muxing demo video+audio...")
        ok = mux_video_audio(cfg, demo_video, demo_audio, demo_final)

        if ok:
            abr.log(f"[demo] OK: {demo_final}")
            # Cleanup temp files
            for p in (demo_video, demo_audio):
                try:
                    os.remove(p)
                except Exception:
                    pass
        else:
            abr.log("[demo] Mux failed; keeping temp files for debugging:")
            abr.log(f"  {demo_video}")
            abr.log(f"  {demo_audio}")

        # Best-effort camera close
        try:
            cam.stop_close()
        except Exception:
            pass


def main():
    demo_loop(abr.CFG)


if __name__ == "__main__":
    main()
