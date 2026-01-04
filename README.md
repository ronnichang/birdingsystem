# DIY Birding System empowered by Raspberry Pi 4

A Raspberry Pi 4 “birding / backyard wildlife” recorder automatically detects approaching birds
or animals of interest. Upon a bird being detected, the system starts recording till the bird 
leaves.

## Major technical steps:
- Runs **YOLO (Ultralytics YOLOv8n)** on a low-res camera stream to detect animals.
- Starts recording when a target animal is detected for **N consecutive frames**.
- Stops recording after **M consecutive misses** (hysteresis) and a minimum clip length.
- Saves clips to `captures/` using the filename format: `YYYYMMDD-HHMMSS-<animal>.mp4`.
- Includes a **headless demo recorder** that produces a single MP4 with status overlays
  (heartbeat text + flashing REC indicator) without needing VNC/monitor.
- Optional: exposes `captures/` via a simple password-protected **nginx** web directory listing.

## Hardware
- Raspberry Pi 4
- CSI camera (e.g., Arducam / Raspberry Pi Camera Module, OV5647-based)
- (Optional) Motorized focus support via I2C (configurable)

## Software
- Raspberry Pi OS (Debian-based)
- Python 3
- `picamera2` / `libcamera`
- `ultralytics` (YOLOv8)
- `opencv-python`
- `ffmpeg` (for demo MP4 writing, and/or video muxing)

---

## Repository Layout

- `auto_bird_recorder.py`  
  Main wildlife recorder: YOLO detection + start/stop recording + camera restart reliability.

- `demo_headless_overlay.py`  
  Headless demo: writes a **single continuous MP4** with overlays burned in (no GUI required).

- `captures/`  
  Output directory for recordings and demo videos.

---

## System-wide environment

```bash
sudo apt update
sudo apt install -y ffmpeg python3-picamera2 rpicam-apps python3-opencv python3-torch python3-torchvision
```

## Python environment (venv)

From the project root:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

## Demo Videos
- Full camera action from **power on → power off** (from `demo_headless_overlay.py`): [YouTube](https://youtu.be/E2w8IsiXREs?si=CW3ZZTqgefNI25TH). When the camera does not detect a bird, it is in the STANDBY mode. When a bird
is detected, it is in the RECORDING mode, which will trigger video writing.
- Auto-detected & recorded clip (from `auto_bird_recorder.py`) — example 1: [YouTube](https://youtu.be/dDopLY99IWo?si=3fvarHAWod5dv105)
- Auto-detected & recorded clip (from `auto_bird_recorder.py`) — example 2: [YouTube](https://youtu.be/q5BS_q3UHrA?si=_YpgvA6xsWA2XPwC)
