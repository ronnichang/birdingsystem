import time
from picamera2 import Picamera2

# 1. Initialize the camera
picam2 = Picamera2()

# 2. Configure for 1080p video (without the 'codec' argument here)
config = picam2.create_video_configuration(main={'size': (1920, 1080)})
picam2.configure(config)

# 3. Start the camera
picam2.start()

print("Recording 10 seconds of birding footage...")

# 4. Record using the high-level method which handles MP4 muxing natively
# This will save as a standard, playable MP4 file.
picam2.start_and_record_video("bird_capture.mp4", duration=10)

# 5. Cleanup
picam2.stop()
print("Capture complete! Download 'bird_capture.mp4' to your laptop and it will play.")