import os
import time
from picamera2 import Picamera2

# 1. Initialize camera for viewing ONLY
picam2 = Picamera2()
picam2.start()

def set_arducam_focus(val):
    # This manually formats the command for the Arducam motor (0-1023)
    # It uses 'i2cset' which is a system tool to talk to the chip at 0x0c
    value = (val << 4) & 0x3ff0
    dat1 = (value >> 8) & 0x3f
    dat2 = value & 0xf0
    os.system(f"i2cset -y 10 0x0c {dat1} {dat2}")

# 2. Sweep values: 0 (Far/Infinity) to 1000 (Very Close)
#test_points = [0, 250, 500, 750, 1000]
# Focus on the 300-600 range with smaller steps
test_points = [300, 350, 400, 450, 500, 550, 600]

print("Starting manual I2C focus sweep...")
for val in test_points:
    print(f"Moving lens to: {val}")
    set_arducam_focus(val)
    time.sleep(1) # Let the motor move
    
    filename = f"focus_{val}.jpg"
    picam2.capture_file(filename)
    print(f"Captured {filename}")

picam2.stop()
print("Sweep complete! Check your folder for the images.")