#!/usr/bin/env python3
"""
STEAG COAL FIELD MONITOR (RADIOMETRIC PRODUCTION)
- Hardware: Raspberry Pi 5
- Sensors: Topdon TC001, GPS (UART), IMU (I2C)
- Features: Exact Temp Readings, Map Tracking, Raw Data Logging
"""

import csv
import cv2
import numpy as np
import threading
import time
import subprocess
import atexit
import serial
import pynmea2
import smbus2
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from collections import deque

# ================= CONFIGURATION =================
# Hardware Paths
GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600
IMU_ADDR = 0x68
THERMAL_DEVICE_ID = 0  # Run 'v4l2-ctl --list-devices' to check (0 or 1)

# Alert Logic
ALERT_THRESHOLD = 40.0 # °C

# File Storage
DATA_DIR = Path.home() / "Steag_Data_Production"
LOG_FILE = DATA_DIR / "production_log.csv"

# ================= FILTERS =================
class MovingAverage:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
    def update(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)

# ================= FLASK SETUP =================
app = Flask(__name__)
class SharedState:
    def __init__(self):
        self.frame = None
        self.data = {
            "gps": {"lat": 0.0, "lon": 0.0, "status": "No Fix"},
            "imu": {"accel_z": 0.0},
            "thermal": {"max": 0.0, "avg": 0.0, "center": 0.0},
            "alert": False
        }
        self.lock = threading.Lock()
state = SharedState()

# =============================================================================
# 1. REAL GPS DRIVER
# =============================================================================
class RealGPS:
    def __init__(self):
        self.lat_filter = MovingAverage(window_size=5)
        self.lon_filter = MovingAverage(window_size=5)
        self.raw_lat = 0.0
        self.raw_lon = 0.0
        self.status = "Searching..."
        self.running = True
        try:
            self.ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
            print(f"GPS: Connected to {GPS_PORT}")
            threading.Thread(target=self._update_loop, daemon=True).start()
        except Exception as e:
            print(f"GPS ERROR: {e}")
            self.status = "Error"

    def _update_loop(self):
        while self.running:
            try:
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('$GNRMC') or line.startswith('$GPRMC'):
                        msg = pynmea2.parse(line)
                        if msg.latitude != 0.0:
                            self.raw_lat = msg.latitude
                            self.raw_lon = msg.longitude
                            self.status = "Active"
                        else:
                            self.status = "No Lock"
            except Exception as e:
                print(f"IMU read error: {e}")

            time.sleep(0.1)

    def get_latest(self):
        return self.lat_filter.update(self.raw_lat), self.lon_filter.update(self.raw_lon), self.status

# =============================================================================
# 2. REAL IMU DRIVER
# =============================================================================
class RealIMU:
    def __init__(self):
        self.z_filter = MovingAverage(window_size=3)
        self.raw_z = 0.0
        self.running = True
        try:
            self.bus = smbus2.SMBus(1)
            self.bus.write_byte_data(IMU_ADDR, 0x6B, 0)
            print(f"IMU: Connected at 0x{IMU_ADDR:02X}")
            threading.Thread(target=self._update_loop, daemon=True).start()
        except Exception as e:
            print(f"IMU ERROR: {e}")

    def _update_loop(self):
        while self.running:
            try:
                high = self.bus.read_byte_data(IMU_ADDR, 0x3F)
                low = self.bus.read_byte_data(IMU_ADDR, 0x40)
                val = (high << 8) | low
                if val > 32768: val -= 65536
                self.raw_z = val / 16384.0 
            except: pass
            time.sleep(0.01) # Pi 5 can handle fast updates

    def get_latest(self):
        return self.z_filter.update(self.raw_z)

# =============================================================================
# 3. RADIOMETRIC THERMAL CAMERA (Topdon TC001 Pi 5 Fix)
# =============================================================================
class RadiometricThermalCamera:
    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.latest_temps = {'max': 0.0, 'avg': 0.0, 'center': 0.0}
        self.latest_grid = None
        self.running = True
        self.width = 256
        self.height = 192
        self.scale = 2 
        self.mode = "SIM"
        
        # PI 5 FIX: Explicitly test both video0 and video1 as strings
        for dev_path in ['/dev/video0', '/dev/video1']:
            try:
                print(f"Probing {dev_path} for Topdon TC001...")
                cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L)
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0) # Request raw YUV data
                
                # Test if we actually get a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.mode = "REAL"
                    print(f"RADIOMETRIC CAM: Locked on {dev_path}")
                    break # Stop hunting, we found it!
                else:
                    cap.release()
            except Exception as e:
                pass

        if self.mode == "SIM":
            print("⚠️ CAMERA ERROR: Topdon not responding. Using Simulation Fallback.")

        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while self.running:
            if self.mode == "REAL" and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret: 
                    self._process_frame(frame)
            else:
                self._simulate_frame()
            time.sleep(0.01)

    def _process_frame(self, frame):
        try:
            # Split Topdon Double-Height Frame
            imdata, thdata = np.array_split(frame, 2)
            
            # Radiometric Math
            lo_all = thdata[...,1].astype(np.uint16) * 256
            hi_all = thdata[...,0].astype(np.uint16)
            raw_all = lo_all + hi_all
            temp_grid = (raw_all / 64.0) - 273.15
            self.latest_grid = temp_grid
            
            t_max = np.max(temp_grid)
            t_avg = np.mean(temp_grid)
            t_center = temp_grid[96, 128]
            
            self.latest_temps = {
                'max': round(float(t_max), 2),
                'avg': round(float(t_avg), 2),
                'center': round(float(t_center), 2)
            }
            
            # Visuals
            bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
            new_w, new_h = self.width * self.scale, self.height * self.scale
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
            
            # Crosshair
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(temp_grid)
            screen_max_x = int(maxLoc[0] * self.scale)
            screen_max_y = int(maxLoc[1] * self.scale)
            
            cv2.circle(heatmap, (screen_max_x, screen_max_y), 10, (255, 255, 255), 2)
            cv2.line(heatmap, (screen_max_x-10, screen_max_y), (screen_max_x+10, screen_max_y), (255,255,255), 2)
            
            label = f"MAX: {t_max:.1f} C"
            cv2.putText(heatmap, label, (screen_max_x+15, screen_max_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.latest_frame = heatmap
            
        except Exception as e:
            print(f"Frame Process Error: {e}")

    def _simulate_frame(self):
        """The missing method! Generates a fake heatmap if hardware fails."""
        new_w, new_h = self.width * self.scale, self.height * self.scale
        frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        frame[:] = (40, 20, 0) 
        
        # Bouncing red circle
        t = time.time()
        x = int((np.sin(t) + 1) / 2 * (new_w - 50)) + 25
        y = int(new_h / 2)
        cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
        cv2.putText(frame, "SIMULATION MODE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.latest_frame = frame
        self.latest_temps = {'max': 65.0, 'avg': 35.0, 'center': 40.0}

    def get_data(self):
        if self.latest_frame is None:
             return np.zeros((384, 512, 3), dtype=np.uint8), {'max': 0, 'avg': 0}
        return self.latest_frame, self.latest_temps
# =============================================================================
# MAIN LOOP
# =============================================================================
# =============================================================================
# MAIN LOOP – PRODUCTION READY
# =============================================================================
def sensor_loop():
    gps = RealGPS()
    imu = RealIMU()
    cam = RadiometricThermalCamera()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Timestamp",
                "Latitude",
                "Longitude",
                "GPS_Status",
                "Accel_Z",
                "Temp_Max",
                "Temp_Avg",
                "Hot_Ratio",
                "Severity",
                "Alert",
                "CPU_Temp"
            ])

    print(f"🚀 SYSTEM ACTIVE: Logging to {LOG_FILE}")

    alert_start_time = None
    ALERT_DELAY = 0.1  # seconds persistence required

    last_log_time = 0

    while True:

        lat, lon, status = gps.get_latest()
        accel_z = imu.get_latest()
        frame, temps = cam.get_data()

        # Ensure camera has valid data
        if frame is None or temps is None:
            time.sleep(0.05)
            continue

        # --------------------------------------------------
        # HOT AREA RATIO CALCULATION
        # --------------------------------------------------
        if hasattr(cam, "latest_grid") and cam.latest_grid is not None:
            temp_grid = cam.latest_grid
            hot_mask = temp_grid > ALERT_THRESHOLD
            hot_ratio = float(np.sum(hot_mask) / hot_mask.size)
        else:
            hot_ratio = 0.0

        # --------------------------------------------------
        # SEVERITY CLASSIFICATION
        # --------------------------------------------------
        max_temp = temps.get('max', 0)

        if max_temp > 80:
            severity = "CRITICAL"
        elif max_temp > 65:
            severity = "HIGH"
        elif max_temp > 55:
            severity = "ELEVATED"
        else:
            severity = "NORMAL"

        # --------------------------------------------------
        # PERSISTENT ALERT LOGIC
        # --------------------------------------------------
        is_alert = False

        # --------------------------------------------------
        # PERSISTENT ALERT LOGIC (Time-Based Only)
        # --------------------------------------------------
        if max_temp >= ALERT_THRESHOLD:
            if alert_start_time is None:
                # 1. Start the clock the exact moment it crosses the threshold
                alert_start_time = time.time()
                is_alert = False # Not an alert YET
                print(f"Heat detected ({max_temp:.1f}C). Starting {ALERT_DELAY}s timer...")
                
            elif time.time() - alert_start_time >= ALERT_DELAY:
                # 2. It has stayed hot for the required duration! Trigger it.
                is_alert = True
                print("ALERT TRIGGERED!")
            else:
                # 3. Still waiting for the delay to finish counting up
                is_alert = False 
        else:
            # 4. It cooled down below the threshold. Reset the timer.
            if alert_start_time is not None:
                print("Cooled down, resetting timer.")
            alert_start_time = None
            is_alert = False

        # --------------------------------------------------
        # SYSTEM TELEMETRY
        # --------------------------------------------------
        try:
            cpu_temp = float(
                subprocess.check_output(["vcgencmd", "measure_temp"])
                .decode()
                .split("=")[1]
                .split("'")[0]
            )
        except:
            cpu_temp = 0.0

        telemetry_packet = {
            "timestamp": datetime.now().isoformat(),
            "gps": {
                "lat": lat,
                "lon": lon,
                "status": status
            },
            "thermal": {
                "max": max_temp,
                "avg": temps.get("avg", 0),
                "hot_ratio": hot_ratio,
                "severity": severity
            },
            "imu": {
                "accel_z": accel_z
            },
            "system": {
                "cpu_temp": cpu_temp,
                "fps": getattr(cam, "fps", 0)
            }
        }

        # --------------------------------------------------
        # UPDATE SHARED STATE
        # --------------------------------------------------
        with state.lock:
            state.data['gps'] = telemetry_packet["gps"]
            state.data['imu'] = telemetry_packet["imu"]
            state.data['thermal'] = telemetry_packet["thermal"]
            state.data['alert'] = is_alert
            state.data['system'] = telemetry_packet["system"]
            state.frame = frame

        # --------------------------------------------------
        # LOGGING (1Hz)
        # --------------------------------------------------
        if time.time() - last_log_time >= 1.0:
            try:
                with open(LOG_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        lat,
                        lon,
                        status,
                        accel_z,
                        max_temp,
                        temps.get("avg", 0),
                        hot_ratio,
                        severity,
                        is_alert,
                        cpu_temp
                    ])
                last_log_time = time.time()
            except:
                pass

        time.sleep(0.05)

# =============================================================================
# WEB SERVER
# =============================================================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/data')
def get_data_json():
    with state.lock: return jsonify(state.data)

def generate_mjpeg():
    while True:
        with state.lock:
            if state.frame is None: continue
            ret, buffer = cv2.imencode('.jpg', state.frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.02) 

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=sensor_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)