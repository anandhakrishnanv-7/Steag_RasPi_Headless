#!/usr/bin/env python3
"""
Integrated Thermal Camera and Sensor Logging System
For Coal Field Hotspot Detection
"""

import cv2
import numpy as np
import serial
import pynmea2
import time
import sys
import csv
import json
import threading
from datetime import datetime
import math
from pathlib import Path
import requests
from queue import Queue

# Try importing RPi.GPIO and MPU, but allow running without hardware for testing
try:
    import RPi.GPIO as GPIO
    from mpu9250_jmdev.registers import *
    from mpu9250_jmdev.mpu_9250 import MPU9250
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Warning: Running without hardware support (GPIO/MPU)")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Thermal Camera Settings
THERMAL_DEVICE = 0
THERMAL_WIDTH = 256
THERMAL_HEIGHT = 192
THERMAL_SCALE = 3
HOTSPOT_THRESHOLD_TEMP = 45.0  # Default hotspot threshold in Celsius

# GPS/Sensor Settings
GPS_SERIAL_PORT = "/dev/serial0"
GPS_BAUDRATE = 9600
GPS_TIMEOUT = 1.0

# Timing Settings
SENSOR_LOG_INTERVAL = 1.0  # Log sensors every 1 second
THERMAL_CAPTURE_INTERVAL = 1.0  # Capture thermal image every 1 second

# File Paths
DATA_DIR = Path("/home/pi/coalfield_data")
THERMAL_DIR = DATA_DIR / "thermal_images"
ALERT_DIR = DATA_DIR / "alerts"
LOG_FILE = DATA_DIR / "sensor_log.csv"

# Web Dashboard API Endpoint
DASHBOARD_URL = "http://localhost:5000/api"  # Will be your Flask server

# LED Pins (if using hardware)
LED_ALERT_PIN = 17  # Red LED for alerts
LED_STATUS_PIN = 27  # Green LED for normal operation

# =============================================================================
# HELPER CLASSES
# =============================================================================

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.pitch = 0.0
        self.roll = 0.0
    
    def calculate_angles(self, accel_data, gyro_data, dt):
        accel_roll = math.degrees(math.atan2(accel_data[1], accel_data[2]))
        accel_pitch = math.degrees(math.atan2(-accel_data[0], 
                                    math.sqrt(accel_data[1]**2 + accel_data[2]**2)))
        gyro_roll_rate = gyro_data[0]
        gyro_pitch_rate = gyro_data[1]
        
        self.roll = self.alpha * (self.roll + gyro_roll_rate * dt) + \
                    (1.0 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gyro_pitch_rate * dt) + \
                     (1.0 - self.alpha) * accel_pitch
        
        return self.pitch, self.roll

class GPSReader:
    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.current_data = {
            'lat': 0.0,
            'lon': 0.0,
            'status': 'V',
            'speed': 0.0,
            'timestamp': None
        }
        self.lock = threading.Lock()
    
    def connect(self):
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, 
                                    timeout=self.timeout)
            print(f"✅ GPS connected on {self.port}")
            return True
        except Exception as e:
            print(f"❌ GPS connection failed: {e}")
            return False
    
    def read(self):
        if not self.ser or not self.ser.is_open:
            return None
        
        try:
            raw_data = self.ser.readline()
            if raw_data:
                data = raw_data.decode('utf-8', errors='ignore').strip()
                if data.startswith('$GNRMC') or data.startswith('$GPRMC'):
                    msg = pynmea2.parse(data)
                    with self.lock:
                        if msg.status == 'A':
                            self.current_data['lat'] = msg.latitude
                            self.current_data['lon'] = msg.longitude
                            self.current_data['status'] = 'A'
                            self.current_data['speed'] = msg.spd_over_grnd
                            self.current_data['timestamp'] = datetime.now()
                        else:
                            self.current_data['status'] = 'V'
                    return self.current_data
        except Exception as e:
            pass
        
        return None
    
    def get_current_position(self):
        with self.lock:
            return self.current_data.copy()
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

class ThermalCamera:
    def __init__(self, device=0, width=256, height=192, scale=3):
        self.device = device
        self.width = width
        self.height = height
        self.scale = scale
        self.newWidth = width * scale
        self.newHeight = height * scale
        self.cap = None
        self.colormap = cv2.COLORMAP_JET
        self.alpha = 1.0
    
    def connect(self):
        try:
            self.cap = cv2.VideoCapture(f'/dev/video{self.device}', cv2.CAP_V4L)
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
            print(f"✅ Thermal camera connected on /dev/video{self.device}")
            return True
        except Exception as e:
            print(f"❌ Thermal camera connection failed: {e}")
            return False
    
    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None, None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        
        imdata, thdata = np.array_split(frame, 2)
        
        # Calculate temperatures
        temps = self._calculate_temperatures(thdata)
        
        # Create heatmap
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
        bgr = cv2.resize(bgr, (self.newWidth, self.newHeight), 
                        interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(bgr, self.colormap)
        
        return heatmap, thdata, temps
    
    def _calculate_temperatures(self, thdata):
        temps = {}
        
        # Center temperature
        hi = thdata[96][128][0]
        lo = thdata[96][128][1] * 256
        temps['center'] = round((hi + lo) / 64 - 273.15, 2)
        
        # Max temperature
        lomax = thdata[..., 1].max()
        posmax = thdata[..., 1].argmax()
        mcol, mrow = divmod(posmax, self.width)
        himax = thdata[mcol][mrow][0]
        temps['max'] = round((himax + lomax * 256) / 64 - 273.15, 2)
        temps['max_pos'] = (mrow, mcol)
        
        # Min temperature
        lomin = thdata[..., 1].min()
        posmin = thdata[..., 1].argmin()
        lcol, lrow = divmod(posmin, self.width)
        himin = thdata[lcol][lrow][0]
        temps['min'] = round((himin + lomin * 256) / 64 - 273.15, 2)
        temps['min_pos'] = (lrow, lcol)
        
        # Average temperature
        loavg = thdata[..., 1].mean()
        hiavg = thdata[..., 0].mean()
        temps['avg'] = round((loavg * 256 + hiavg) / 64 - 273.15, 2)
        
        return temps
    
    def detect_hotspots(self, thdata, threshold_temp):
        """Detect regions exceeding threshold temperature"""
        temp_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                hi = thdata[row][col][0]
                lo = thdata[row][col][1] * 256
                temp = (hi + lo) / 64 - 273.15
                temp_map[row][col] = temp
        
        # Create binary mask
        mask = (temp_map > threshold_temp).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        hotspots = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                x, y, w, h = cv2.boundingRect(contour)
                hotspot_temps = temp_map[y:y+h, x:x+w]
                max_temp = np.max(hotspot_temps)
                avg_temp = np.mean(hotspot_temps)
                
                hotspots.append({
                    'bbox': (x, y, w, h),
                    'max_temp': round(max_temp, 2),
                    'avg_temp': round(avg_temp, 2),
                    'area': int(cv2.contourArea(contour))
                })
        
        return hotspots, mask
    
    def draw_hotspots(self, heatmap, hotspots):
        """Draw bounding boxes and labels for hotspots"""
        for hotspot in hotspots:
            x, y, w, h = hotspot['bbox']
            x_scaled = x * self.scale
            y_scaled = y * self.scale
            w_scaled = w * self.scale
            h_scaled = h * self.scale
            
            # Draw red bounding box
            cv2.rectangle(heatmap, (x_scaled, y_scaled),
                         (x_scaled + w_scaled, y_scaled + h_scaled),
                         (0, 0, 255), 3)
            
            # Add temperature label
            label = f"ALERT: {hotspot['max_temp']}C"
            label_y = y_scaled - 5 if y_scaled > 20 else y_scaled + h_scaled + 15
            cv2.putText(heatmap, label, (x_scaled, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, label, (x_scaled, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        return heatmap
    
    def close(self):
        if self.cap:
            self.cap.release()

# =============================================================================
# MAIN SYSTEM CLASS
# =============================================================================

class CoalFieldMonitor:
    def __init__(self):
        self.gps = GPSReader(GPS_SERIAL_PORT, GPS_BAUDRATE, GPS_TIMEOUT)
        self.thermal = ThermalCamera(THERMAL_DEVICE, THERMAL_WIDTH, 
                                     THERMAL_HEIGHT, THERMAL_SCALE)
        self.imu_filter = ComplementaryFilter(alpha=0.98)
        
        self.hotspot_threshold = HOTSPOT_THRESHOLD_TEMP
        self.running = False
        self.alert_queue = Queue()
        
        # MPU setup (if hardware available)
        self.mpu = None
        if HARDWARE_AVAILABLE:
            try:
                self.mpu = MPU9250(address_ak=0x0C, 
                                  address_mpu_master=MPU9050_ADDRESS_68,
                                  address_mpu_slave=None, bus=1, 
                                  gfs=GFS_1000, afs=AFS_8G,
                                  mfs=AK8963_BIT_16, 
                                  mode=AK8963_MODE_C100HZ)
                self.mpu.configure()
                print("✅ MPU9250 configured")
            except Exception as e:
                print(f"⚠️ MPU9250 initialization failed: {e}")
        
        # Create directories
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        THERMAL_DIR.mkdir(exist_ok=True)
        ALERT_DIR.mkdir(exist_ok=True)
        
        # Initialize log file
        self._initialize_log()
        
        # Setup GPIO if available
        if HARDWARE_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(LED_ALERT_PIN, GPIO.OUT)
                GPIO.setup(LED_STATUS_PIN, GPIO.OUT)
                GPIO.output(LED_ALERT_PIN, GPIO.LOW)
                GPIO.output(LED_STATUS_PIN, GPIO.LOW)
                print("✅ GPIO configured")
            except Exception as e:
                print(f"⚠️ GPIO setup failed: {e}")
    
    def _initialize_log(self):
        """Initialize CSV log file with headers"""
        if not LOG_FILE.exists():
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'GPS_Status', 'Latitude', 'Longitude', 
                    'Speed(kn)', 'Accel_X', 'Accel_Y', 'Accel_Z',
                    'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Pitch', 'Roll',
                    'Thermal_Max', 'Thermal_Avg', 'Alert'
                ])
            print(f"✅ Log file initialized: {LOG_FILE}")
    
    def _read_imu(self):
        """Read IMU data"""
        if self.mpu:
            try:
                accel = self.mpu.readAccelerometerMaster()
                gyro = self.mpu.readGyroscopeMaster()
                return accel, gyro
            except Exception:
                pass
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    def _log_sensor_data(self, gps_data, accel, gyro, pitch, roll, 
                        thermal_temps, alert):
        """Log all sensor data to CSV"""
        try:
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                row = [
                    timestamp, gps_data['status'], gps_data['lat'], 
                    gps_data['lon'], gps_data['speed'],
                    *accel, *gyro, pitch, roll,
                    thermal_temps.get('max', 0), thermal_temps.get('avg', 0),
                    alert
                ]
                writer.writerow(row)
        except Exception as e:
            print(f"Error logging data: {e}")
    
    def _send_alert_to_dashboard(self, gps_data, thermal_image_path, 
                                 hotspots, thermal_temps):
        """Send alert data to web dashboard"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'gps': {
                'lat': gps_data['lat'],
                'lon': gps_data['lon'],
                'status': gps_data['status']
            },
            'thermal': {
                'max_temp': thermal_temps['max'],
                'avg_temp': thermal_temps['avg'],
                'hotspots': hotspots
            },
            'image_path': str(thermal_image_path)
        }
        
        try:
            response = requests.post(
                f"{DASHBOARD_URL}/alert",
                json=alert_data,
                timeout=2
            )
            if response.status_code == 200:
                print(f"✅ Alert sent to dashboard")
            else:
                print(f"⚠️ Dashboard response: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to send alert: {e}")
            # Queue for retry
            self.alert_queue.put(alert_data)
    
    def _save_thermal_image(self, heatmap, is_alert=False):
        """Save thermal image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if is_alert:
            filepath = ALERT_DIR / f"alert_{timestamp}.jpg"
        else:
            filepath = THERMAL_DIR / f"thermal_{timestamp}.jpg"
        
        cv2.imwrite(str(filepath), heatmap)
        return filepath
    
    def start(self):
        """Start the monitoring system"""
        print("\n" + "="*60)
        print("Coal Field Hotspot Monitoring System")
        print("="*60 + "\n")
        
        # Connect hardware
        if not self.gps.connect():
            print("⚠️ Continuing without GPS")
        
        if not self.thermal.connect():
            print("❌ Thermal camera required. Exiting.")
            return
        
        self.running = True
        
        # Start threads
        sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        thermal_thread = threading.Thread(target=self._thermal_loop, daemon=True)
        gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
        
        sensor_thread.start()
        thermal_thread.start()
        gps_thread.start()
        
        print("\n✅ System started. Press Ctrl+C to stop.\n")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping system...")
            self.running = False
        
        # Wait for threads to finish
        sensor_thread.join(timeout=2)
        thermal_thread.join(timeout=2)
        gps_thread.join(timeout=2)
        
        self.cleanup()
    
    def _gps_loop(self):
        """Continuous GPS reading thread"""
        while self.running:
            self.gps.read()
            time.sleep(0.1)
    
    def _sensor_loop(self):
        """Sensor logging loop - runs every 1 second"""
        last_log_time = time.time()
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_log_time >= SENSOR_LOG_INTERVAL:
                dt = current_time - last_time
                last_time = current_time
                
                # Read sensors
                accel, gyro = self._read_imu()
                gps_data = self.gps.get_current_position()
                
                # Calculate angles
                if accel[2] != 0.0:
                    pitch, roll = self.imu_filter.calculate_angles(accel, gyro, dt)
                else:
                    pitch, roll = self.imu_filter.pitch, self.imu_filter.roll
                
                # This will be updated by thermal loop
                thermal_temps = {'max': 0, 'avg': 0}
                
                # Log data
                self._log_sensor_data(gps_data, accel, gyro, pitch, roll,
                                     thermal_temps, False)
                
                # Update status LED
                if HARDWARE_AVAILABLE and gps_data['status'] == 'A':
                    GPIO.output(LED_STATUS_PIN, GPIO.HIGH)
                else:
                    GPIO.output(LED_STATUS_PIN, GPIO.LOW)
                
                last_log_time = current_time
            
            time.sleep(0.1)
    
    def _thermal_loop(self):
        """Thermal camera loop - captures and analyzes every 1 second"""
        last_capture_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_capture_time >= THERMAL_CAPTURE_INTERVAL:
                # Read thermal frame
                heatmap, thdata, temps = self.thermal.read_frame()
                
                if heatmap is not None:
                    # Detect hotspots
                    hotspots, mask = self.thermal.detect_hotspots(
                        thdata, self.hotspot_threshold
                    )
                    
                    # Draw hotspots on heatmap
                    if hotspots:
                        heatmap = self.thermal.draw_hotspots(heatmap, hotspots)
                    
                    # Check for alerts
                    is_alert = temps['max'] > self.hotspot_threshold
                    
                    # Save image
                    image_path = self._save_thermal_image(heatmap, is_alert)
                    
                    # If alert, send to dashboard
                    if is_alert:
                        print(f"\n🔥 ALERT! Hotspot detected: {temps['max']}°C")
                        
                        # Flash alert LED
                        if HARDWARE_AVAILABLE:
                            GPIO.output(LED_ALERT_PIN, GPIO.HIGH)
                        
                        # Get current GPS position
                        gps_data = self.gps.get_current_position()
                        
                        # Send to dashboard
                        self._send_alert_to_dashboard(
                            gps_data, image_path, hotspots, temps
                        )
                        
                        # Log alert
                        accel, gyro = self._read_imu()
                        self._log_sensor_data(
                            gps_data, accel, gyro,
                            self.imu_filter.pitch, self.imu_filter.roll,
                            temps, True
                        )
                        
                        time.sleep(0.5)
                        if HARDWARE_AVAILABLE:
                            GPIO.output(LED_ALERT_PIN, GPIO.LOW)
                    
                    # Print status
                    status = f"[Thermal] Max: {temps['max']}°C | " \
                            f"Avg: {temps['avg']}°C | " \
                            f"Hotspots: {len(hotspots)}"
                    print(status)
                
                last_capture_time = current_time
            
            time.sleep(0.1)
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        self.thermal.close()
        self.gps.close()
        
        if HARDWARE_AVAILABLE:
            GPIO.output(LED_ALERT_PIN, GPIO.LOW)
            GPIO.output(LED_STATUS_PIN, GPIO.LOW)
            GPIO.cleanup()
        
        print("✅ Cleanup complete")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Coal Field Hotspot Monitoring System"
    )
    parser.add_argument("--threshold", type=float, default=45.0,
                       help="Hotspot temperature threshold in Celsius")
    parser.add_argument("--device", type=int, default=0,
                       help="Thermal camera device number")
    
    args = parser.parse_args()
    
    monitor = CoalFieldMonitor()
    monitor.hotspot_threshold = args.threshold
    monitor.start()