# Coal Field Hotspot Monitoring System - Complete Setup Guide

## System Overview

This system integrates thermal imaging, GPS tracking, and IMU sensors to detect and monitor hotspots in coal fields with a real-time web dashboard.

### Components:
1. **Thermal Camera Module** - Topdon TC001 for hotspot detection
2. **Sensor Logging** - GPS + MPU9250 IMU for location and orientation
3. **Web Dashboard** - Real-time monitoring with OpenStreetMap integration
4. **Video Streaming** - Live 240p thermal feed

---

## Hardware Requirements

### Required Hardware:
- Raspberry Pi 4 (4GB+ recommended) with Raspberry Pi OS Lite
- Topdon TC001 Thermal Camera
- GPS Module (UART/Serial compatible)
- MPU9250 9-axis IMU
- 2x LEDs (Red for alerts, Green for status)
- 220Ω resistors for LEDs
- MicroSD Card (32GB+ recommended)
- Power supply (5V 3A)

### Optional:
- External WiFi dongle for better range
- Battery pack for mobile operation

---

## Software Installation

### 1. Prepare Raspberry Pi OS Lite

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-dev python3-opencv \
    libatlas-base-dev libjasper-dev libqtgui4 libqt4-test \
    libhdf5-dev libhdf5-serial-dev v4l-utils i2c-tools

# Enable camera, I2C, and serial interfaces
sudo raspi-config
# Navigate to: Interface Options
# Enable: Camera, I2C, Serial Port (disable console, enable hardware)

# Reboot
sudo reboot
```

### 2. Install Python Dependencies

Create a requirements.txt file:

```
# requirements.txt
opencv-python==4.8.1.78
numpy==1.24.3
pyserial==3.5
pynmea2==1.19.0
RPi.GPIO==0.7.1
mpu9250-jmdev==1.0.13
flask==3.0.0
flask-socketio==5.3.5
flask-cors==4.0.0
requests==2.31.0
python-socketio==5.10.0
eventlet==0.33.3
```

Install dependencies:

```bash
# Create virtual environment (recommended)
python3 -m venv ~/coalfield_venv
source ~/coalfield_venv/bin/activate

# Install packages
pip install -r requirements.txt

# For OpenCV with GUI support (if needed for debugging)
sudo apt install -y python3-opencv
```

### 3. Verify Hardware Connections

#### Thermal Camera:
```bash
# Check if camera is detected
v4l2-ctl --list-devices

# Test camera
v4l2-ctl -d /dev/video0 --list-formats-ext
```

#### GPS Module:
```bash
# Check serial port
ls -l /dev/serial*
ls -l /dev/ttyAMA* /dev/ttyS*

# Test GPS (should see NMEA sentences)
cat /dev/serial0
```

#### IMU (MPU9250):
```bash
# Check I2C devices
sudo i2cdetect -y 1

# Should show device at 0x68 (MPU) and 0x0C (magnetometer)
```

#### GPIO/LEDs:
```bash
# Test LED control
python3 << EOF
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # Alert LED
GPIO.setup(27, GPIO.OUT)  # Status LED

GPIO.output(17, GPIO.HIGH)
time.sleep(1)
GPIO.output(17, GPIO.LOW)

GPIO.output(27, GPIO.HIGH)
time.sleep(1)
GPIO.output(27, GPIO.LOW)

GPIO.cleanup()
EOF
```

---

## File Structure

Create the following directory structure:

```
/home/pi/coalfield_monitor/
├── thermal_sensor_integration.py  # Main monitoring system
├── flask_dashboard.py             # Web dashboard backend
├── video_streamer.py              # Video streaming module
├── requirements.txt
├── templates/
│   └── dashboard.html             # Web interface
├── static/                        # Static files (if any)
└── coalfield_data/                # Data directory (auto-created)
    ├── thermal_images/
    ├── alerts/
    └── sensor_log.csv
```

### Setup Commands:

```bash
# Create directories
mkdir -p ~/coalfield_monitor/templates
cd ~/coalfield_monitor

# Copy your Python files here
# - thermal_sensor_integration.py
# - flask_dashboard.py
# - video_streamer.py

# Copy HTML to templates
cp dashboard.html templates/

# Make data directory
mkdir -p /home/pi/coalfield_data/{thermal_images,alerts}
```

---

## Configuration

### 1. Update Configuration in thermal_sensor_integration.py

Edit the configuration section:

```python
# Adjust these based on your setup
THERMAL_DEVICE = 0  # Usually 0, check with v4l2-ctl
GPS_SERIAL_PORT = "/dev/serial0"  # Or /dev/ttyAMA0, /dev/ttyS0
HOTSPOT_THRESHOLD_TEMP = 45.0  # Adjust for your coal field

# LED pins (BCM numbering)
LED_ALERT_PIN = 17  # Physical pin 11
LED_STATUS_PIN = 27  # Physical pin 13

# Dashboard URL
DASHBOARD_URL = "http://localhost:5000/api"
```

### 2. Configure GPS Serial Port

Edit `/boot/config.txt`:

```bash
sudo nano /boot/config.txt

# Add these lines:
dtoverlay=pi3-disable-bt
enable_uart=1
```

Edit `/boot/cmdline.txt` - Remove console reference:

```bash
sudo nano /boot/cmdline.txt

# Remove: console=serial0,115200
# Keep everything else
```

---

## Running the System

### Method 1: Manual Start (for testing)

Open 2 terminal windows:

**Terminal 1 - Start Flask Dashboard:**
```bash
cd ~/coalfield_monitor
source ~/coalfield_venv/bin/activate
python3 flask_dashboard.py
```

**Terminal 2 - Start Monitoring System:**
```bash
cd ~/coalfield_monitor
source ~/coalfield_venv/bin/activate
python3 thermal_sensor_integration.py --threshold 45.0
```

### Method 2: Auto-start with systemd (recommended)

Create systemd service files:

**Dashboard Service:**
```bash
sudo nano /etc/systemd/system/coalfield-dashboard.service
```

```ini
[Unit]
Description=Coal Field Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/coalfield_monitor
Environment="PATH=/home/pi/coalfield_venv/bin"
ExecStart=/home/pi/coalfield_venv/bin/python3 flask_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Monitoring Service:**
```bash
sudo nano /etc/systemd/system/coalfield-monitor.service
```

```ini
[Unit]
Description=Coal Field Thermal Monitoring
After=network.target coalfield-dashboard.service
Requires=coalfield-dashboard.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/coalfield_monitor
Environment="PATH=/home/pi/coalfield_venv/bin"
ExecStart=/home/pi/coalfield_venv/bin/python3 thermal_sensor_integration.py --threshold 45.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start services:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable coalfield-dashboard
sudo systemctl enable coalfield-monitor

# Start services now
sudo systemctl start coalfield-dashboard
sudo systemctl start coalfield-monitor

# Check status
sudo systemctl status coalfield-dashboard
sudo systemctl status coalfield-monitor

# View logs
sudo journalctl -u coalfield-dashboard -f
sudo journalctl -u coalfield-monitor -f
```

---

## Accessing the Dashboard

### On the Raspberry Pi itself:
```
http://localhost:5000
```

### From another device on the same network:
```
http://[RASPBERRY_PI_IP]:5000
```

Find Raspberry Pi IP address:
```bash
hostname -I
```

### Port Forwarding (for remote access):

Configure your router to forward port 5000 to your Raspberry Pi's local IP.

**Security Note:** For production use, add authentication and use HTTPS.

---

## Usage Guide

### Dashboard Tabs:

1. **Overview** - Statistics and recent alerts
2. **Live Map** - OpenStreetMap with GPS track and hotspot markers
3. **Live Stream** - Real-time thermal video feed (240p)
4. **Thermal Images** - Gallery of captured thermal images
5. **Alerts** - Complete list of all hotspot alerts
6. **Sensor Data** - Current and historical sensor readings

### LED Indicators:

- **Green LED (GPIO 27)** - GPS Fix Status
  - ON: GPS has valid fix
  - OFF: GPS acquiring satellites

- **Red LED (GPIO 17)** - Hotspot Alert
  - FLASH: Hotspot detected above threshold
  - OFF: Normal operation

### Key Features:

- **Automatic Hotspot Detection** - Triggers when temperature exceeds threshold
- **GPS-Tagged Alerts** - Each alert includes precise GPS coordinates
- **Real-time Map Updates** - Track displayed on OpenStreetMap
- **1-Second Intervals** - Thermal capture and sensor logging every second
- **Alert Images** - Automatically saved with bounding boxes
- **Live Video Stream** - 240p feed at ~10 FPS

---

## Troubleshooting

### Thermal Camera Not Detected:
```bash
# Check USB connection
lsusb

# Check video devices
ls -l /dev/video*
v4l2-ctl --list-devices

# Try different device number
python3 thermal_sensor_integration.py --device 1
```

### GPS Not Working:
```bash
# Test serial port
cat /dev/serial0
# Should see NMEA sentences like $GNRMC...

# Check permissions
sudo usermod -a -G dialout pi
# Log out and back in

# Verify UART enabled
ls -l /dev/serial0
```

### I2C/IMU Issues:
```bash
# Check I2C
sudo i2cdetect -y 1

# Enable I2C if not enabled
sudo raspi-config
# Interface Options > I2C > Enable

# Check permissions
sudo usermod -a -G i2c pi
```

### Dashboard Not Accessible:
```bash
# Check if Flask is running
sudo netstat -tlnp | grep 5000

# Check firewall (if enabled)
sudo ufw allow 5000

# Test locally first
curl http://localhost:5000
```

### High CPU Usage:
```bash
# Reduce thermal capture rate
# Edit thermal_sensor_integration.py:
THERMAL_CAPTURE_INTERVAL = 2.0  # Capture every 2 seconds

# Reduce video streaming FPS
# Edit video_streamer.py:
target_fps = 5  # Lower FPS
```

---

## Performance Optimization

### For Raspberry Pi OS Lite:

1. **Disable unnecessary services:**
```bash
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
sudo systemctl disable triggerhappy
```

2. **Increase GPU memory (for camera):**
```bash
sudo nano /boot/config.txt
# Add: gpu_mem=128
```

3. **Overclock (optional, at your own risk):**
```bash
sudo nano /boot/config.txt
# Add:
# over_voltage=2
# arm_freq=1750
```

4. **Use ramfs for temporary images:**
```python
# In thermal_sensor_integration.py, change:
THERMAL_DIR = Path("/tmp/thermal_images")  # RAM disk
```

---

## Data Management

### Automatic Cleanup Script:

Create a cleanup script to manage disk space:

```bash
nano ~/cleanup_old_data.sh
```

```bash
#!/bin/bash
# Delete thermal images older than 7 days
find /home/pi/coalfield_data/thermal_images -name "*.jpg" -mtime +7 -delete

# Keep alert images for 30 days
find /home/pi/coalfield_data/alerts -name "*.jpg" -mtime +30 -delete

# Compress old log files
find /home/pi/coalfield_data -name "sensor_log_*.csv" -mtime +7 -exec gzip {} \;
```

```bash
chmod +x ~/cleanup_old_data.sh

# Add to crontab (run daily at 2 AM)
crontab -e
# Add: 0 2 * * * /home/pi/cleanup_old_data.sh
```

---

## Safety and Maintenance

### Regular Maintenance:

1. Check disk space weekly: `df -h`
2. Monitor system temperature: `vcgencmd measure_temp`
3. Review logs for errors: `sudo journalctl -p err`
4. Backup data regularly
5. Clean thermal camera lens monthly

### Backup Script:

```bash
#!/bin/bash
BACKUP_DIR="/media/usb_backup/coalfield_$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
rsync -av /home/pi/coalfield_data/ $BACKUP_DIR/
echo "Backup complete: $BACKUP_DIR"
```

---

## Advanced Features

### Email Alerts (Optional):

Install and configure email notifications:

```bash
pip install smtp
```

Add to thermal_sensor_integration.py:

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(temp, gps):
    msg = MIMEText(f"Hotspot: {temp}°C at {gps['lat']}, {gps['lon']}")
    msg['Subject'] = 'Coal Field Hotspot Alert'
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = 'alert-receiver@example.com'
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('your-email@gmail.com', 'app-password')
        smtp.send_message(msg)
```

---

## Support and Further Development

### Logging Verbosity:

Increase logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Custom Threshold per Zone:

Implement zone-based thresholds using GPS coordinates.

### Machine Learning Integration:

Train models to predict hotspot development patterns.

---

## Summary

You now have a complete coal field monitoring system with:

✅ Real-time thermal imaging with hotspot detection  
✅ GPS-tagged location tracking  
✅ IMU sensor integration  
✅ Web dashboard with live map  
✅ 240p video streaming  
✅ Automatic alerts with image capture  
✅ Data logging and archival

The system runs automatically on boot and provides comprehensive monitoring capabilities for coal field safety management.