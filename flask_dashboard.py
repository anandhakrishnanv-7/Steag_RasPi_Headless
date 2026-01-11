#!/usr/bin/env python3
"""
Flask Web Dashboard Backend for Coal Field Monitoring
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time
import cv2
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'coalfield_monitoring_secret'
CORS(app)
# Use gevent for Python 3.13 compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Data directory paths
DATA_DIR = Path("/home/pi/coalfield_data")
THERMAL_DIR = DATA_DIR / "thermal_images"
ALERT_DIR = DATA_DIR / "alerts"
LOG_FILE = DATA_DIR / "sensor_log.csv"

# In-memory storage for recent data
recent_alerts = []
recent_gps_positions = []
current_sensor_data = {}
MAX_ALERTS = 100
MAX_GPS_POINTS = 500

# Video streaming
camera_frame = None
camera_lock = threading.Lock()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """Serve main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/alert', methods=['POST'])
def receive_alert():
    """Receive alert from monitoring system"""
    try:
        alert_data = request.json
        alert_data['id'] = len(recent_alerts) + 1
        alert_data['received_at'] = datetime.now().isoformat()
        
        recent_alerts.insert(0, alert_data)
        if len(recent_alerts) > MAX_ALERTS:
            recent_alerts.pop()
        
        # Broadcast to all connected clients
        socketio.emit('new_alert', alert_data, broadcast=True)
        
        return jsonify({'status': 'success', 'id': alert_data['id']}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/gps', methods=['POST'])
def receive_gps():
    """Receive GPS position update"""
    try:
        gps_data = request.json
        gps_data['timestamp'] = datetime.now().isoformat()
        
        recent_gps_positions.append(gps_data)
        if len(recent_gps_positions) > MAX_GPS_POINTS:
            recent_gps_positions.pop(0)
        
        # Broadcast to clients
        socketio.emit('gps_update', gps_data, broadcast=True)
        
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/alerts')
def get_alerts():
    """Get all recent alerts"""
    return jsonify(recent_alerts)

@app.route('/api/gps/track')
def get_gps_track():
    """Get GPS track data"""
    return jsonify(recent_gps_positions)

@app.route('/api/sensor/current')
def get_current_sensor():
    """Get current sensor readings"""
    return jsonify(current_sensor_data)

@app.route('/api/sensor/history')
def get_sensor_history():
    """Get sensor history from log file"""
    try:
        hours = int(request.args.get('hours', 1))
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        data = []
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        row_time = datetime.strptime(row['Timestamp'], 
                                                    "%Y-%m-%d %H:%M:%S.%f")
                        if row_time >= cutoff_time:
                            data.append(row)
                    except:
                        pass
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/images/thermal')
def get_thermal_images():
    """Get list of thermal images"""
    try:
        limit = int(request.args.get('limit', 50))
        images = []
        
        if THERMAL_DIR.exists():
            image_files = sorted(THERMAL_DIR.glob('*.jpg'), 
                               key=lambda x: x.stat().st_mtime, 
                               reverse=True)
            
            for img_path in image_files[:limit]:
                images.append({
                    'filename': img_path.name,
                    'path': f'/images/thermal/{img_path.name}',
                    'timestamp': datetime.fromtimestamp(
                        img_path.stat().st_mtime
                    ).isoformat()
                })
        
        return jsonify(images)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/images/alerts')
def get_alert_images():
    """Get list of alert images"""
    try:
        limit = int(request.args.get('limit', 50))
        images = []
        
        if ALERT_DIR.exists():
            image_files = sorted(ALERT_DIR.glob('*.jpg'),
                               key=lambda x: x.stat().st_mtime,
                               reverse=True)
            
            for img_path in image_files[:limit]:
                images.append({
                    'filename': img_path.name,
                    'path': f'/images/alerts/{img_path.name}',
                    'timestamp': datetime.fromtimestamp(
                        img_path.stat().st_mtime
                    ).isoformat()
                })
        
        return jsonify(images)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/images/thermal/<filename>')
def serve_thermal_image(filename):
    """Serve thermal image"""
    return send_from_directory(THERMAL_DIR, filename)

@app.route('/images/alerts/<filename>')
def serve_alert_image(filename):
    """Serve alert image"""
    return send_from_directory(ALERT_DIR, filename)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    stats = {
        'total_alerts': len(recent_alerts),
        'gps_points': len(recent_gps_positions),
        'system_uptime': get_uptime(),
        'last_alert': recent_alerts[0] if recent_alerts else None,
        'current_gps': recent_gps_positions[-1] if recent_gps_positions else None
    }
    return jsonify(stats)

# =============================================================================
# VIDEO STREAMING
# =============================================================================

def generate_video_frames():
    """Generator for video streaming"""
    global camera_frame
    
    while True:
        with camera_lock:
            if camera_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', camera_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video/feed')
def video_feed():
    """Video streaming route"""
    from flask import Response
    return Response(generate_video_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video/frame', methods=['POST'])
def receive_video_frame():
    """Receive video frame from camera system"""
    global camera_frame
    
    try:
        # Receive base64 encoded frame
        data = request.json
        img_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        with camera_lock:
            camera_frame = frame
        
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('request_data')
def handle_data_request(data):
    """Handle data request from client"""
    data_type = data.get('type')
    
    if data_type == 'alerts':
        emit('alerts_data', recent_alerts)
    elif data_type == 'gps':
        emit('gps_data', recent_gps_positions)
    elif data_type == 'sensor':
        emit('sensor_data', current_sensor_data)

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def sensor_log_monitor():
    """Monitor sensor log file and update current data"""
    global current_sensor_data
    
    while True:
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r') as f:
                    # Read last line
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1]
                        reader = csv.DictReader([lines[0], last_line])
                        current_sensor_data = next(reader)
                        
                        # Broadcast update
                        socketio.emit('sensor_update', current_sensor_data, 
                                    broadcast=True)
        except Exception as e:
            print(f"Error reading sensor log: {e}")
        
        time.sleep(1)

def get_uptime():
    """Get system uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            return str(timedelta(seconds=int(uptime_seconds)))
    except:
        return "Unknown"

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    
    # Create data directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    THERMAL_DIR.mkdir(exist_ok=True)
    ALERT_DIR.mkdir(exist_ok=True)
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=sensor_log_monitor, daemon=True)
    monitor_thread.start()
    
    print("\n" + "="*60)
    print("Coal Field Monitoring Dashboard Server")
    print("="*60)
    print(f"\nServer starting on http://0.0.0.0:5000")
    print(f"Data directory: {DATA_DIR}")
    print(f"Python version: {sys.version}")
    print("\nPress Ctrl+C to stop\n")
    
    # Run server with gevent (Python 3.13 compatible)
    # Use async_mode='gevent' for Python 3.13+
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, 
                               handler_class=WebSocketHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop()