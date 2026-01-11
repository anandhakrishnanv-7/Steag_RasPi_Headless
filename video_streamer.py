#!/usr/bin/env python3
"""
Video Streaming Module for Thermal Camera
Streams thermal video feed to web dashboard at 240p resolution
"""

import cv2
import requests
import base64
import time
import threading
from pathlib import Path

class ThermalVideoStreamer:
    def __init__(self, thermal_camera, dashboard_url, target_fps=10):
        """
        Initialize video streamer
        
        Args:
            thermal_camera: ThermalCamera instance
            dashboard_url: URL of the Flask dashboard
            target_fps: Target frames per second for streaming
        """
        self.thermal_camera = thermal_camera
        self.dashboard_url = dashboard_url
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # 240p resolution (426x240)
        self.stream_width = 426
        self.stream_height = 240
        
        self.running = False
        self.stream_thread = None
        
    def start(self):
        """Start video streaming in a separate thread"""
        if self.running:
            print("Video streamer already running")
            return
        
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print(f"✅ Video streaming started at {self.target_fps} FPS (240p)")
    
    def stop(self):
        """Stop video streaming"""
        self.running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        print("Video streaming stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        last_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.01)
                continue
            
            last_frame_time = current_time
            
            try:
                # Get thermal frame
                heatmap, thdata, temps = self.thermal_camera.read_frame()
                
                if heatmap is not None:
                    # Resize to 240p
                    frame_240p = cv2.resize(heatmap, 
                                           (self.stream_width, self.stream_height),
                                           interpolation=cv2.INTER_AREA)
                    
                    # Add overlay information
                    frame_240p = self._add_overlay(frame_240p, temps)
                    
                    # Send to dashboard
                    self._send_frame(frame_240p)
                    
            except Exception as e:
                print(f"Streaming error: {e}")
                time.sleep(0.5)
    
    def _add_overlay(self, frame, temps):
        """Add temperature overlay to frame"""
        # Add semi-transparent black bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.stream_width, 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Add temperature text
        text = f"Max: {temps.get('max', 0)}°C  Avg: {temps.get('avg', 0)}°C"
        cv2.putText(frame, text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def _send_frame(self, frame):
        """Send frame to dashboard via HTTP POST"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, 
                                    [cv2.IMWRITE_JPEG_QUALITY, 60])
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to dashboard
            response = requests.post(
                f"{self.dashboard_url}/api/video/frame",
                json={'frame': frame_base64},
                timeout=1
            )
            
            if response.status_code != 200:
                print(f"Frame send failed: {response.status_code}")
                
        except requests.exceptions.RequestException:
            # Silently fail if dashboard is not reachable
            pass
        except Exception as e:
            print(f"Frame encoding error: {e}")


# Alternative: MJPEG Streaming Server
class MJPEGStreamServer:
    """
    Alternative streaming method using MJPEG protocol
    This can be used if the POST method causes performance issues
    """
    
    def __init__(self, thermal_camera, port=8081, target_fps=10):
        self.thermal_camera = thermal_camera
        self.port = port
        self.target_fps = target_fps
        self.running = False
        
        # 240p resolution
        self.stream_width = 426
        self.stream_height = 240
        
        # Frame buffer
        self.current_frame = None
        self.frame_lock = threading.Lock()
    
    def start(self):
        """Start MJPEG server"""
        from flask import Flask, Response
        
        app = Flask(__name__)
        
        @app.route('/stream')
        def stream():
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Start frame capture thread
        self.running = True
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        print(f"✅ MJPEG stream server started on port {self.port}")
        print(f"   Stream URL: http://localhost:{self.port}/stream")
        
        # Run Flask server
        app.run(host='0.0.0.0', port=self.port, threaded=True)
    
    def _capture_loop(self):
        """Continuously capture frames"""
        frame_interval = 1.0 / self.target_fps
        last_capture = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_capture >= frame_interval:
                try:
                    heatmap, thdata, temps = self.thermal_camera.read_frame()
                    
                    if heatmap is not None:
                        # Resize to 240p
                        frame_240p = cv2.resize(heatmap,
                                               (self.stream_width, self.stream_height),
                                               interpolation=cv2.INTER_AREA)
                        
                        # Add overlay
                        frame_240p = self._add_overlay(frame_240p, temps)
                        
                        # Update current frame
                        with self.frame_lock:
                            self.current_frame = frame_240p
                    
                    last_capture = current_time
                    
                except Exception as e:
                    print(f"Capture error: {e}")
            
            time.sleep(0.01)
    
    def _generate_frames(self):
        """Generator for MJPEG stream"""
        while True:
            with self.frame_lock:
                if self.current_frame is not None:
                    # Encode as JPEG
                    ret, buffer = cv2.imencode('.jpg', self.current_frame,
                                              [cv2.IMWRITE_JPEG_QUALITY, 60])
                    
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_bytes + b'\r\n')
            
            time.sleep(1.0 / self.target_fps)
    
    def _add_overlay(self, frame, temps):
        """Add temperature overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.stream_width, 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        text = f"Max: {temps.get('max', 0)}°C  Avg: {temps.get('avg', 0)}°C"
        cv2.putText(frame, text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame


# Usage example for integration
if __name__ == "__main__":
    from thermal_sensor_integration import ThermalCamera
    
    print("Video Streamer Test")
    print("This module should be imported and used with the main monitoring system")
    print("\nUsage:")
    print("  from video_streamer import ThermalVideoStreamer")
    print("  streamer = ThermalVideoStreamer(thermal_camera, 'http://localhost:5000')")
    print("  streamer.start()")