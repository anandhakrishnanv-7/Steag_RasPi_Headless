#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
Modified with hotspot threshold detection and bounding box feature
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Key Bindings:')
print('')
print('a z: Increase/Decrease Blur')
print('s x: Floating High and Low Temp Label Threshold')
print('d c: Change Interpolated scale Note: This will not change the window size on the Pi')
print('f v: Contrast')
print('q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)')
print('r t: Record and Stop')
print('p : Snapshot')
print('m : Cycle through ColorMaps')
print('h : Toggle HUD')
print('n b: Increase/Decrease Hotspot Threshold Temperature')
print('g : Toggle Hotspot Detection')

import cv2
import numpy as np
import argparse
import time
import io

#We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
#https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
parser.add_argument("--hotspot-temp", type=float, default=30.0, help="Hotspot threshold temperature in Celsius")
args = parser.parse_args()
	
if args.device:
	dev = args.device
else:
	dev = 0
	
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
#cap = cv2.VideoCapture(0)
#pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
#https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"

# Hotspot detection settings
hotspot_threshold_temp = args.hotspot_temp
hotspot_detection_enabled = True

def rec():
	now = time.strftime("%Y%m%d--%H%M%S")
	#do NOT use mp4 here, it is flakey!
	videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
	return(videoOut)

def snapshot(heatmap):
	#I would put colons in here, but it Win throws a fit if you try and open them!
	now = time.strftime("%Y%m%d-%H%M%S") 
	snaptime = time.strftime("%H:%M:%S")
	cv2.imwrite("TC001"+now+".png", heatmap)
	return snaptime

def detect_hotspots(thdata, threshold_temp):
	"""
	Detect regions where temperature exceeds the threshold
	Returns a binary mask and list of contours
	"""
	# Create temperature map
	temp_map = np.zeros((height, width), dtype=np.float32)
	
	for row in range(height):
		for col in range(width):
			hi = thdata[row][col][0]
			lo = thdata[row][col][1]
			lo = lo * 256
			rawtemp = hi + lo
			temp = (rawtemp / 64) - 273.15
			temp_map[row][col] = temp
	
	# Create binary mask where temperature exceeds threshold
	mask = (temp_map > threshold_temp).astype(np.uint8) * 255
	
	# Apply morphological operations to clean up the mask
	kernel = np.ones((3,3), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	
	# Find contours
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	return mask, contours, temp_map
 

while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		imdata,thdata = np.array_split(frame, 2)
		#now parse the data from the bottom frame and convert to temp!
		#https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
		#Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
		#grab data from the center pixel...
		hi = thdata[96][128][0]
		lo = thdata[96][128][1]
		#print(hi,lo)
		lo = lo*256
		rawtemp = hi+lo
		#print(rawtemp)
		temp = (rawtemp/64)-273.15
		temp = round(temp,2)
		#print(temp)
		#break

		#find the max temperature in the frame
		lomax = thdata[...,1].max()
		posmax = thdata[...,1].argmax()
		#since argmax returns a linear index, convert back to row and col
		mcol,mrow = divmod(posmax,width)
		himax = thdata[mcol][mrow][0]
		lomax=lomax*256
		maxtemp = himax+lomax
		maxtemp = (maxtemp/64)-273.15
		maxtemp = round(maxtemp,2)

		
		#find the lowest temperature in the frame
		lomin = thdata[...,1].min()
		posmin = thdata[...,1].argmin()
		#since argmax returns a linear index, convert back to row and col
		lcol,lrow = divmod(posmin,width)
		himin = thdata[lcol][lrow][0]
		lomin=lomin*256
		mintemp = himin+lomin
		mintemp = (mintemp/64)-273.15
		mintemp = round(mintemp,2)

		#find the average temperature in the frame
		loavg = thdata[...,1].mean()
		hiavg = thdata[...,0].mean()
		loavg=loavg*256
		avgtemp = loavg+hiavg
		avgtemp = (avgtemp/64)-273.15
		avgtemp = round(avgtemp,2)

		

		# Convert the real image to RGB
		bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
		#Contrast
		bgr = cv2.convertScaleAbs(bgr, alpha=alpha)#Contrast
		#bicubic interpolate, upscale and blur
		bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
		if rad>0:
			bgr = cv2.blur(bgr,(rad,rad))

		#apply colormap
		if colormap == 0:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
			cmapText = 'Jet'
		if colormap == 1:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
			cmapText = 'Hot'
		if colormap == 2:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
			cmapText = 'Magma'
		if colormap == 3:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
			cmapText = 'Inferno'
		if colormap == 4:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
			cmapText = 'Plasma'
		if colormap == 5:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
			cmapText = 'Bone'
		if colormap == 6:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
			cmapText = 'Spring'
		if colormap == 7:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
			cmapText = 'Autumn'
		if colormap == 8:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
			cmapText = 'Viridis'
		if colormap == 9:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
			cmapText = 'Parula'
		if colormap == 10:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
			heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
			cmapText = 'Inv Rainbow'

		#print(heatmap.shape)

		# Hotspot detection
		if hotspot_detection_enabled:
			mask, contours, temp_map = detect_hotspots(thdata, hotspot_threshold_temp)
			
			# Draw bounding boxes around hotspots
			for contour in contours:
				# Filter out very small contours (noise)
				if cv2.contourArea(contour) > 5:  # minimum area threshold
					# Get bounding rectangle
					x, y, w, h = cv2.boundingRect(contour)
					
					# Scale coordinates to match the display size
					x_scaled = x * scale
					y_scaled = y * scale
					w_scaled = w * scale
					h_scaled = h * scale
					
					# Draw bounding box with thick red border
					cv2.rectangle(heatmap, (x_scaled, y_scaled), 
								(x_scaled + w_scaled, y_scaled + h_scaled), 
								(0, 0, 255), 3)
					
					# Add a label with "HOTSPOT"
					label_y = y_scaled - 5 if y_scaled > 20 else y_scaled + h_scaled + 15
					cv2.putText(heatmap, 'HOTSPOT', (x_scaled, label_y),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
					cv2.putText(heatmap, 'HOTSPOT', (x_scaled, label_y),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

		# draw crosshairs
		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
		#show temp
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		if hud==True:
			# display black box for our data
			cv2.rectangle(heatmap, (0, 0),(160, 148), (0,0,0), -1)
			# put text in the box
			cv2.putText(heatmap,'Avg Temp: '+str(avgtemp)+' C', (10, 14),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Label Threshold: '+str(threshold)+' C', (10, 28),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Colormap: '+cmapText, (10, 42),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Blur: '+str(rad)+' ', (10, 56),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Scaling: '+str(scale)+' ', (10, 70),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Contrast: '+str(alpha)+' ', (10, 84),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Snapshot: '+snaptime+' ', (10, 98),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			if recording == False:
				cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
				cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
			if recording == True:
				cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
				cv2.FONT_HERSHEY_SIMPLEX, 0.4,(40, 40, 255), 1, cv2.LINE_AA)
			
			# Display hotspot threshold info
			hotspot_status = 'ON' if hotspot_detection_enabled else 'OFF'
			hotspot_color = (0, 255, 0) if hotspot_detection_enabled else (128, 128, 128)
			cv2.putText(heatmap,'Hotspot: '+hotspot_status+' @ '+str(hotspot_threshold_temp)+' C', (10, 126),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, hotspot_color, 1, cv2.LINE_AA)
			
			cv2.putText(heatmap,'Max: '+str(maxtemp)+' C', (10, 140),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
		
		#Yeah, this looks like we can probably do this next bit more efficiently!
		#display floating max temp
		if maxtemp > avgtemp+threshold:
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		#display floating min temp
		if mintemp < avgtemp-threshold:
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		#display image
		cv2.imshow('Thermal',heatmap)

		if recording == True:
			elapsed = (time.time() - start)
			elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
			#print(elapsed)
			videoOut.write(heatmap)
		
		keyPress = cv2.waitKey(1)
		if keyPress == ord('a'): #Increase blur radius
			rad += 1
		if keyPress == ord('z'): #Decrease blur radius
			rad -= 1
			if rad <= 0:
				rad = 0

		if keyPress == ord('s'): #Increase threshold
			threshold += 1
		if keyPress == ord('x'): #Decrease threashold
			threshold -= 1
			if threshold <= 0:
				threshold = 0

		if keyPress == ord('d'): #Increase scale
			scale += 1
			if scale >=5:
				scale = 5
			newWidth = width*scale
			newHeight = height*scale
			if dispFullscreen == False and isPi == False:
				cv2.resizeWindow('Thermal', newWidth,newHeight)
		if keyPress == ord('c'): #Decrease scale
			scale -= 1
			if scale <= 1:
				scale = 1
			newWidth = width*scale
			newHeight = height*scale
			if dispFullscreen == False and isPi == False:
				cv2.resizeWindow('Thermal', newWidth,newHeight)

		if keyPress == ord('q'): #enable fullscreen
			dispFullscreen = True
			cv2.namedWindow('Thermal',cv2.WND_PROP_FULLSCREEN)
			cv2.setWindowProperty('Thermal',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		if keyPress == ord('w'): #disable fullscreen
			dispFullscreen = False
			cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
			cv2.setWindowProperty('Thermal',cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
			cv2.resizeWindow('Thermal', newWidth,newHeight)

		if keyPress == ord('f'): #contrast+
			alpha += 0.1
			alpha = round(alpha,1)#fix round error
			if alpha >= 3.0:
				alpha=3.0
		if keyPress == ord('v'): #contrast-
			alpha -= 0.1
			alpha = round(alpha,1)#fix round error
			if alpha<=0:
				alpha = 0.0

		if keyPress == ord('h'):
			if hud==True:
				hud=False
			elif hud==False:
				hud=True

		if keyPress == ord('m'): #m to cycle through color maps
			colormap += 1
			if colormap == 11:
				colormap = 0

		if keyPress == ord('n'): #Increase hotspot threshold temp
			hotspot_threshold_temp += 1.0
			hotspot_threshold_temp = round(hotspot_threshold_temp, 1)
		
		if keyPress == ord('b'): #Decrease hotspot threshold temp
			hotspot_threshold_temp -= 1.0
			hotspot_threshold_temp = round(hotspot_threshold_temp, 1)
		
		if keyPress == ord('g'): #Toggle hotspot detection
			hotspot_detection_enabled = not hotspot_detection_enabled

		if keyPress == ord('r') and recording == False: #r to start reording
			videoOut = rec()
			recording = True
			start = time.time()
		if keyPress == ord('t'): #f to finish reording
			recording = False
			elapsed = "00:00:00"

		if keyPress == ord('p'): #f to finish reording
			snaptime = snapshot(heatmap)

		if keyPress == ord('q'):
			break
			capture.release()
			cv2.destroyAllWindows()