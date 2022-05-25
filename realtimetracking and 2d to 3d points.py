import numpy as np
import cv2 as cv2
import glob
import serial
# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
# ser.flush()
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 2.2
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
#Importing Images
images = glob.glob('/home/pi/Desktop/Work/Work/New folder/Final Images/*.jpg')                                         
if images == False:                                 # If No Image is Detected
    print("No Image detected in this resolution")
else:                                               #If Images are Detected
    for fname in images:
        img = cv2.imread(fname)
        # Converting all images to GrayScale for OpenCV Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detecting the Chessboard's Corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        # Detection, Refinement and Addition of Corner Points
        if ret == True:
            objpoints.append(objp)
            # Refinement of Corner Points
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            cv2.imshow('Figure', img)
            cv2.waitKey(1)
cv2.destroyAllWindows()
# Finding Instrinsic and Extrinsic Parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx =", mtx)
#mtx = [[886.6791737, 0.0, 545.46004313], [0.0, 922.16715059 ,489.67985791], [0.0, 0.0, 1.0]]
#print(mtx)
fx = mtx[0][0]
cx = mtx[0][2]
fy = mtx[1][1]
cy = mtx[1][2]
arduinodist = "Empty"
z_world = int(input("Enter Height to Maintain: "))
print(z_world)
# ser.write(str(z_world).encode('utf-8'))
# 
# while (arduinodist != "Reached"):
#     if ser.in_waiting > 0:
#             arduinodist = ser.readline().decode('utf-8').rstrip()
#             print(arduinodist)
# Forcefully bringing all parameters to Decimal
np.set_printoptions(suppress=True)
objpoints1=[]
imgpoints1=[]
gst_str = ('v4l2src device=/dev/video0 ! videorate ! image/jpeg, width=1280, height=480, framerate=30/2  ! jpegdec ! videocrop left=640 ! videoconvert ! appsink' )
cap = cv2.VideoCapture (gst_str, cv2.CAP_GSTREAMER)
framerate = 30
#host = input("Enter IP of Host")
out = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=I420, width=640, height=480, framerate=30/1 !'
                     'jpegenc ! jpegparse ! rtpjpegpay !'
                     'udpsink host=192.168.137.181 port=5000', cv2.CAP_GSTREAMER, 0, framerate, (640, 480))
#cap.set(cv2.CAP_PROP_FPS, 30.0)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# take first frame of the video
ret,frame = cap.read()
#frame=cv2.flip(frame,1)
rotation_matrix=np.eye(3)

roi_extract = cv2.selectROI("Select ROI", frame)
cv2.destroyWindow('Select ROI')

x, y, w, h = (roi_extract[0], roi_extract[1], roi_extract[2], roi_extract[3])
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
while(1):
#     if ser.in_waiting > 0:
#             line = ser.readline().decode('utf-8').rstrip()
#             print(line)
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if ret == True:
        ret, frame1 = cv2.threshold(frame, 180, 155, cv2.THRESH_TOZERO_INV)
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, criteria)
        x, y, w, h = track_window
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        #print (pts)
        s=1
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        u = x + int(w/2)
        v = y + int(h/2)
        cv2.circle(frame,(u,v),3,(255, 200, 50),-1)
        text = "x: " + str(u) + ", y: " + str(v)
        cv2.putText(frame, text, (u - 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
        cv2.imshow('Tracking Video',img2)
        out.write(frame)
        x_world = round((u - cx) * z_world / fx,2)
        y_world = round((v - cy) * z_world / fy,2)
        world_coordinates = [x_world, y_world, z_world]
#         if u >320 & v>240:
#             ser.write(b"Left and Back\n")
#             line = ser.readline().decode('utf-8').rstrip()
#         elif u>320 & v<240:
#             ser.write(b"Left and Forw\n")
#             line = ser.readline().decode('utf-8').rstrip()
#         elif u<320 & v>240:
#             ser.write(b"Right and Back\n")
#             line = ser.readline().decode('utf-8').rstrip()
#         elif u<320 & v<240:
#             ser.write(b"Right and Forw\n")
#             line = ser.readline().decode('utf-8').rstrip()
#         else:
#             ser.write(b"Stay\n")
#             line = ser.readline().decode('utf-8').rstrip()
        line = "Hello"
        print(line + '\tWorld Coordinates = ', world_coordinates)
        k = cv2.waitKey(1)
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows() 