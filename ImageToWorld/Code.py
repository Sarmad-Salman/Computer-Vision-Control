#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sat Oct 24 11:39:16 2020

@author: omen
"""
import numpy as np
import cv2
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('/mnt/hgfs/Thesis Project/Camera Calibration (Task 1)/Images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
rotation_mat = np.zeros(shape=(3, 3))
for rvec in range (12) :
    R = cv2.Rodrigues(rvecs[rvec], rotation_mat)[0]
    Trans_mat = np.column_stack((np.matmul(mtx,R), tvecs[rvec]))
print("Transformation Matrix = \n", Trans_mat)
img = cv2.imread('/mnt/hgfs/Thesis Project/Camera Calibration (Task 1)/Images/left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort1
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/mnt/hgfs/Thesis Project/Camera Calibration (Task 1)/Images/calibresult1.png',dst)
# undistort2
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/mnt/hgfs/Thesis Project/Camera Calibration (Task 1)/Images/calibresult2.png',dst)
mean_error = 0
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, G = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error
    mean_error = tot_error / len(objpoints)
print ("\nmean error: ", mean_error)
print ("total error: ", tot_error)
new_img=cv2.imread('/home/omen/Pictures/Webcam/2020-10-28-045607.jpg')
# u=int(mtx.item(2)) 
# v=int(mtx.item(5))
u=300
v=470
print ("u = ", u, "v = ", v)
text = "x: " + str(u) + ", y: " + str(v)
cv2.circle(new_img,(u,v),3,(255, 200, 50),-1)
cv2.rectangle(new_img,(u-45,v-45),(u+45,v+45),(0,255,0),3,2)
cv2.putText(new_img,text,(u-35,v-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,50),1)
cv2.imshow('new_img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Final_product= mtx.dot(Trans_mat)
print('\nFinal_product = \n',Final_product)
Inverse_mat=np.linalg.pinv(Final_product,rcond=1e-15)
print('\nInverse_Matrix = \n', Inverse_mat)
s=10
world_coordinates=s*(Inverse_mat.dot(np.array([u, v, 1])))
print('\nWorld Coordinates = ', world_coordinates)
pixel_coordinates=(Final_product.dot(world_coordinates))/s
print('\nImage Coordinates = ', pixel_coordinates)