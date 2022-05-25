#-----------------------------------------------------------------------------------------------------------------------
#-------------------------CORNER DETECTION-------------------------

import numpy as np
import cv2
import glob

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining Object Points for Project
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Initializing Arrays for Object Points and Image Points
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Reading all jpg images in Pictures Directory
images = glob.glob('/media/*.jpg')

if images == False:                                 # If No Image is Detected
    print("No Image detected in this resolution")

else:
    #resolution = (500, 500)
    for k in images:
        img = cv2.imread(k)

        # Resizing Pictures
        #img = cv2.resize(img, resolution, interpolation=None)

        # Converting all images to GrayScale for OpenCV Detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting the Chessboard's Corners
        flag, corners = cv2.findChessboardCorners(gray_img, (7, 7), None)

        # Detection, Refinement and Addition of Corner Points
        if flag == True:
            objpoints.append(objp)

            # Refinement of Corner Points
            ref_corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(ref_corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 7), ref_corners, flag)
            print("Corners Detected")
            cv2.imshow('Figure', img)
            cv2.waitKey(0)

cv2.destroyAllWindows()

"""
# Performing the calibration
calibDone, cam_mtx, coeffDist, rVect, tVect = cv2.calibrateCamera(
    objpoints, imgpoints, gray_img.shape[::-1], None, None)

# Writing the results to a .txt file
L = ["Camera Matrix:\n \n" + str(cam_mtx) + "\n \n", "Distortion Coefficients:\n \n" + str(coeffDist) + "\n \n",
     "Rotation Matrix:\n \n" + str(rVect) + "\n \n", "Translation Matrix:\n \n" + str(tVect)]

calibResults = open("calibResults.txt", "w")
calibResults.write("The camera calibration results are as follows: \n \n")
calibResults.writelines(L)
print("The calibration results can be viewed in calibResults.txt in the project directory.")
"""
#------------------------------------------------------------------------------------------------------------------------
#-------------------------CAMERA CALIBRATION-------------------------

cal_flag, cam_matrix, dist_coeff, r_vec, t_vec = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

#------------------------------------------------------------------------------------------------------------------------
#-------------------------UNDISTORTION ALGOS-------------------------

img = cv2.imread('/media/specific_image.jpg')
h,  w = img.shape[:2]
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (w, h), 1, (w, h))

# undistort1
dst = cv2.undistort(img, cam_matrix, dist_coeff, None, newcam_mtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/media/calibresult1.png', dst)

# undistort2
mapx, mapy = cv2.initUndistortRectifyMap(cam_matrix, dist_coeff, None, newcam_mtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/media/calibresult2.png', dst)

#------------------------------------------------------------------------------------------------------------------------
#-------------------------RE-PROJECTION ERROR-------------------------

mean_error = 0
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], r_vec[i], t_vec[i], cam_matrix, dist_coeff)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error
mean_error = tot_error / len(objpoints)
print("mean error: ", mean_error)
print("total error: ", tot_error)

"""
# Writing the results to a .txt file
L = ["Camera Matrix:\n \n", str(cam_matrix), "\n \n", "Distortion Coefficients:\n \n", str(dist_coeff), "\n \n",
     "Rotation Matrix:\n \n", str(r_vec), "\n \n", "Translation Matrix:\n \n", str(t_vec)]

calibResults = open("calibResults.txt", "w")
calibResults.write("The camera calibration results are as follows: \n \n")
calibResults.writelines(L)
print("The calibration results can be viewed in calibResults.txt in the project directory.")
"""
