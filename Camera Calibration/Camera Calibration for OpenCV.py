# -----------------------------------------------------------------------------------------------------------------------
# -------------------------CORNER DETECTION-------------------------

#import numpy.core.multiarray
import numpy as np
import cv2
import glob

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining Object Points for Project
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Initializing Arrays for Object Points and Image Points
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Reading all jpg images in Pictures Directory
images = glob.glob('Camera Calibration\Images\*.jpg')

if images == False:                                 # If No Image is Detected
    print("No Image detected in this resolution")

else:
    resolution = (500, 500)
    for k in images:
        img = cv2.imread(k)

        # Resizing Pictures
        #img = cv2.resize(img, resolution, interpolation = None)

        # Converting all images to GrayScale for OpenCV Detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting the Chessboard's Corners
        flag, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)

        # Detection, Refinement and Addition of Corner Points
        if flag == True:
            objpoints.append(objp)

            # Refinement of Corner Points
            ref_corners = cv2.cornerSubPix(
                gray_img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(ref_corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), ref_corners, flag)
            cv2.imshow('Figure', img)
            cv2.waitKey(1)

cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------------------
# -------------------------CAMERA CALIBRATION-------------------------

flag_c, cam_matrix, dist_coeff, r_vec, t_vec = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
print(str(cam_matrix))
# ------------------------------------------------------------------------------------------------------------------------
# -------------------------UNDISTORTION ALGOS-------------------------

img = cv2.imread('Camera Calibration/Images/left02.jpg')
h,  w = img.shape[:2]
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(
    cam_matrix, dist_coeff, (w, h), 1, (w, h))
print(str(newcam_mtx))
"""
# undistort1
dst = cv2.undistort(img, cam_matrix, dist_coeff, None, newcam_mtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult1.png', dst)

# undistort2
mapx, mapy = cv2.initUndistortRectifyMap(
    cam_matrix, dist_coeff, None, newcam_mtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult2.png', dst)


# ------------------------------------------------------------------------------------------------------------------------
# -------------------------RE-PROJECTION ERROR-------------------------

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
