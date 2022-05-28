import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 2.2

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

#Importing Images
images = glob.glob('ImageToWorld/Images/*.jpg')                                         

if images == False:                                 # If No Image is Detected
    print("No Image detected in this resolution")

else:                                               #If Images are Detected
    resolution = (500, 666)
    for fname in images:
        img = cv2.imread(fname)

        # Resizing Pictures
        img = cv2.resize(img, resolution, interpolation = None)

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

# Forcefully bringing all parameters to Decimal
np.set_printoptions(suppress=True)

#print(objpoints[0])
#print(imgpoints[0])


# Undistortion
img = cv2.imread('ImageToWorld/Images/IMG20201028222432.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistort1
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('ImageToWorld/Images/calibresult1.png', dst)


# Obtaining Transformation Matrix
rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
R[0:3,2] = 0
Trans_mat = np.column_stack((R, tvecs[0]))
print("Transformation Matrix = \n", Trans_mat)

# Product of Camera Matrix and Transformation Matrix
Final_product = mtx.dot(Trans_mat)
print('\nFinal_product = \n', Final_product)

# Inverse Matrix for inverse computations
Inverse_mat = np.linalg.pinv(Final_product, rcond=1e-15)
print('\nInverse_Matrix = \n', Inverse_mat)

# Opening image for display of Image Coordinates
new_img = cv2.imread('ImageToWorld/Images/IMG20201028222432.jpg')
new_img = cv2.resize(new_img, resolution, interpolation=None)

#World Coordinates to Image Coordinates
x = 13.2
y = 13.2
z = 0
s = 1
world_coordinates=[x, y, z, 1]
pixel_coordinates = (Final_product.dot(world_coordinates))/s
s = pixel_coordinates[2]
pixel_coordinates = (Final_product.dot(world_coordinates))/s
print('\nImage Coordinates = ', pixel_coordinates)
u = int(pixel_coordinates[0])
v = int(pixel_coordinates[1])

# Image Coordinates to World Coordinates
#u = 334
#v = 488
s = 1
world_coordinates = s*(Inverse_mat.dot(np.array([u, v, 1])))
s = 1 / world_coordinates[3]
world_coordinates = s*(Inverse_mat.dot(np.array([u, v, 1])))
print('\nWorld Coordinates = ', world_coordinates)

# Displaying Coordinates on Image
print("u = ", u, "v = ", v)
text = "x: " + str(u) + ", y: " + str(v)
cv2.circle(new_img, (u, v), 3, (255, 200, 50), -1)
cv2.rectangle(new_img, (u-45, v-45), (u+45, v+45), (0, 255, 0), 3, 2)
cv2.putText(new_img, text, (u-35, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
