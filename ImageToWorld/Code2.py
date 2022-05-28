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
images = glob.glob('ImageToWorld/Images/*.jpg')
resolution = (500, 666)
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, resolution, interpolation=None)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.set_printoptions(suppress=True)

#print(objpoints[0])
#print(imgpoints[0])

"""
rotation_mat = np.zeros(shape=(3, 3))
for rvec in range(14):
    R = cv2.Rodrigues(rvecs[rvec], rotation_mat)[0]
    Trans_mat = np.column_stack((np.matmul(mtx, R), tvecs[rvec]))
  #  Trans_mat = np.column_stack((R, tvecs[rvec]))
print("Transformation Matrix = \n", Trans_mat)
"""

rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
#Trans_mat = np.column_stack((np.matmul(mtx, R), tvecs[0]))
Trans_mat = np.column_stack((R, tvecs[0]))
print("Transformation Matrix = \n", Trans_mat)

new_img = cv2.imread('ImageToWorld/Images/IMG20201028222432.jpg')
new_img = cv2.resize(new_img, resolution, interpolation=None)
#u=int(mtx.item(2))
#v=int(mtx.item(5))

"""
x = 0
y = 0
z = 0
s = 32.03988519711839
world_coordinates=[x, y, z, 1]
Final_product = mtx.dot(Trans_mat)
pixel_coordinates = (Final_product.dot(world_coordinates))/s
print('\nImage Coordinates = ', pixel_coordinates)
"""

u = 334.7013
v = 488.5621

"""
print("u = ", u, "v = ", v)
text = "x: " + str(u) + ", y: " + str(v)
cv2.circle(new_img, (u, v), 3, (255, 200, 50), -1)
cv2.rectangle(new_img, (u-45, v-45), (u+45, v+45), (0, 255, 0), 3, 2)
cv2.putText(new_img, text, (u-35, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Final_product = Trans_mat
Final_product = mtx.dot(Trans_mat)
print('\nFinal_product = \n', Final_product)
Inverse_mat = np.linalg.pinv(Final_product, rcond=1e-15)
print('\nInverse_Matrix = \n', Inverse_mat)
s = 31.46351930790326
world_coordinates = s*(Inverse_mat.dot(np.array([u, v, 1])))
print('\nWorld Coordinates = ', world_coordinates)
pixel_coordinates = (Final_product.dot(world_coordinates))/s
print('\nImage Coordinates = ', pixel_coordinates)
