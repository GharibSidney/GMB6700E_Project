# Source - https://stackoverflow.com/a
# Posted by N.S
# Retrieved 2025-11-06, License - CC BY-SA 4.0

import numpy as np
from numpy import load
import cv2

data = load('calibration.npz')
lst = data.files
for item in lst:
    print(item)
    if  item == "camera_matrix" or item == "rvecs" or item == "tvecs":
        print(data[item].shape)
    if item == "camera_matrix":
        print(data[item])
    
    if item == "rvecs":
        rvecs = data[item]
    elif item == "tvecs":
        tvecs = data[item]
    elif item == "camera_matrix":
        intrinsic_camera_matrix = data[item]


# inputs you should already have:
# objpoints_list: list of objectPoints for each image (len N)
# imgpoints_list: list of corresponding imagePoints for each image (len N)
# rvecs, tvecs, camera_matrix, dist_coeffs from calibration

N = len(rvecs)
errors = np.zeros(N)

for i in range(N):
    imgpoints_proj, _ = cv2.projectPoints(
        objpoints_list[i], rvecs[i], tvecs[i], intrinsic_camera_matrix, dist_coeffs
    )
    imgpoints_proj = imgpoints_proj.reshape(-1,2)
    imgpoints = np.asarray(imgpoints_list[i]).reshape(-1,2)
    # mean reprojection error for this image
    errors[i] = np.sqrt(np.mean(np.sum((imgpoints - imgpoints_proj)**2, axis=1)))

best_idx = int(np.argmin(errors))
best_rvec = rvecs[best_idx].reshape(3)
best_tvec = tvecs[best_idx].reshape(3)
print("best index:", best_idx, "error(px):", errors[best_idx])
