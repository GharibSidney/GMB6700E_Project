import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load calibration
data = np.load('calibration.npz')
K = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# Load and sort images
image_files = sorted(glob.glob('image_calibration/*.png'))  # or jpg
print(f"Total images: {len(image_files)}")

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Store camera poses and 3D points
camera_poses = []
points3D_all = []

# Initialize first camera at origin
R_prev = np.eye(3)
t_prev = np.zeros((3, 1))
camera_poses.append((R_prev, t_prev))

# Read first image
img_prev = cv2.imread(image_files[0])
gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = sift.detectAndCompute(gray_prev, None)

for i in range(1, len(image_files)):
    print(f"\nProcessing pair: {i-1} -> {i}")
    img_curr = cv2.imread(image_files[i])
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched points
    pts_prev = np.array([kp_prev[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts_curr = np.array([kp_curr[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Undistort
    pts_prev_ud = cv2.undistortPoints(pts_prev.reshape(-1, 1, 2), K, dist_coeffs)
    pts_curr_ud = cv2.undistortPoints(pts_curr.reshape(-1, 1, 2), K, dist_coeffs)

    # Essential matrix
    E, mask = cv2.findEssentialMat(pts_prev_ud, pts_curr_ud, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev_ud, pts_curr_ud, K)

    # Compute absolute pose of current camera
    R_curr = R_rel @ R_prev
    t_curr = t_prev + R_prev @ t_rel
    camera_poses.append((R_curr, t_curr))

    # Triangulate between previous and current
    P1 = K @ np.hstack((R_prev, t_prev))
    P2 = K @ np.hstack((R_curr, t_curr))

    pts4d_h = cv2.triangulatePoints(P1, P2, pts_prev.T, pts_curr.T)
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T
    points3D_all.append(pts3d)

    # Prepare for next iteration
    kp_prev, des_prev = kp_curr, des_curr
    R_prev, t_prev = R_curr, t_curr

points3D_all = np.concatenate(points3D_all, axis=0)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D_all[:,0], points3D_all[:,1], points3D_all[:,2], s=2)

# Plot camera centers
for R, t in camera_poses:
    cam_center = -R.T @ t
    ax.scatter(cam_center[0], cam_center[1], cam_center[2], c='r', marker='^')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Reconstruction from 22 Images')
plt.show()
