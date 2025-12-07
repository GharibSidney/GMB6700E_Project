import numpy as np
pred = np.load('input_images_20251206_223356_852312/predictions.npz')
print(pred.files)

Ks = pred["intrinsic"]         # (N, 3, 3)
extrinsics = pred["extrinsic"] # (N, 3, 4)
world_points = pred["world_points"]  # (N, H, W, 3)

N, H, W, _ = world_points.shape

u, v = np.meshgrid(np.arange(W), np.arange(H))
pixels_obs = np.stack([u, v], axis=-1).reshape(-1, 2)

def project_world_points(Pw, K, extr):
    """
    Pw: (M, 3) world coordinates
    extr: (3,4) world-to-camera matrix [R|t]
    """

    R = extr[:, :3]
    t = extr[:, 3]

    Pc = (R @ Pw.T).T + t

    z = Pc[:, 2]
    valid = z > 1e-6

    x = Pc[:, 0] / z
    y = Pc[:, 1] / z

    pts = np.stack([x, y, np.ones_like(x)], axis=0)

    proj = (K @ pts).T
    proj = proj[:, :2]

    return proj, valid

all_errors = []

for i in range(N):
    Pw = world_points[i].reshape(-1, 3)
    K = Ks[i]
    extr = extrinsics[i]

    proj_pixels, valid = project_world_points(Pw, K, extr)

    obs = pixels_obs[valid]
    proj = proj_pixels[valid]

    err = np.linalg.norm(proj - obs, axis=1)
    all_errors.append(err)

all_errors = np.concatenate(all_errors)

print("Mean reprojection error:", all_errors.mean())
print("Median reprojection error:", np.median(all_errors))
print("90th percentile:", np.percentile(all_errors, 90))
print("Max error:", all_errors.max())
