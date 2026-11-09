import numpy as np
import open3d as o3d

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

def estimate_scale_pairwise(X, Y, DEBUG=False):
    assert X.shape == Y.shape
    # Camera-frame points: distances from origin = depths.
    # Ratio of depths between same-scene point clouds from two VGGT batches
    # gives the inter-submap scale factor.
    x_dists = np.linalg.norm(X, axis=1)
    y_dists = np.linalg.norm(Y, axis=1)
    # Avoid division by zero or near-zero depths
    valid = (x_dists > 1e-6) & (y_dists > 1e-6) & np.isfinite(x_dists) & np.isfinite(y_dists)
    if valid.sum() < 50:
        return 1.0, None
    scales = y_dists[valid] / x_dists[valid]
    # ── IQR-based outlier rejection ──
    q25, q75 = np.percentile(scales, [25, 75])
    iqr = q75 - q25
    lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    inlier = (scales >= lo) & (scales <= hi)
    if inlier.sum() >= 20:
        scale = np.median(scales[inlier])
    else:
        scale = np.median(scales)
    # ── Safety clip ──
    scale = np.clip(scale, 0.7, 1.3)
    if DEBUG:
        debug_visualize(X * scale, Y)
    return scale, None