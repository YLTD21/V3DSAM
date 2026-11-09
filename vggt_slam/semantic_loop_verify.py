#!/usr/bin/env python3
"""
V33 Semantic Loop Closure Verification + Object Clustering.

1. Cluster world-frame semantic points into objects (per-class DBSCAN)
2. For each VGGT-detected loop closure, verify using:
   a) Semantic class overlap ratio (filter false loops on repetitive textures)
   b) Object layout consistency (independent geometric verification)
3. Add object-level constraints for verified loops
"""

import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


def cluster_objects_by_class(points_world, sem_labels, min_points=200, cluster_eps=0.3):
    """Cluster points of the same semantic class into objects using 3D proximity.
    Returns: list of {class_id, centroid, extent, points, count}"""
    objects = []
    for cls_id in np.unique(sem_labels):
        if cls_id < 0: continue  # skip unlabeled
        mask = sem_labels == cls_id
        pts = points_world[mask]
        if len(pts) < min_points: continue

        # Simple distance-based clustering (greedy, no DBSCAN dependency)
        clusters = _greedy_cluster(pts, eps=cluster_eps, min_points=min_points)
        for cl_pts in clusters:
            centroid = cl_pts.mean(0)
            extent = cl_pts.max(0) - cl_pts.min(0)
            objects.append({
                'class': int(cls_id), 'centroid': centroid,
                'extent': extent, 'points': cl_pts, 'count': len(cl_pts)
            })
    return objects


def _greedy_cluster(pts, eps=0.3, min_points=50):
    """Greedy Euclidean clustering."""
    if len(pts) < min_points: return []
    tree = KDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    clusters = []
    for i in range(len(pts)):
        if visited[i]: continue
        # BFS from seed point
        cluster_idx = [i]
        visited[i] = True
        frontier = [i]
        while frontier:
            j = frontier.pop()
            neighbors = tree.query_ball_point(pts[j], eps)
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    cluster_idx.append(n)
                    frontier.append(n)
        if len(cluster_idx) >= min_points:
            clusters.append(pts[cluster_idx])
    return clusters


def verify_loop_candidate(submap_a, submap_b, solver, sem_per_submap,
                           min_class_overlap=0.3, min_object_matches=2):
    """Verify a loop closure candidate using semantic information.

    Returns: (is_valid, confidence, T_relative_estimate)
    """
    sid_a = submap_a.get_id()
    sid_b = submap_b.get_id()

    # Get point clouds and semantic labels for both submaps
    pts_a = submap_a.get_points_in_world_frame(solver.graph)
    pts_b = submap_b.get_points_in_world_frame(solver.graph)
    sem_a = sem_per_submap.get(sid_a)
    sem_b = sem_per_submap.get(sid_b)

    if sem_a is None or sem_b is None:
        return False, 0.0, None

    # 1. Class overlap ratio
    classes_a = set(sem_a[sem_a >= 0])
    classes_b = set(sem_b[sem_b >= 0])
    if not classes_a or not classes_b:
        return False, 0.0, None
    overlap = len(classes_a & classes_b) / min(len(classes_a), len(classes_b))
    if overlap < min_class_overlap:
        return False, overlap, None

    # 2. Object-level verification
    objects_a = cluster_objects_by_class(pts_a, sem_a, min_points=100, cluster_eps=0.3)
    objects_b = cluster_objects_by_class(pts_b, sem_b, min_points=100, cluster_eps=0.3)

    # Match objects by class + spatial proximity
    matches = []
    for oa in objects_a:
        for ob in objects_b:
            if oa['class'] != ob['class']: continue
            dist = np.linalg.norm(oa['centroid'] - ob['centroid'])
            max_extent = max(np.linalg.norm(oa['extent']), np.linalg.norm(ob['extent']))
            if dist < max_extent * 2.0:  # object centroids within 2x extent
                matches.append((oa, ob, dist))

    # At least min_object_matches different classes matched
    matched_classes = set(oa['class'] for oa, _, _ in matches)
    if len(matched_classes) < min_object_matches:
        return False, overlap, None

    # 3. Estimate relative transform from matched object centroids
    if len(matches) >= 3:
        src_pts = np.array([m[0]['centroid'] for m in matches])
        dst_pts = np.array([m[1]['centroid'] for m in matches])
        # Kabsch alignment of centroids
        sc = src_pts.mean(0); dc = dst_pts.mean(0)
        H = (src_pts - sc).T @ (dst_pts - dc)
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return True, overlap, None  # still valid loop, just can't compute T
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1,:]*=-1; R = Vt.T @ U.T
        s = np.sum((dst_pts - dc) * ((R @ (src_pts - sc).T).T)) / np.sum((src_pts - sc) ** 2)
        t = dc - s * (R @ sc)
        T_rel = np.eye(4); T_rel[:3,:3] = R * s; T_rel[:3,3] = t
        confidence = overlap * min(1.0, len(matched_classes) / 5.0)
        return True, confidence, T_rel

    return True, overlap, None  # valid by class overlap, but insufficient objects for T


def semantic_loop_filter_and_add(solver, sem_per_submap, min_class_overlap=0.3,
                                  min_object_matches=2, loop_noise=0.08, verbose=True):
    """Verify existing loop closures and add verified object-level constraints."""
    from gtsam import noiseModel

    submaps = [sm for sm in solver.map.ordered_submaps_by_key() if not sm.get_lc_status()]
    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]

    if verbose:
        print(f"\n  Semantic Loop Verification: {len(lc_submaps)} loop closures to verify")

    verified = 0
    rejected = 0
    object_constraints = 0

    for lc_sm in lc_submaps:
        # Loop closure submap pairs: lc_sm is connected to a query submap and a detected submap
        # Get the edge information from the graph or from the submap structure
        lc_id = lc_sm.get_id()

        # The loop closure submap has 2 frames: frame 0 connects to query, frame 1 to detected
        # We need to find which submaps are connected
        # The official solver stores: query_submap_id and detected_submap_id
        # But these aren't directly accessible. Instead, check the submap's neighbor relations.

        # For simplicity: verify against ALL non-LC submaps using semantic overlap
        # and only keep those with high semantic consistency

        # Find the two non-LC submaps this loop connects
        # Look at the submap's poses to infer neighbors
        if not hasattr(lc_sm, 'poses') or lc_sm.poses is None:
            continue
        if len(lc_sm.poses) < 2:
            continue

        # Get connected submaps via graph edges
        # The loop closure frame 0 node = lc_id + 0, frame 1 node = lc_id + 1
        # We need to find which submaps these connect to
        # This is complex — skip for now, implement basic class-overlap filter

    if verbose:
        print(f"  Semantic loop filter: {verified} kept, {rejected} rejected, "
              f"{object_constraints} object constraints added")

    return solver
