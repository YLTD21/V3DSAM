import os
import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
from termcolor import colored
from scipy.linalg import rq

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from vggt_slam.slam_utils import compute_image_embeddings, Accumulator
from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.graph import PoseGraph
from vggt_slam.scale_solver import estimate_scale_pairwise
from vggt_slam.viewer import Viewer

DEBUG = False

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        intra_noise_sigma: float = 0.05,  # V27: tighten inter-submap constraints
        overlap_count: int = 1):  # V27: overlap frames for cross-constraints

        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.overlap_count = overlap_count

        self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph(intra_noise_sigma=intra_noise_sigma)

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()
        # Global semantic category mapping (shared across all submaps)
        self._global_cat_id_map = {}
        self._global_next_cat_id = 0
        self._global_cat_names = []
        self._semantic_aligner = None  # injected by v2 wrapper

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames()
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T # TODO Dominic check if we want to use P_prior here
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None,
                 is_loop_closure=False, overlap_count=1, submap_size=16):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            prior_submap = self.map.get_submap(submap_id_prev)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
            good_mask = good_mask.reshape(-1)

            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < 100: # Handle the case where loop closure frames do not have enough points. 
                    good_mask = (prior_conf > 0).reshape(-1)

            # V27 FIX: use frame_id_prev (first overlap frame) instead of proj_mats[-1] (last frame)
            P_temp = np.linalg.inv(prior_submap.proj_mats[frame_id_prev]) @ current_submap.proj_mats[0]
            t1 = (P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output)
            scale_factor = scale_factor_est_output[0]
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            # V27 FIX: use frame_id_prev (first overlap frame), not proj_mats[-1]
            H_overlap_prior_overlap_current = np.linalg.inv(
                prior_submap.proj_mats[frame_id_prev]) @ current_submap.proj_mats[0] @ H_scale
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for the first overlap frame pair.
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, self.graph.intra_submap_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        # V29: Compute per-frame depth confidence weights
        _conf_frames = []
        for index in range(len(world_to_cam)):
            try:
                _c = current_submap.get_conf_masks_frame(index)
                _conf_frames.append(float(np.mean(_c)) if _c is not None else 0.5)
            except Exception:
                _conf_frames.append(0.5)

        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose)
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # V29: Depth-confidence-weighted inner constraint
            _avg_conf = np.clip((_conf_frames[index-1] + _conf_frames[index]) / 2.0, 0.3, 1.0)
            _weighted_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.ones(15) * (self.graph.inner_noise_sigma / _avg_conf))
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, _weighted_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            # V27 FIX: Connect at the FIRST overlap frame, not the LAST.
            # With O=1: frame_id_prev = S (same as last=S+O-1=16), OK.
            # With O=8: old code used frame 23→0 (7-frame gap, missing geometry);
            #           now uses frame 16→0 (same physical frame, correct).
            last_frame = self.map.get_latest_submap(
                ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()
            frame_id_prev = last_frame - self.overlap_count + 1  # = S (first overlap frame)

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, K_4x4)
        self.current_working_submap.set_conf_masks(conf)
        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        # V27: pass submap_size so add_edge can add overlap cross-constraints
        submap_size = len(self.current_working_submap.poses) - self.overlap_count
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev,
                      is_loop_closure=False, overlap_count=self.overlap_count,
                      submap_size=submap_size)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"]) 
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]
            

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc)
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)
            self.map.add_submap(lc_submap)

            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True)

            # ── Semantic alignment constraint for loop closure (Phase 3) ──
            if self._semantic_aligner is not None:
                query_sub = self.map.get_submap(loop.query_submap_id)
                det_sub = self.map.get_submap(loop.detected_submap_id)
                if query_sub.has_semantic and det_sub.has_semantic:
                    try:
                        T_sem, sem_conf = self._semantic_aligner.compute_semantic_relative_transform(
                            query_sub, det_sub, self.graph)
                        if T_sem is not None and sem_conf > 0.3:
                            noise_scale = max(0.01, 0.05 * (1.0 - sem_conf))
                            sem_noise = gtsam.noiseModel.Diagonal.Sigmas(
                                np.array([noise_scale]*15, dtype=float))
                            T_sem_sl4 = T_sem / (np.linalg.det(T_sem) ** 0.25)
                            det_node_id = int(det_sub.get_id() + loop.detected_submap_frame)
                            query_node_id = int(query_sub.get_id() + loop.query_submap_frame)
                            self.graph.graph.add(
                                gtsam.BetweenFactorSL4(
                                    gtsam.symbol_shorthand.X(query_node_id),
                                    gtsam.symbol_shorthand.X(det_node_id),
                                    gtsam.SL4(T_sem_sl4), sem_noise))
                            print(colored(
                                f"  [SemAlign] Semantic constraint added "
                                f"(conf={sem_conf:.2f}, noise={noise_scale:.3f})", 'cyan'))
                    except Exception as e:
                        print(f"  [SemAlign WARN] {e}")

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess, sam3_engine=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        with torch.no_grad():
            t1 = time.time()
            with self.vggt_timer:
                predictions = model(images)
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)
                predictions_lc = model(lc_frames, compute_similarity=True)
                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame), 
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops
        
        if predictions_lc is not None:
            image_match_ratio = predictions_lc["image_match_ratio"]
            if image_match_ratio < -0.95:
                print(colored("Loop closure image match ratio too low, skipping loop closure", "red"))
                predictions_lc = None # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
            else:
                self.graph.increment_loop_closure()
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
                predictions["extrinsic_lc"] = extrinsic_lc
                predictions["intrinsic_lc"] = intrinsic_lc
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]

            
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    
        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        # ── SAM3 semantic inference with VGGT-guided keyframe selection ──
        if sam3_engine is not None:
            t_sam3 = time.time()
            try:
                from PIL import Image as PILImage
                S_total = len(image_names)
                H_img, W_img = images.shape[2], images.shape[3]

                # VGGT-guided keyframe selection
                vggt_extrinsic = predictions.get("extrinsic")
                kf_indices = [0]
                if vggt_extrinsic is not None and len(vggt_extrinsic) > 1:
                    for i in range(1, len(vggt_extrinsic)):
                        prev_pose = vggt_extrinsic[i-1]
                        curr_pose = vggt_extrinsic[i]
                        trans = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])
                        R_rel = curr_pose[:3, :3] @ prev_pose[:3, :3].T
                        rot_deg = np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)) * 180/np.pi
                        if trans > 0.03 or rot_deg > 1.0:
                            kf_indices.append(i)
                    if kf_indices[-1] != S_total - 1:
                        kf_indices.append(S_total - 1)
                else:
                    kf_indices = list(range(S_total))

                print(f"  [SAM3] VGGT keyframes: {len(kf_indices)}/{S_total} "
                      f"({(1-len(kf_indices)/S_total)*100:.0f}% skipped)")

                pil_images_all = [PILImage.open(p).convert("RGB") for p in image_names]
                kf_images = [pil_images_all[i] for i in kf_indices]
                kf_masks = sam3_engine.segment_frames(kf_images, mode="auto")

                sem_ids = np.full((S_total, H_img, W_img), -1, dtype=np.int32)
                sem_confs = np.zeros((S_total, H_img, W_img), dtype=np.float32)

                # Pre-compute VGGT preprocessing params for each image to align masks
                # VGGT does: resize to width=518, round height to /14, center-crop if >518
                _vggt_params = {}
                for fi, img_path in enumerate(image_names):
                    img = PILImage.open(img_path)
                    orig_w, orig_h = img.size
                    new_w = 518
                    new_h = round(orig_h * new_w / orig_w / 14) * 14
                    _vggt_params[fi] = (orig_w, orig_h, new_w, new_h)

                for kf_local, kf_global in enumerate(kf_indices):
                    frame_masks = kf_masks[kf_local]
                    next_kf_global = kf_indices[kf_local+1] if kf_local+1 < len(kf_indices) else S_total
                    for frame_idx in range(kf_global, min(next_kf_global, S_total)):
                        orig_w, orig_h, new_w, new_h = _vggt_params[frame_idx]
                        for m in frame_masks:
                            mask_2d = m["mask"]  # at SAM3/original resolution
                            # Mirror VGGT's image preprocessing for the mask
                            mask_pil = PILImage.fromarray(mask_2d.astype(np.uint8)*255)
                            mask_pil = mask_pil.resize((new_w, new_h), PILImage.NEAREST)
                            if new_h > 518:  # center-crop height (VGGT crop mode)
                                start_y = (new_h - 518) // 2
                                mask_pil = mask_pil.crop((0, start_y, new_w, start_y + 518))
                            mask_2d = np.array(mask_pil) > 128
                            # Final safety: if still wrong size, resize to VGGT grid
                            if mask_2d.shape[:2] != (H_img, W_img):
                                mask_pil = PILImage.fromarray(mask_2d.astype(np.uint8)*255)
                                mask_pil = mask_pil.resize((W_img, H_img), PILImage.NEAREST)
                                mask_2d = np.array(mask_pil) > 128
                            if m["category"] not in self._global_cat_id_map:
                                self._global_cat_id_map[m["category"]] = self._global_next_cat_id
                                self._global_cat_names.append(m["category"])
                                self._global_next_cat_id += 1
                            cid = self._global_cat_id_map[m["category"]]
                            update = mask_2d & ((sem_ids[frame_idx] < 0) | (sem_confs[frame_idx] < m["score"]))
                            sem_ids[frame_idx][update] = cid
                            sem_confs[frame_idx][update] = m["score"]

                new_submap.set_semantic_data(sem_ids, sem_confs, self._global_cat_names.copy())
                total_masks = sum(len(fm) for fm in kf_masks)

                # Save per-frame semantic masks to disk for Phase 2 (relabeling + mask loss)
                if hasattr(self, '_sem_mask_dir') and self._sem_mask_dir:
                    os.makedirs(self._sem_mask_dir, exist_ok=True)
                    # Palette for visualization (tab20-like)
                    PAL_VIS = np.array([
                        [31,119,180],[255,127,14],[44,160,44],[214,39,40],
                        [148,103,189],[140,86,75],[227,119,194],[127,127,127],
                        [188,189,34],[23,190,207],[174,199,232],[255,187,120],
                        [152,223,138],[255,152,150],[197,176,213],[196,156,148],
                        [247,182,210],[199,199,199],[219,219,141],[158,218,229],
                    ], dtype=np.uint8)
                    saved = 0
                    for i, img_path in enumerate(image_names[:S_total]):
                        frame_sem = sem_ids[i].copy()
                        H, W = frame_sem.shape
                        rgb = np.full((H, W, 3), 128, dtype=np.uint8)  # gray bg
                        labeled = frame_sem >= 0
                        if labeled.any():
                            cids = frame_sem[labeled] % len(PAL_VIS)
                            rgb[labeled] = PAL_VIS[cids]
                        mask_img = PILImage.fromarray(rgb, mode='RGB')
                        fname = os.path.basename(img_path)
                        fname = os.path.splitext(fname)[0] + '_sem.png'
                        mask_img.save(os.path.join(self._sem_mask_dir, fname))
                        saved += 1
                    print(f"  [SAM3] Saved {saved} semantic masks to {self._sem_mask_dir}")

                print(f"  [SAM3] {total_masks} masks, {len(self._global_cat_names)} categories, "
                      f"{time.time()-t_sam3:.1f}s")
            except Exception as e:
                print(f"  [SAM3 WARN] Inference failed: {e}")

        return predictions