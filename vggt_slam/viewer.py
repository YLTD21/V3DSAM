import time
from typing import Dict, List
import numpy as np
import torch
import viser
import viser.transforms as viser_tf


class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- GUI Elements ---
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Add a button to trigger the walkthrough
        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        np.random.seed(100)
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()
        self.obj_id = 0

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=self.random_colors[self.submap_id_to_color[submap_id]]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def visualize_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color = (255, 0, 0),
        line_width: float = 2.0,
        mode: str = "wireframe",
        label: str = "",
        score: float = 0.0,
    ):
        """
        Visualize an oriented bounding box (OBB) in Viser with multiple modes.

        Parameters
        ----------
        center : (3,) array
            World-space center of the OBB.
        extent : (3,) array
            Full side lengths of the OBB (dx, dy, dz).
        rotation : (3,3) array
            Rotation matrix of the OBB in world coordinates.
        color : tuple[int,int,int]
            RGB color of the wireframe box.
        line_width : float
            Thickness of the box edges.
        mode : str
            Visualization mode:
            - "wireframe": Only wireframe edges (default)
            - "solid": Solid box with thicker lines
            - "wireframe_label": Wireframe with 3D text label
            - "highlight": Highlighted wireframe with glow effect
            - "arrow": Wireframe with direction arrow
        label : str
            Text label to display (e.g., object name)
        score : float
            Confidence score for the detection

        Notes
        -----
        The box is drawn with the specified visualization mode.
        """

        # Compute local corners (8)
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float32)  # shape (8,3)

        # Transform to world
        corners_world = (rotation @ corners_local.T).T + center  # shape (8,3)

        # Build edges (12 line segments) as start/end pairs
        edges_idx = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7)   # vertical edges
        ]

        segments = []
        for (i,j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        # segments is list of length 24, reshape into (N,2,3)
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        name = f"obb_{self.obj_id}"
        self.obj_id += 1

        if mode == "wireframe":
            # Default: simple wireframe
            self.server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=color,
                line_width=line_width,
                visible=True
            )
        
        elif mode == "solid":
            # Solid box with thicker lines
            self.server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=color,
                line_width=line_width * 3,
                visible=True
            )
        
        elif mode == "wireframe_label":
            # Wireframe with 3D text label
            self.server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=color,
                line_width=line_width,
                visible=True
            )
            # Add label above the box
            label_text = f"{label} ({score:.2f})" if label else f"Object ({score:.2f})"
            label_pos = center + np.array([0, 0, dz + 0.05])
            self.server.scene.add_label(
                name=f"{name}_label",
                text=label_text,
                position=label_pos,
            )
        
        elif mode == "highlight":
            # Highlighted wireframe with multiple layers for glow effect
            # Inner bright line
            self.server.scene.add_line_segments(
                name=f"{name}_inner",
                points=segments,
                colors=(255, 255, 255),
                line_width=line_width * 0.5,
                visible=True
            )
            # Outer colored line
            self.server.scene.add_line_segments(
                name=f"{name}_outer",
                points=segments,
                colors=color,
                line_width=line_width * 2,
                visible=True
            )
        
        elif mode == "arrow":
            # Wireframe with direction arrow
            self.server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=color,
                line_width=line_width,
                visible=True
            )
            # Add arrow from center pointing up
            arrow_start = center
            arrow_end = center + rotation @ np.array([0, 0, dz * 1.5])
            arrow_points = np.array([arrow_start, arrow_end], dtype=np.float32)
            self.server.scene.add_line_segments(
                name=f"{name}_arrow",
                points=arrow_points.reshape(1, 2, 3),
                colors=(0, 255, 0),
                line_width=line_width * 1.5,
                visible=True
            )

    def run_walkthrough(self, fps: float = 20.0):
            """
            Walks through the map using the current live positions of all frames.
            This accounts for loop closures because it pulls data from the scene handles.
            """
            # 1. Gather all submap IDs and sort them to ensure a logical sequence
            sorted_submap_ids = sorted(self.submap_frames.keys())
            
            if not sorted_submap_ids:
                print("No frames found to walk through.")
                return

            clients = self.server.get_clients()
            if not clients:
                print("No clients connected to perform walkthrough.")
                return

            print("Starting walkthrough of updated poses...")

            for sub_id in sorted_submap_ids:
                frames = self.submap_frames[sub_id]
                # Assumes frames were added in chronological order to the list
                for frame_handle in frames:
                    # Get the current world-space pose from the visualizer
                    # If a loop closure moved the submap, these values will be updated
                    current_pos = frame_handle.position
                    current_wxyz = frame_handle.wxyz

                    # Update all connected clients
                    for client in clients.values():
                        client.camera.position = current_pos
                        client.camera.wxyz = current_wxyz
                    
                    # Control speed (1/fps)
                    time.sleep(1.0 / fps)