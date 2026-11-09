import os
import re
import numpy as np
import matplotlib
import scipy
import time
from PIL import Image
from torchvision import transforms as TF
import torch


def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits (with optional decimal), followed by any non-digit characters, then the extension
        match = re.search(r'\d+(?:\.\d+)?(?=[^\d]*\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')
        
    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]
        P = P[0:3, :]

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    # print("Scale factor from K[2,2]:", scale)
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    
    return K, R, t, scale

def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    all_embs = []

    # Load all images into memory (PIL -> tensor)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    # Stack into a single tensor
    imgs = torch.stack(imgs).to(device)

    # Loop over batches
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

    # Combine into one (N, D) array
    return torch.cat(all_embs, dim=0).numpy()

def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) for a Nx3 point cloud.

    Returns:
        center      : (3,) world-space center of OBB
        extent      : (3,) lengths of OBB along its principal axes
        rotation    : (3,3) rotation matrix (columns = principal axes)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3 points"

    # Remove NaN/inf if any
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    # 1. Compute centroid
    centroid = points.mean(axis=0)

    # 2. PCA on centered points
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # Eigen decomposition (sorted by eigenvalue descending)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotation = eigvecs  # columns = principal axes (R)

    # 3. Project points to PCA frame
    points_local = centered @ rotation

    # 4. Compute min/max in PCA frame → box extents
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    # 5. Compute box center (local), then world
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation

def overlay_masks(image, masks, mode="fill", alpha=0.5, line_width=5):
    """
    Overlay masks on image with different visualization modes.
    
    Args:
        image: PIL Image
        masks: torch.Tensor of shape (N, 1, H, W) or (N, H, W)
        mode: Visualization mode
            - "fill": Semi-transparent fill (default)
            - "contour": Only contour lines
            - "fill_contour": Fill + contour lines
            - "heatmap": Heatmap style overlay
            - "blend": Blend mode
        alpha: Transparency for fill modes (0.0-1.0)
        line_width: Width for contour lines
    
    Returns:
        PIL Image with masks overlaid
    """
    import cv2
    
    image = image.convert("RGBA")
    image_np = np.array(image)
    
    # masks: (N, 1, H, W) or (N, H, W)
    masks = (255 * masks.cpu().numpy()).astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        # Now mask is shape (H, W)
        
        if mode == "fill":
            # Semi-transparent fill
            mask_pil = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha_mask = mask_pil.point(lambda v: int(v * alpha))
            overlay.putalpha(alpha_mask)
            image = Image.alpha_composite(image, overlay)
        
        elif mode == "contour":
            # Only contour lines
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_with_contours = image_np.copy()
            cv2.drawContours(img_with_contours, contours, -1, color, line_width)
            image = Image.fromarray(img_with_contours)
        
        elif mode == "fill_contour":
            # Fill + contour lines
            # First add fill
            mask_pil = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha_mask = mask_pil.point(lambda v: int(v * alpha))
            overlay.putalpha(alpha_mask)
            image = Image.alpha_composite(image, overlay)
            # Then add contour
            image_np = np.array(image)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_with_contours = image_np.copy()
            cv2.drawContours(img_with_contours, contours, -1, color, line_width)
            image = Image.fromarray(img_with_contours)
        
        elif mode == "heatmap":
            # Heatmap style with gradient
            mask_float = mask.astype(float) / 255.0
            heatmap = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            for c in range(3):
                heatmap[:, :, c] = (mask_float * color[c]).astype(np.uint8)
            heatmap[:, :, 3] = (mask_float * 200).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap, "RGBA")
            image = Image.alpha_composite(image, heatmap_pil)
        
        elif mode == "blend":
            # Blend mode - more subtle
            mask_float = mask.astype(float) / 255.0
            blended = image_np.copy().astype(float)
            for c in range(3):
                blended[:, :, c] = (1 - mask_float * alpha) * blended[:, :, c] + mask_float * alpha * color[c]
            image = Image.fromarray(blended.astype(np.uint8))
    
    return image


def add_labels_and_scores(image, masks, scores, query):
    """
    Add text labels and confidence scores to the image.
    
    Args:
        image: PIL Image
        masks: torch.Tensor of masks
        scores: torch.Tensor of confidence scores
        query: Text query
    
    Returns:
        PIL Image with labels added
    """
    import cv2
    
    image_np = np.array(image.convert("RGB"))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    masks_np = (255 * masks.cpu().numpy()).astype(np.uint8)
    
    for i in range(masks_np.shape[0]):
        mask = np.squeeze(masks_np[i])
        score = float(scores[i].cpu().numpy())
        
        # Find mask center
        y_indices, x_indices = np.where(mask > 128)
        if len(y_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            
            # Draw label background
            label_text = f"{query} ({score:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                image_np,
                (center_x - 5, center_y - text_height - 10),
                (center_x + text_width + 5, center_y + baseline),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                image_np,
                label_text,
                (center_x, center_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_np)

class Accumulator:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)
