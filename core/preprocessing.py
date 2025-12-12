import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional
from PIL import Image

moge_path = os.path.join(os.path.dirname(__file__), '../../MoGe')
sys.path.insert(0, moge_path)
from baselines.moge import Baseline as MoGeBaseline
from moge.utils.image_utils import load_image


def estimate_depth_map(rgb_image: np.ndarray, moge_model_path: Optional[str] = None, device: str = 'cuda') -> np.ndarray:
    if moge_model_path is None:
        moge_model_path = 'Ruicheng/moge-vitl'
    
    moge_model = MoGeBaseline.load(
        num_tokens=None,
        resolution_level=9,
        pretrained_model_name_or_path=moge_model_path,
        use_fp16=True,
        device=device,
        version='v1'
    )
    
    if isinstance(rgb_image, np.ndarray):
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
    
    image_tensor, _, f_px = load_image(rgb_image, device=device)
    
    with torch.no_grad():
        output = moge_model.infer(image_tensor, intrinsics=None)
        depth_map = output['depth_scale_invariant'].cpu().numpy()
    
    if len(depth_map.shape) > 2:
        depth_map = depth_map.squeeze()
    
    return depth_map.astype(np.float32)


def estimate_3d_bbox(
    rgb_image: np.ndarray,
    text_prompt: str,
    depth_map: Optional[np.ndarray] = None,
    ovmono3d_model_path: Optional[str] = None,
    device: str = 'cuda'
) -> np.ndarray:
    if depth_map is not None and depth_map.size > 0:
        H, W = depth_map.shape
        center_y, center_x = H // 2, W // 2
        
        center_depth = depth_map[center_y, center_x]
        if center_depth <= 0 or center_depth > 100:
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 0:
                center_depth = np.median(valid_depths)
            else:
                center_depth = 5.0
        
        fx = fy = W
        cx, cy = W / 2, H / 2
        
        x_3d = (center_x - cx) * center_depth / fx
        y_3d = (center_y - cy) * center_depth / fy
        z_3d = center_depth
        
        region_size = min(H, W) // 4
        y_start = max(0, center_y - region_size)
        y_end = min(H, center_y + region_size)
        x_start = max(0, center_x - region_size)
        x_end = min(W, center_x + region_size)
        
        region_depths = depth_map[y_start:y_end, x_start:x_end]
        valid_region = region_depths[region_depths > 0]
        
        if len(valid_region) > 0:
            depth_std = np.std(valid_region)
            scale = max(0.5, min(2.0, depth_std * 2))
        else:
            scale = 1.0
        
        translation = np.array([x_3d, -y_3d, -z_3d], dtype=np.float32)
        rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale_3d = np.array([scale, scale, scale], dtype=np.float32)
        
        bbox_3d = np.concatenate([translation, rotation, scale_3d])
    else:
        bbox_3d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    return bbox_3d


def extract_geometric_inputs(
    rgb_image: np.ndarray,
    text_prompt: str,
    moge_model_path: Optional[str] = None,
    ovmono3d_model_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    depth_map = estimate_depth_map(rgb_image, moge_model_path)
    bbox_3d = estimate_3d_bbox(rgb_image, text_prompt, depth_map, ovmono3d_model_path)
    return depth_map, bbox_3d
