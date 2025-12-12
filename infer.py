import os
import cv2
import tyro
import glob
import time
import json
import math
import shutil
import numpy as np
import torch
from PIL import Image
import seaborn as sns
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
import trimesh
from kiui.op import recenter

from core.options import AllConfigs, Options
from core.models import LMM
from core.utils import monkey_patch_transformers
from core.utils import camera_to_token, camera_to_token_single, token_to_camera, quaternion_to_matrix, matrix_to_quaternion, quaternion_slerp, sample_from_two_pose, sample_from_dense_cameras
from core.preprocessing import extract_geometric_inputs, estimate_depth_map, estimate_3d_bbox

from util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt

monkey_patch_transformers()

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)

# model
model = LMM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sane?')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)

def draw_json(c2ws, vis_path):
    output_dir = os.path.dirname(vis_path)
    parent_dir = os.path.dirname(output_dir)

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])
    num_matrices = c2ws.shape[0]

    # Create a color gradient from red to purple
    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{parent_dir}/{view['name']}_view.png"
        os.makedirs(output_dir, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
    
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")


def process_data(opt, output_dir, name, image_path, text=None, text_path=None, depth_path=None, bbox_3d_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    new_traj_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
    if os.path.exists(new_traj_path):
        print(f"Skipping {name} as it already exists.")
        return

    # Load or extract text
    if text is None:
        if text_path is not None and os.path.exists(text_path):
            info = json.load(open(text_path, 'r'))
            if opt.cond_mode == 'text':
                text_key = opt.text_key if opt.text_key else 'text'
            else:
                text_key = 'Concise Interaction' if 'Concise Interaction' in info else 'text'
            text = info.get(text_key, '')
        else:
            text = opt.text if opt.text else ''
    

    def standard_image(rgb_path, target_height=512, target_width=512):
        image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if image.shape[2] == 4:  # RGBA
            image = image[..., :3]  # Remove alpha
        image = image[..., [2, 1, 0]]  # BGR to RGB
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        height, width = image_tensor.shape[1], image_tensor.shape[2]

        if height > target_height:
            start_y = (height - target_height) // 2
            image_tensor = image_tensor[:, start_y:start_y + target_height, :]
        
        if width > target_width:
            start_x = (width - target_width) // 2
            image_tensor = image_tensor[:, :, start_x:start_x + target_width]

        if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
            padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
            
            top_padding = (target_height - image_tensor.shape[1]) // 2
            bottom_padding = target_height - image_tensor.shape[1] - top_padding
            left_padding = (target_width - image_tensor.shape[2]) // 2
            right_padding = target_width - image_tensor.shape[2] - left_padding

            padded_image[:, top_padding:top_padding + image_tensor.shape[1], left_padding:left_padding + image_tensor.shape[2]] = image_tensor
            image_tensor = padded_image
        return image_tensor

    def standard_depth(depth_path, target_height=512, target_width=512):
        if depth_path.endswith('.npy'):
            depth_image = np.load(depth_path).astype(np.float32)
        else:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if len(depth_image.shape) == 3:
                depth_image = depth_image[:, :, 0]  # Take first channel
        
        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()
        height, width = depth_tensor.shape[1], depth_tensor.shape[2]

        if height > target_height:
            start_y = (height - target_height) // 2
            depth_tensor = depth_tensor[:, start_y:start_y + target_height, :]

        if width > target_width:
            start_x = (width - target_width) // 2
            depth_tensor = depth_tensor[:, :, start_x:start_x + target_width]

        if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
            padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
            
            top_padding = (target_height - depth_tensor.shape[1]) // 2
            bottom_padding = target_height - depth_tensor.shape[1] - top_padding
            left_padding = (target_width - depth_tensor.shape[2]) // 2
            right_padding = target_width - depth_tensor.shape[2] - left_padding

            padded_depth[:, top_padding:top_padding + depth_tensor.shape[1], left_padding:left_padding + depth_tensor.shape[2]] = depth_tensor
            depth_tensor = padded_depth

        return depth_tensor

    # Load RGB image
    rgb = None
    rgb_batch = None
    if image_path is not None and opt.cond_mode != 'text':
        rgb = standard_image(image_path, target_height=opt.target_height, target_width=opt.target_width).to(device)
        rgb_show = rgb.permute(1, 2, 0).cpu().numpy()
        rgb_batch = rgb.unsqueeze(0)
        kiui.write_image(os.path.join(output_dir, f"{name}_rgb.png"), rgb_show)

    depth = None
    depth_batch = None
    bbox_3d = None
    
    if opt.cond_mode == 'text+rgbd+bbox':
        if depth_path is not None and os.path.exists(depth_path):
            depth = standard_depth(depth_path, target_height=opt.target_height, target_width=opt.target_width)
        else:
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
            
            depth_map = estimate_depth_map(rgb_np, device=device)
            depth = torch.from_numpy(depth_map).unsqueeze(0).float()
            depth = F.interpolate(depth.unsqueeze(0), size=(opt.target_height, opt.target_width), mode='bilinear', align_corners=False).squeeze(0)
        
        depth_batch = depth.unsqueeze(0).to(device)
        
        depth_show = depth.squeeze().cpu().numpy()
        plt.figure(figsize=(12, 12))
        sns.heatmap(depth_show, cmap='viridis')
        plt.savefig(os.path.join(output_dir, f"{name}_depth.png"))
        plt.close()
        
        if bbox_3d_path is not None and os.path.exists(bbox_3d_path):
            bbox_3d = np.load(bbox_3d_path)
        else:
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
            
            depth_np = depth.squeeze().cpu().numpy()
            bbox_3d = estimate_3d_bbox(rgb_np, text, depth_map=depth_np, device=device)
        
        bbox_3d = torch.from_numpy(bbox_3d).float().unsqueeze(0).to(device)
        
    elif opt.cond_mode == 'depth+image+text':
        if depth_path is not None:
            depth = standard_depth(depth_path, target_height=opt.target_height, target_width=opt.target_width)
            depth_batch = depth.unsqueeze(0).to(device)

    if opt.cond_mode == 'text':
        conds = [text]
    elif opt.cond_mode == 'image':
        conds = rgb_batch
    elif opt.cond_mode == 'image+text':
        conds = [[text], rgb_batch]
    elif opt.cond_mode == 'image+depth':
        conds = [depth_batch, rgb_batch]
    elif opt.cond_mode == 'depth+image+text':
        conds = [[text], rgb_batch, depth_batch]
    elif opt.cond_mode == 'text+rgbd+bbox':
        conds = [[text], rgb_batch, depth_batch, bbox_3d]
    else:
        raise ValueError(f"Unsupported cond_mode: {opt.cond_mode}")

    # Generate trajectory
    for i in range(opt.test_repeat):
        t0 = time.time()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tokens = model.generate(conds, max_new_tokens=opt.test_max_seq_length, clean=True)
        t1 = time.time()
        print(f'[INFO] Processing, time = {t1 - t0:.4f}s')
        
        token = tokens[0]
        if len(token) < opt.pose_length * 10:
            token = torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60]) / 256 * opt.discrete_bins
            token = token.repeat(opt.pose_length)
            coords = token.reshape(-1, 10)
        else:
            coords = token[:opt.pose_length * 10].reshape(-1, 10)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        discrete_bins = opt.discrete_bins

        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1] if coords_instri.shape[1] > 0 else torch.zeros(coords_traj.shape[0])

        temp_traj = coords_traj / (0.5 * discrete_bins) - 1
        temp_instri = coords_instri / (discrete_bins / 10) if coords_instri.shape[1] > 0 else torch.zeros_like(coords_traj[:, :2])
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2) if coords_scale.numel() > 0 else torch.ones(1)

        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.expand(1, -1, -1)
        camera_pose = token_to_camera(camera_tokens, opt.target_width, opt.target_height)
        
        c2ws = np.array(camera_pose[:, :, :12].cpu())
        scale_value = np.array(scale[0].cpu()) if scale.numel() > 0 else 1.0
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * scale_value

        row_to_add = np.array([0, 0, 0, 1])
        c2ws = np.array([np.vstack((matrix, row_to_add)) for matrix in c2ws])
        
        def pose_normalize(camera_pose, pred_pose_path):
            camera_pose = camera_pose
            transforms_path = pred_pose_path

            f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()
            transforms_dict = {
                "w": w,
                "h": h,
                "fl_x": f_x,
                "fl_y": f_y,
                "cx": c_x,
                "cy": c_y,
                'frames': []
            }
            traj_tensor = camera_pose[:,:,:12]
            camera_list = []
            for i in range(120):
                t = torch.full((1, 1), fill_value=i/120)
                camera = sample_from_dense_cameras(traj_tensor, t)
                camera_list.append(camera[0])
            camera_tensor = torch.cat(camera_list, dim=0)
            camera_numpy = camera_tensor.clone().cpu().numpy()
            for idx, row in enumerate(camera_numpy):
                RT = row.reshape(3, 4)
                transform_matrix = np.vstack([RT, [0, 0, 0, 1]])
                transform_matrix_list = transform_matrix.tolist()
                frame_data = {
                    "transform_matrix": transform_matrix_list,
                    "monst3r_im_id": idx + 1
                }
                transforms_dict['frames'].append(frame_data)

            with open(transforms_path, 'w') as f:
                json.dump(transforms_dict, f, indent=4)
                
        def save_results(output_dir, name, camera_pose, text):
            # Save caption
            caption_dict = {'text': text, 'Concise Interaction': text}
            caption_path = os.path.join(output_dir, f"{name}_caption.json")
            with open(caption_path, 'w') as f:
                json.dump(caption_dict, f, indent=4)
            
            # Save predicted pose
            pred_pose_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
            pose_normalize(camera_pose, pred_pose_path)
        
        draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))
        save_results(output_dir, name, camera_pose, text)
        
        torch.cuda.synchronize()



output_dir = os.path.join(opt.workspace, opt.exp_name or 'inference_output')
if opt.resume is not None:
    output_dir = os.path.join(opt.workspace, opt.resume.split('/')[-1].split('.')[0])

if opt.cond_mode == 'text+rgbd+bbox':
    print("Start processing text+rgbd+bbox")
    if opt.image_path is not None:
        # Single image inference
        image_path = opt.image_path
        name = os.path.basename(image_path).replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        text_path = opt.text_path if opt.text_path else image_path.replace('_rgb.png', '_caption.json').replace('.png', '_caption.json').replace('.jpg', '_caption.json')
        depth_path = opt.depth_path if opt.depth_path else image_path.replace('_rgb.png', '_depth.npy').replace('.png', '_depth.npy').replace('.jpg', '_depth.npy')
        bbox_3d_path = None  # Will be estimated
        
        process_data(opt, output_dir, name, image_path, text=opt.text, text_path=text_path, depth_path=depth_path, bbox_3d_path=bbox_3d_path)
    elif os.path.isdir(opt.test_path):
        # Batch processing
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        if len(image_paths) == 0:
            image_paths = glob.glob(os.path.join(opt.test_path, "*.png"))
        print(f"Number of images: {len(image_paths)}")
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            if not os.path.exists(text_path):
                text_path = image_path.replace(".png", "_caption.json")
            depth_path = image_path.replace("_rgb.png", "_depth.npy")
            if not os.path.exists(depth_path):
                depth_path = image_path.replace(".png", "_depth.npy")
            name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8] if '/' in image_path else 'test/' + os.path.basename(image_path)[:-4]
            process_data(opt, output_dir, name, image_path, text_path=text_path, depth_path=depth_path)
elif opt.cond_mode == 'text':
    print("Start processing text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8]
            process_data(opt, output_dir, name, image_path, text_path=text_path)
elif opt.cond_mode == 'depth+image+text':
    print("Start processing depth+image+text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            depth_path = image_path.replace("_rgb.png", "_depth.npy")
            name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8]
            process_data(opt, output_dir, name, image_path, text_path=text_path, depth_path=depth_path)

