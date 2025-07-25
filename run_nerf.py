import torch
from torch import nn
import json

import os
import json
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

class RayDataset(Dataset):
    def __init__(self, dataset_root, split='train', img_wh=(400, 400), samples_per_epoch=4096, mode='train'):
        super().__init__()
        self.root = dataset_root
        self.split = split
        self.img_wh = img_wh
        self.samples_per_epoch = samples_per_epoch
        
        self.mode = mode
        assert mode in ['train', 'val'], "mode must be 'train' or 'val'"
        
        self._load_meta()
        self._load_images_and_poses()
        self._precompute_rays()

    def _load_meta(self):
        json_path = os.path.join(self.root, f"transforms_{self.split}.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.frames = meta['frames']
        self.camera_angle_x = meta['camera_angle_x']

        self.focal = 0.5 * self.img_wh[0] / np.tan(0.5 * self.camera_angle_x)
    
    def _load_image_pil(self, path, img_wh):
        img = Image.open(path).convert('RGB')
        img = img.resize(img_wh, Image.BICUBIC)
        img = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3), [0,1]
        img = img.permute(2, 0, 1)  # to (3, H, W)
        return img
    
    def _load_images_and_poses(self):

        self.images = []
        self.poses = []

        for frame in self.frames:
            img_path = os.path.join(self.root, frame['file_path'] + '.png')
            img = self._load_image_pil(img_path, self.img_wh)
            self.images.append(img)

            pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)  # (4, 4)
            self.poses.append(pose)

        self.images = torch.stack(self.images)  # (N, 3, H, W)
        self.poses = torch.stack(self.poses)    # (N, 4, 4)

    def _precompute_rays(self):
        H, W = self.img_wh[1], self.img_wh[0]
        i_coords, j_coords = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing='xy'
        )  # (W, H)

        i_coords = i_coords.t()  # (H, W)
        j_coords = j_coords.t()

        directions = torch.stack([
            (i_coords - W * 0.5) / self.focal,
            -(j_coords - H * 0.5) / self.focal,
            -torch.ones_like(i_coords)
        ], dim=-1)  # (H, W, 3)

        rays_o = []
        rays_d = []
        rgbs = []

        for img, pose in zip(self.images, self.poses):
            rays_d_cam = directions @ pose[:3, :3].T  # (H, W, 3)
            rays_o_cam = pose[:3, 3].expand(H, W, 3)  # (H, W, 3)

            rays_o.append(rays_o_cam)
            rays_d.append(rays_d_cam)
            rgbs.append(img.permute(1, 2, 0))  # (H, W, 3)

        self.all_rays_o = torch.cat([r.reshape(-1, 3) for r in rays_o], dim=0)
        self.all_rays_d = torch.cat([r.reshape(-1, 3) for r in rays_d], dim=0)
        self.all_rgbs = torch.cat([rgb.reshape(-1, 3) for rgb in rgbs], dim=0)

        self.N = self.all_rays_o.shape[0]  # total number of rays

    def __len__(self):
        if self.mode == 'train':
            return self.samples_per_epoch
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            rand_idx = torch.randint(0, self.N, (1,)).item()
            ray_o = self.all_rays_o[rand_idx]
            ray_d = self.all_rays_d[rand_idx]
            rgb = self.all_rgbs[rand_idx]
            ray = torch.cat([ray_o, ray_d], dim=0)  # (6,)
            return ray, rgb

        else:  # validation mode
            rays_o = self.all_rays_o[idx * self.img_wh[0] * self.img_wh[1]:(idx + 1) * self.img_wh[0] * self.img_wh[1]]
            rays_d = self.all_rays_d[idx * self.img_wh[0] * self.img_wh[1]:(idx + 1) * self.img_wh[0] * self.img_wh[1]]
            rgbs = self.all_rgbs[idx * self.img_wh[0] * self.img_wh[1]:(idx + 1) * self.img_wh[0] * self.img_wh[1]]
            rays = torch.cat([rays_o, rays_d], dim=1)  # (H*W, 6)
            return rays, rgbs  # both (H*W, *)

        
        

# NERF without positional encoding and with a single mlp 
# processing both position and view direction
class NaiveNERF(nn.Module):  
    def __init__(self, inter_dim=(64, 128, 64), input_dim=5, activation=nn.ReLU):
        super().__init__()
        layers = []

        dims = [input_dim] + list(inter_dim)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())

        self.shared_mlp = nn.Sequential(*layers)

        last_dim = inter_dim[-1]
        self.rgb_head = nn.Sequential(
            nn.Linear(last_dim, 3),
            nn.Sigmoid()  # RGB in [0,1]
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(last_dim, 1),
            nn.ReLU()     # σ >= 0
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 5) where each row is [x, y, z, phi, theta]
        Returns:
            Tensor of shape (B, 4) where each row is [R, G, B, σ]
        """
        features = self.shared_mlp(x)
        rgb = self.rgb_head(features)
        sigma = self.sigma_head(features)
        return torch.cat([rgb, sigma], dim=-1)

if __name__ == "__main__":
    # Training
    train_dataset = RayDataset('./data/nerf_synthetic/lego', mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Validation
    val_dataset = RayDataset('./data/nerf_synthetic/lego', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # one image at a time

    # Iterate over one image in val
    for rays, rgbs in val_loader:
        rays = rays.squeeze(0)  # (H*W, 6)
        rgbs = rgbs.squeeze(0)  # (H*W, 3)
        print("Validation rays shape:", rays.shape)
        print("Validation RGBs shape:", rgbs.shape)
        break
    
    model = NaiveNERF()
    inputs = torch.tensor([
        [0.0, 1.0, 2.0, 0.5, 1.2],
        [0.2, 0.9, 2.1, 0.4, 1.0],
        [0.1, 1.1, 1.8, 0.6, 1.4],
    ], dtype=torch.float32)
    output = model(inputs)
    print("Output shape:", output.shape)  # (3, 4)
    print("Output (R, G, B, σ):\n", output)
