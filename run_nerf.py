import torch
from torch import nn
import json

import os
import json
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt

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
    def __init__(self, inter_dim=(64, 128, 64), input_dim=6, activation=nn.ReLU):
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
            x: Tensor of shape (B, 6) where each row is [x, y, z, dx, dy, dz]
        Returns:
            Tensor of shape (B, 4) where each row is [R, G, B, σ]
        """
        features = self.shared_mlp(x)
        rgb = self.rgb_head(features)
        sigma = self.sigma_head(features) + 1e-2  # prevents all-zero sigma
        return torch.cat([rgb, sigma], dim=-1)

def render_rays(rays, model, 
                num_samples=64,
                near=2.0,
                far=6.0,
                device='cpu'):
    """
    Args:
        rays: (N_rays, 6) tensor of [ray_o (3), ray_d (3)]
        model: NeRF MLP model
        num_samples: number of sampled points along each ray
        near, far: bounds for sampling
        device: 'cuda' or 'cpu'
    Returns:
        rgb_map: (N_rays, 3) final rendered colors
    """
    N_rays = rays.shape[0]
    ray_o, ray_d = rays[:, :3], rays[:, 3:]  # (N_rays, 3)
    
    #Stratified sampling
    bins = torch.linspace(near, far, steps=num_samples + 1).to(device)  # (N+1,)
    lower = bins[:-1]  # (N,)
    upper = bins[1:]   # (N,)
    t_vals = lower + (upper - lower) * torch.rand((rays.shape[0], num_samples), device=device)  # (N_rays, N)
    pts = ray_o.unsqueeze(1).repeat(1,64,1) + t_vals.unsqueeze(-1) * ray_d.unsqueeze(1).repeat(1,64,1)
    
    # Flatten all points to feed into the MLP
    dirs = ray_d.unsqueeze(1).expand(-1, num_samples, -1)  # (N_rays, num_samples, 3)
    inputs = torch.cat([pts, dirs], dim=-1).reshape(-1, 6)  # (N_rays * num_samples, 6)
    
    #Predict RGB and sigma
    preds = model(inputs)    
    rgb = preds[:, :3].reshape(N_rays, num_samples, 3)       # (N_rays, num_samples, 3)
    sigma = preds[:, 3].reshape(N_rays, num_samples) 
    
    #compute deltas
    deltas = t_vals[:,1:] - t_vals[:,:-1]
    delta_last = torch.full((N_rays, 1), 1e10, device=device)
    deltas = torch.cat([deltas, delta_last], dim=-1)  # (N_rays, num_samples)
    
    #compute alpha values
    alphas = 1.0 - torch.exp(-sigma * deltas)
    accum_prod = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=device), 1. - alphas + 1e-10], -1), -1)[:, :-1]
    weights = alphas * accum_prod  # (N_rays, num_samples)
    
    
    # Weighted RGB output
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (N_rays, 3)

    return rgb_map

def render_rays_in_chunks(rays, model, 
                          num_samples=64, 
                          near=2.0, far=6.0, 
                          chunk_size=1024, 
                          device='cpu'):
    """
    Args:
        rays: (N_rays, 6)
        model: NeRF MLP
    Returns:
        rgb_map: (N_rays, 3)
    """
    model.eval()
    N_rays = rays.shape[0]
    rgb_out = []

    with torch.no_grad():
        for i in range(0, N_rays, chunk_size):
            rays_chunk = rays[i:i+chunk_size]
            rgb_chunk = render_rays(
                rays_chunk, model, num_samples=num_samples, near=near, far=far, device=device
            )
            rgb_out.append(rgb_chunk)

    return torch.cat(rgb_out, dim=0)  # (N_rays, 3)

def train_nerf(model, train_dataset, val_dataset, 
               epochs=1000, batch_size=1024, lr=5e-4, 
               val_every=10, output_dir='val_output', device='cuda'):
    
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for rays, rgbs in train_loader:
            rays, rgbs = rays.to(device), rgbs.to(device)
            preds = model(rays)
            rgb_preds = preds[:, :3]
            loss = criterion(rgb_preds, rgbs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch:04d}] Train Loss: {avg_loss:.6f}")

        # --- Validation ---
        if epoch % val_every == 0:
            model.eval()
            with torch.no_grad():
                rays, rgbs_gt = val_dataset[0]  # (H*W, 6), (H*W, 3)
                rays = rays.to(device)
                rgb_pred = render_rays_in_chunks(rays, model, device=device, chunk_size=1024)


                H, W = val_dataset.img_wh[1], val_dataset.img_wh[0]
                img_pred = rgb_pred.reshape(H, W, 3).cpu().numpy()

                save_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
                imageio.imwrite(save_path, (img_pred * 255).astype('uint8'))
                print(f"Saved validation image to {save_path}")
            model.train()

if __name__ == "__main__":
    
    # Prepare datasets
    train_dataset = RayDataset('./data/nerf_synthetic/lego', mode='train')
    val_dataset = RayDataset('./data/nerf_synthetic/lego', mode='val')

    # Create model
    model = NaiveNERF(inter_dim=(64, 128, 64))  # Or custom architecture

    # Train it!
    train_nerf(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=200,
        batch_size=1024,
        val_every=10,
        output_dir='./val_output',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
