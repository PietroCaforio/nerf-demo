import os, json, math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

# -------------------------------------------------------------------
from run_tensorf import TensoRF_VM


# -----------------------------
# Ray-box intersection (AABB)
# -----------------------------
def ray_aabb_intersect(ray_o, ray_d, aabb_min, aabb_max, eps=1e-9):
    inv_d = 1.0 / (ray_d + eps)
    t0s = (aabb_min - ray_o) * inv_d
    t1s = (aabb_max - ray_o) * inv_d
    tmin = torch.minimum(t0s, t1s).amax(dim=-1)
    tmax = torch.maximum(t0s, t1s).amin(dim=-1)
    tmin = torch.clamp(tmin, min=0.0)
    hit = tmax > tmin
    return tmin, tmax, hit


# -----------------------------
# NaN-safe compositing
# -----------------------------
def _composite_rgb_sigma(rgb, sigma, z, white_bkgd=True):
    """
    rgb:   (B,S,3)
    sigma: (B,S)  >=0
    z:     (B,S)
    """
    B, S = z.shape
    deltas = z[:, 1:] - z[:, :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)  # (B,S)

    exp_arg = (-sigma * deltas).clamp(min=-60.0, max=0.0)
    alpha = 1.0 - torch.exp(exp_arg)
    alpha = alpha.clamp_(0.0, 1.0)

    T = torch.cumprod(torch.cat(
        [torch.ones(B, 1, device=alpha.device, dtype=alpha.dtype),
         1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    w = alpha * T

    rgb_map = (w.unsqueeze(-1) * rgb).sum(dim=1)
    if white_bkgd:
        acc = w.sum(dim=1, keepdim=True)
        rgb_map = rgb_map + (1.0 - acc) * 1.0
    return rgb_map


# -----------------------------
# Memory-frugal validation renderer
# Streams over rays and samples, uses AMP.
# -----------------------------
@torch.no_grad()
def render_rays_in_chunks(
    rays, model,
    num_samples=64,
    chunk_rays=1024,          # reduce if you still OOM
    samples_per_iter=16,      # stream along sample axis
    white_bkgd=True,
    device='cuda'
):
    model.eval()
    N = rays.shape[0]
    out = torch.ones(N, 3, device=device) if white_bkgd else torch.zeros(N, 3, device=device)

    for i in range(0, N, chunk_rays):
        rc = rays[i:i+chunk_rays].to(device, non_blocking=True)
        ro, rd = rc[:, :3], F.normalize(rc[:, 3:], dim=-1)

        t0, t1, hit = ray_aabb_intersect(ro, rd, model.aabb_min, model.aabb_max)
        if not hit.any():
            continue
        idx = torch.nonzero(hit, as_tuple=False).squeeze(-1)
        roh, rdh = ro[idx], rd[idx]
        t0h, t1h = t0[idx], t1[idx]
        B = idx.numel()

        # accumulators
        rgb_map = torch.zeros(B, 3, device=device)
        T = torch.ones(B, 1, device=device)

        # uniform bins in [0,1] over the per-ray interval
        edges = torch.linspace(0.0, 1.0, steps=num_samples+1, device=device)
        z_prev = t0h

        with torch.cuda.amp.autocast(enabled=(device=='cuda'), dtype=torch.float16):
            s = 0
            while s < num_samples:
                nstep = min(samples_per_iter, num_samples - s)
                low  = edges[s:s+nstep]
                high = edges[s+1:s+1+nstep]

                u = torch.rand(B, nstep, device=device)
                t = low.unsqueeze(0)*(1-u) + high.unsqueeze(0)*u        # (B, nstep)
                z = t0h.unsqueeze(1)*(1.0 - t) + t1h.unsqueeze(1)*t     # (B, nstep)

                # deltas (include cross-block delta)
                deltas = torch.empty(B, nstep, device=device)
                deltas[:, 0] = z[:, 0] - z_prev
                if nstep > 1:
                    deltas[:, 1:] = z[:, 1:] - z[:, :-1]
                z_prev = z[:, -1]

                pts  = roh.unsqueeze(1) + rdh.unsqueeze(1) * z.unsqueeze(-1)  # (B, nstep, 3)
                dirs = rdh.unsqueeze(1).expand(-1, nstep, -1)
                flat = torch.cat([pts, dirs], dim=-1).reshape(-1, 6)

                pred  = model(flat)
                rgb   = pred[:, :3].view(B, nstep, 3)
                sigma = pred[:,  3].view(B, nstep)

                # composite step
                exp_arg = (-sigma * deltas).clamp(min=-60.0, max=0.0)
                alpha = 1.0 - torch.exp(exp_arg)
                alpha = alpha.clamp_(0.0, 1.0)

                w = alpha * T                          # (B, nstep)
                rgb_map = rgb_map + (w.unsqueeze(-1) * rgb).sum(dim=1)
                T = T * (1.0 - alpha).prod(dim=1, keepdim=True)

                if (T < 1e-3).all():
                    break

                s += nstep

        if white_bkgd:
            rgb_map = rgb_map + T.expand_as(rgb_map)    # white bg

        out[i + idx] = rgb_map
    return out


# -----------------------------
# Camera + ray generation
# -----------------------------
def look_at(eye, target=(0., 0., 0.), up=(0., 1., 0.)):
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)

    forward = (target - eye)
    forward = forward / (forward.norm() + 1e-9)
    right = torch.cross(forward, up)
    right = right / (right.norm() + 1e-9)
    true_up = torch.cross(right, forward)

    # Camera convention in your dataset: camera looks along -Z in camera space.
    R = torch.stack([right, true_up, -forward], dim=1)  # columns
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


def get_rays(H, W, focal, c2w):
    """
    Build rays following your dataset convention:
      dir_cam = [(i - W/2)/f, -(j - H/2)/f, -1]
      dir_world = dir_cam @ R^T
      origin = c2w[:3,3]
    Returns (H*W, 6)
    """
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - W*0.5)/focal,
                        -(j - H*0.5)/focal,
                        -torch.ones_like(i)], dim=-1)  # (H,W,3)
    R = c2w[:3, :3]
    rays_d = dirs.reshape(-1, 3) @ R.t()               # (H*W, 3)
    rays_o = c2w[:3, 3].expand_as(rays_d)              # (H*W, 3)
    return torch.cat([rays_o, rays_d], dim=-1)         # (H*W, 6)


def circle_poses(num_views=120, radius=4.0, elev_deg=0.0, center=(0., 0., 0.)):
    poses = []
    for k in range(num_views):
        theta = 2.0 * math.pi * (k / num_views)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        y = radius * math.sin(math.radians(elev_deg))  # small lift if desired
        poses.append(look_at((x, y, z), target=center))
    return poses


# -----------------------------
# Render one pose to image
# -----------------------------
def render_image(model, H, W, focal, c2w, device='cuda',
                 num_samples=64, chunk_rays=1024, samples_per_iter=16, white_bkgd=True):
    c2w = c2w.to(device)
    rays = get_rays(H, W, focal, c2w)                   # (H*W, 6)
    with torch.no_grad():
        rgb = render_rays_in_chunks(
            rays, model,
            num_samples=num_samples,
            chunk_rays=chunk_rays,
            samples_per_iter=samples_per_iter,
            white_bkgd=white_bkgd,
            device=device
        )
    img = rgb.view(H, W, 3).clamp(0, 1).cpu().numpy()
    return img


# -----------------------------
# Main: load, render 360°, save GIF
# -----------------------------
if __name__ == "__main__":
    # ---- Hard-coded arguments (edit these) ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_root = './data/nerf_synthetic/lego'                   # path to dataset folder
    transforms_json = os.path.join(dataset_root, 'transforms_train.json')
    ckpt_path = './val_output_tensorf/checkpoints/model_epoch_5000.pt'  # path to your checkpoint
    out_dir = './render_360'
    out_gif = os.path.join(out_dir, 'lego_360.gif')

    # Image resolution & renderer settings
    W, H = 800, 800              # output image width/height
    num_views = 120              # frames around the circle
    fps = 24
    radius = 3.0
    elev_deg = 30.0
    num_samples = 64             # samples per ray
    chunk_rays = 1024            # reduce if you OOM
    samples_per_iter = 16        # 8–16 are safe

    # Model hyperparams 
    model = TensoRF_VM(
        aabb=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
        grid_res=(128, 128, 128),
        sigma_rank=24,
        app_rank=24,
        app_dim=27,
        hidden_color=64,
        density_scale=10.0,
        act_shift=-1.0
    ).to(device)

    # ---- Load checkpoint ----
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- Intrinsics (focal) from dataset JSON ----
    with open(transforms_json, 'r') as f:
        meta = json.load(f)
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

    # ---- Render 360° path ----
    os.makedirs(out_dir, exist_ok=True)
    poses = circle_poses(num_views=num_views, radius=radius, elev_deg=elev_deg, center=(0., 0., 0.))
    frames = []

    for idx, c2w in enumerate(tqdm(poses, desc="Rendering 360° views")):
        img = render_image(
            model, H, W, focal, c2w, device=device,
            num_samples=num_samples, chunk_rays=chunk_rays,
            samples_per_iter=samples_per_iter, white_bkgd=True
        )
        # convert to uint8 sRGB for GIF
        frame = (img * 255).astype(np.uint8)
        frames.append(frame)
        

    # ---- Save GIF ----
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"Saved GIF → {out_gif}")
