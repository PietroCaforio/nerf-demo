from run_nerf import *  # keeps your RayDataset etc.

import os, re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from torch.utils.data import DataLoader


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
# TensoRF with VM decomposition
# -----------------------------
class TensoRF_VM(nn.Module):
    """
    Inputs: x = (B,6) : xyz_world (3) and view_dir (3)
    Outputs: (B,4) = [R,G,B,sigma]
    """
    def __init__(
        self,
        aabb=((-2.5, -2.5, -2.5), (2.5, 2.5, 2.5)),
        grid_res=(128, 128, 128),
        sigma_rank=16,
        app_rank=24,
        app_dim=27,
        hidden_color=64,
        density_scale=10.0,   # scales sigma after activation
        act_shift=-1.0        # shifts sigma pre-activation
    ):
        super().__init__()
        self.register_buffer("aabb_min", torch.tensor(aabb[0], dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(aabb[1], dtype=torch.float32))
        self.Nx, self.Ny, self.Nz = grid_res
        self.density_scale = density_scale
        self.act_shift = act_shift

        # Density factors (VM)
        self.density_planes = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Ny, self.Nx)),  # XY
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Nz, self.Nx)),  # XZ
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Nz, self.Ny)),  # YZ
        ])
        self.density_lines = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Nx)),  # X
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Ny)),  # Y
            nn.Parameter(0.01 * torch.randn(sigma_rank, self.Nz)),  # Z
        ])
        self.sigma_act = nn.Softplus()

        # Appearance factors (VM, multi-channel)
        self.app_planes = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Ny, self.Nx)),  # XY
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Nz, self.Nx)),  # XZ
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Nz, self.Ny)),  # YZ
        ])
        self.app_lines = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Nx)),  # X
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Ny)),  # Y
            nn.Parameter(0.01 * torch.randn(app_dim, app_rank, self.Nz)),  # Z
        ])

        # Color MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(app_dim + 3, hidden_color),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_color, 3),
            nn.Sigmoid()
        )

    # -------- Utilities --------
    def _normalize_xyz(self, xyz):
        xyz01 = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8)
        valid = ((xyz01 >= 0) & (xyz01 <= 1)).all(dim=-1, keepdim=True).float()
        xyz_grid = xyz01 * 2.0 - 1.0  # [-1,1]
        return xyz_grid, valid

    @staticmethod
    def _grid_sample_points(plane_chw, uv):
        B = uv.shape[0]
        plane = plane_chw.unsqueeze(0)                    # (1,C,H,W)
        grid = uv.view(1, B, 1, 2)                        # (1,B,1,2)
        out = F.grid_sample(plane, grid, mode='bilinear', align_corners=True)  # (1,C,B,1)
        out = out.squeeze(0).squeeze(-1).permute(1, 0)    # (B,C)
        return out

    @staticmethod
    def _sample_line_1d(line_cn, u):
        C, N = line_cn.shape
        pos = (u + 1.0) * 0.5 * (N - 1)
        i0 = torch.clamp(pos.floor().long(), 0, N - 2)
        i1 = i0 + 1
        w1 = (pos - i0.float()).unsqueeze(1)
        w0 = 1.0 - w1
        line_nc = line_cn.t()
        v0 = line_nc[i0]
        v1 = line_nc[i1]
        return w0 * v0 + w1 * v1

    # -------- Field queries --------
    def _query_sigma_raw(self, xyz_grid):
        xg, yg, zg = xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2]
        p_xy = self._grid_sample_points(self.density_planes[0], torch.stack([xg, yg], dim=-1))
        p_xz = self._grid_sample_points(self.density_planes[1], torch.stack([xg, zg], dim=-1))
        p_yz = self._grid_sample_points(self.density_planes[2], torch.stack([yg, zg], dim=-1))
        l_x = self._sample_line_1d(self.density_lines[0], xg)
        l_y = self._sample_line_1d(self.density_lines[1], yg)
        l_z = self._sample_line_1d(self.density_lines[2], zg)
        sigma_raw = (p_xy * l_z).sum(dim=-1, keepdim=True) \
                  + (p_xz * l_y).sum(dim=-1, keepdim=True) \
                  + (p_yz * l_x).sum(dim=-1, keepdim=True)  # (B,1)
        return sigma_raw

    def _query_appfeat(self, xyz_grid):
        B = xyz_grid.shape[0]
        xg, yg, zg = xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2]
        def sample_app_plane(plane, uv):
            C, R, H, W = plane.shape
            plane_flat = plane.view(C * R, H, W)
            samp = self._grid_sample_points(plane_flat, uv)  # (B, C*R)
            return samp.view(B, C, R)
        def sample_app_line(line, u):
            C, R, N = line.shape
            line_flat = line.view(C * R, N)
            samp = self._sample_line_1d(line_flat, u)        # (B, C*R)
            return samp.view(B, C, R)

        ap_xy = sample_app_plane(self.app_planes[0], torch.stack([xg, yg], dim=-1))
        ap_xz = sample_app_plane(self.app_planes[1], torch.stack([xg, zg], dim=-1))
        ap_yz = sample_app_plane(self.app_planes[2], torch.stack([yg, zg], dim=-1))

        al_x = sample_app_line(self.app_lines[0], xg)
        al_y = sample_app_line(self.app_lines[1], yg)
        al_z = sample_app_line(self.app_lines[2], zg)

        app_feat = (ap_xy * al_z).sum(dim=-1) \
                 + (ap_xz * al_y).sum(dim=-1) \
                 + (ap_yz * al_x).sum(dim=-1)  # (B, app_dim)
        return app_feat

    # Standard forward (used in validation)
    def forward(self, x):
        xyz = x[:, :3]
        d = F.normalize(x[:, 3:], dim=-1)
        xyz_grid, valid = self._normalize_xyz(xyz)
        sigma_raw = self._query_sigma_raw(xyz_grid)
        sigma = self.sigma_act(sigma_raw + self.act_shift) * self.density_scale
        sigma = sigma * valid
        app_feat = self._query_appfeat(xyz_grid)
        rgb = self.color_mlp(torch.cat([app_feat, d], dim=-1))
        return torch.cat([rgb, sigma], dim=-1)


# -----------------------------
# Renderers (train + val), NaN-safe
# -----------------------------
def _composite_rgb_sigma(rgb, sigma, z, white_bkgd=True):
    """
    rgb:   (B,S,3)
    sigma: (B,S)  >=0
    z:     (B,S)  sample depths increasing
    """
    B, S = z.shape
    deltas = z[:, 1:] - z[:, :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)  # (B,S)

    # Stable alpha: clamp exponent argument into [-60, 0]
    exp_arg = (-sigma * deltas).clamp(min=-60.0, max=0.0)
    alpha = 1.0 - torch.exp(exp_arg)            # (B,S)
    alpha = alpha.clamp_(0.0, 1.0)              # avoid tiny negatives/ >1

    T = torch.cumprod(torch.cat(
        [torch.ones(B, 1, device=alpha.device, dtype=alpha.dtype),
         1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    w = alpha * T                                # (B,S)

    rgb_map = (w.unsqueeze(-1) * rgb).sum(dim=1)  # (B,3)
    if white_bkgd:
        acc = w.sum(dim=1, keepdim=True)
        rgb_map = rgb_map + (1.0 - acc) * 1.0
    return rgb_map


def render_rays_train(rays, model, num_samples=32, density_noise_std=1e-3,
                      white_bkgd=True, device='cuda'):
    ro, rd = rays[:, :3], F.normalize(rays[:, 3:], dim=-1)
    t0, t1, hit = ray_aabb_intersect(ro, rd, model.aabb_min, model.aabb_max)
    idx = torch.nonzero(hit, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return torch.ones_like(rays[:, :3]) if white_bkgd else torch.zeros_like(rays[:, :3])

    roh, rdh = ro[idx], rd[idx]
    t0h, t1h = t0[idx], t1[idx]

    B = idx.numel()
    u = torch.linspace(0.0, 1.0, steps=num_samples, device=device).unsqueeze(0).expand(B, -1)
    z = t0h.unsqueeze(1) * (1.0 - u) + t1h.unsqueeze(1) * u  # (B,S)

    pts = roh.unsqueeze(1) + rdh.unsqueeze(1) * z.unsqueeze(-1)  # (B,S,3)
    dirs = rdh.unsqueeze(1).expand(-1, num_samples, -1)          # (B,S,3)

    # Query field (do noise pre-activation in a numerically safe way)
    flat = torch.cat([pts, dirs], dim=-1).reshape(-1, 6)
    xyz = flat[:, :3]
    d   = F.normalize(flat[:, 3:], dim=-1)
    xyz_grid, valid = model._normalize_xyz(xyz)
    sigma_raw = model._query_sigma_raw(xyz_grid)  # (B*S,1)

    if density_noise_std > 0:
        sigma_raw = sigma_raw + torch.randn_like(sigma_raw) * density_noise_std

    sigma = model.sigma_act(sigma_raw + model.act_shift) * model.density_scale
    sigma = (sigma * valid).view(B, num_samples)              # (B,S)  >=0

    app_feat = model._query_appfeat(xyz_grid)
    rgb = model.color_mlp(torch.cat([app_feat, d], dim=-1)).view(B, num_samples, 3)

    rgb_map = _composite_rgb_sigma(rgb, sigma, z, white_bkgd=white_bkgd)
    out = torch.ones(rays.shape[0], 3, device=device) if white_bkgd else torch.zeros(rays.shape[0], 3, device=device)
    out[idx] = rgb_map
    return out


@torch.no_grad()
def render_rays_in_chunks(rays, model, num_samples=32, chunk_size=8192,
                          white_bkgd=True, device='cuda'):
    model.eval()
    N = rays.shape[0]
    out = torch.ones(N, 3, device=device) if white_bkgd else torch.zeros(N, 3, device=device)

    for i in range(0, N, chunk_size):
        rc = rays[i:i+chunk_size].to(device)
        ro, rd = rc[:, :3], F.normalize(rc[:, 3:], dim=-1)

        t0, t1, hit = ray_aabb_intersect(ro, rd, model.aabb_min, model.aabb_max)
        if not hit.any():
            continue
        idx = torch.nonzero(hit, as_tuple=False).squeeze(-1)
        roh, rdh = ro[idx], rd[idx]
        t0h, t1h = t0[idx], t1[idx]

        B = idx.numel()
        u = torch.linspace(0.0, 1.0, steps=num_samples, device=device).unsqueeze(0).expand(B, -1)
        z = t0h.unsqueeze(1) * (1.0 - u) + t1h.unsqueeze(1) * u

        pts = roh.unsqueeze(1) + rdh.unsqueeze(1) * z.unsqueeze(-1)
        dirs = rdh.unsqueeze(1).expand(-1, num_samples, -1)
        flat = torch.cat([pts, dirs], dim=-1).reshape(-1, 6)

        preds = model(flat)
        rgb = preds[:, :3].view(B, num_samples, 3)
        sigma = preds[:, 3].view(B, num_samples)           # already >=0 & masked

        rgb_map = _composite_rgb_sigma(rgb, sigma, z, white_bkgd=white_bkgd)
        out[i + idx] = rgb_map
    return out


# -----------------------------
# Checkpoint save (keep last 3)
# -----------------------------
def save_checkpoint(state, ckpt_dir, epoch, keep_last=3):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = Path(ckpt_dir) / f"model_epoch_{epoch:04d}.pt"
    torch.save(state, ckpt_path)

    rx = re.compile(r"model_epoch_(\d+)\.pt$")
    ckpts = []
    for p in Path(ckpt_dir).glob("model_epoch_*.pt"):
        m = rx.fullmatch(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda t: t[0], reverse=True)
    for _, p in ckpts[keep_last:]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return str(ckpt_path)


# -----------------------------
# Training loop (AMP + resume, NaN-safe)
# -----------------------------
def train_nerf(model, train_dataset, val_dataset,
               epochs=10000, batch_size=2048,
               num_samples_train=32, num_samples_val=32,
               lr_density=5e-3, lr_appearance=2e-3, lr_color=5e-4,
               val_every=50, output_dir='val_output_tensorf',
               device='cuda', resume_path=None, white_bkgd=True):

    device = torch.device(device)
    model.to(device)

    dens_params = list(model.density_planes.parameters()) + list(model.density_lines.parameters())
    app_params  = list(model.app_planes.parameters())    + list(model.app_lines.parameters())
    color_params= model.color_mlp.parameters()
    optimizer = torch.optim.Adam([
        {'params': dens_params,  'lr': lr_density},
        {'params': app_params,   'lr': lr_appearance},
        {'params': color_params, 'lr': lr_color},
    ], betas=(0.9, 0.99))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for g, lr in zip(optimizer.param_groups, [lr_density, lr_appearance, lr_color]):
                g['lr'] = lr
        start_epoch = int(ckpt.get('epoch', 0))
        print(f"Resumed from '{resume_path}' at epoch {start_epoch}.")

    # Train
    model.train()
    for epoch in range(start_epoch + 1, epochs + 1):
        total_loss = 0.0

        for rays, rgbs in train_loader:
            rays = rays.to(device, non_blocking=True)
            rgbs = rgbs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
                rgb_pred = render_rays_train(rays, model, num_samples=num_samples_train,
                                             density_noise_std=1e-3, white_bkgd=white_bkgd, device=device)
                loss = criterion(rgb_pred, rgbs)

            # NaN guard
            if not torch.isfinite(loss):
                print("Non-finite loss detected. Reducing LRs by 10x and continuing.")
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.1
                continue

            scaler.scale(loss).backward()
            # Optional gradient clipping (helps stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch:04d}] Train Loss: {avg_loss:.6e}")

        # Validation + checkpoint
        if epoch % val_every == 0:
            model.eval()
            with torch.no_grad():
                rays, _ = val_dataset[0]  # one full image
                rays = rays.to(device, non_blocking=True)
                rgb_pred = render_rays_in_chunks(rays, model, num_samples=num_samples_val,
                                                 chunk_size=4096, white_bkgd=white_bkgd, device=device)
                H, W = val_dataset.img_wh[1], val_dataset.img_wh[0]
                img_pred = rgb_pred.reshape(H, W, 3).clamp(0, 1).cpu().numpy()
                img_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
                imageio.imwrite(img_path, (img_pred * 255).astype('uint8'))
                print(f"Saved validation image to {img_path}")

                ckpt_path = save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_dir, epoch, keep_last=3)
                print(f"Checkpoint saved to {ckpt_path} (kept last 3).")
            model.train()


# -----------------------------
# Example main
# -----------------------------
if __name__ == "__main__":
    model = TensoRF_VM(
        aabb=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
        grid_res=(256, 256, 256),
        sigma_rank=16,
        app_rank=24,
        app_dim=27,
        hidden_color=64,
        density_scale=10.0,
        act_shift=-1.0
    )

    train_dataset = RayDataset('./data/nerf_synthetic/lego', mode='train')
    val_dataset   = RayDataset('./data/nerf_synthetic/lego', mode='val')

    train_nerf(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10000,
        batch_size=1024,
        num_samples_train=128,
        num_samples_val=128,
        lr_density=5e-3,        # lowered for stability
        lr_appearance=2e-3,
        lr_color=5e-4,
        val_every=50,
        output_dir='./val_output_tensorf',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume_path=None,
        white_bkgd=True
    )
