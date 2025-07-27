from run_nerf import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class TensoRF_VM(nn.Module):
    """
    TensoRF with VM decomposition.
    Inputs:  x = (B, 6) where x[..., :3] = xyz (world), x[..., 3:] = view dir (dx,dy,dz)
    Outputs: (B, 4) = [R, G, B, sigma]
    """
    def __init__(
        self,
        aabb=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),   # scene bounds
        grid_res=(128, 128, 128),                    # (Nx, Ny, Nz)
        sigma_rank=16,                                # rank for density
        app_rank=24,                                  # rank for appearance
        app_dim=27,                                   # appearance feature channels fed to color MLP
        hidden_color=64                               # color MLP width
    ):
        super().__init__()
        self.register_buffer("aabb_min", torch.tensor(aabb[0], dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(aabb[1], dtype=torch.float32))
        self.Nx, self.Ny, self.Nz = grid_res

        # -------- Density factors (VM) --------
        # Planes: (rank, H, W) with pairwise axes
        self.density_planes = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Ny, self.Nx)),  # XY (H=Ny, W=Nx)
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Nz, self.Nx)),  # XZ (H=Nz, W=Nx)
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Nz, self.Ny)),  # YZ (H=Nz, W=Ny)
        ])
        # Lines: (rank, N)
        self.density_lines = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Nx)),  # X
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Ny)),  # Y
            nn.Parameter(0.1 * torch.randn(sigma_rank, self.Nz)),  # Z
        ])
        self.sigma_act = nn.Softplus()

        # -------- Appearance factors (VM, multi-channel) --------
        # Planes: (C_app, rank, H, W)
        self.app_planes = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Ny, self.Nx)),  # XY
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Nz, self.Nx)),  # XZ
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Nz, self.Ny)),  # YZ
        ])
        # Lines: (C_app, rank, N)
        self.app_lines = nn.ParameterList([
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Nx)),  # X
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Ny)),  # Y
            nn.Parameter(0.1 * torch.randn(app_dim, app_rank, self.Nz)),  # Z
        ])

        # Small color MLP: [app_feat, view_dir] -> RGB
        self.color_mlp = nn.Sequential(
            nn.Linear(app_dim + 3, hidden_color),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_color, 3),
            nn.Sigmoid()
        )

    # -------- Utilities --------
    def _normalize_xyz(self, xyz):
        # Map world xyz to [0,1]^3 using AABB, then to [-1,1] for grid_sample.
        xyz01 = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8)
        valid = (xyz01 >= 0) & (xyz01 <= 1)
        valid = valid.all(dim=-1, keepdim=True).float()
        xyz_grid = xyz01 * 2.0 - 1.0  # [-1,1]
        return xyz_grid, valid

    @staticmethod
    def _grid_sample_points(plane_chw, uv):
        """
        plane_chw: (C, H, W)
        uv: (B, 2) in [-1,1], where uv[...,0] -> x (W), uv[...,1] -> y (H)
        returns: (B, C)
        """
        B = uv.shape[0]
        plane = plane_chw.unsqueeze(0)                # (1, C, H, W)
        grid = uv.view(1, B, 1, 2)                    # (1, B, 1, 2) -> H_out=B, W_out=1
        out = F.grid_sample(plane, grid, mode='bilinear', align_corners=True)  # (1, C, B, 1)
        out = out.squeeze(0).squeeze(-1).permute(1, 0)                         # (B, C)
        return out

    @staticmethod
    def _sample_line_1d(line_cn, u):
        """
        line_cn: (C, N) values along one axis
        u: (B,) in [-1,1] -> linear sample in [0, N-1]
        returns: (B, C)
        """
        C, N = line_cn.shape
        pos = (u + 1.0) * 0.5 * (N - 1)              # [0, N-1]
        i0 = torch.clamp(pos.floor().long(), 0, N - 2)
        i1 = i0 + 1
        w1 = (pos - i0.float()).unsqueeze(1)         # (B, 1)
        w0 = 1.0 - w1
        line_nc = line_cn.t()                         # (N, C)
        v0 = line_nc[i0]                              # (B, C)
        v1 = line_nc[i1]                              # (B, C)
        return w0 * v0 + w1 * v1                      # (B, C)

    # -------- Field queries --------
    def _query_sigma(self, xyz_grid):
        """
        xyz_grid: (B, 3) in [-1,1]; returns (B,1) raw sigma before activation
        """
        xg, yg, zg = xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2]

        # Planes
        p_xy = self._grid_sample_points(self.density_planes[0], torch.stack([xg, yg], dim=-1))  # (B, R)
        p_xz = self._grid_sample_points(self.density_planes[1], torch.stack([xg, zg], dim=-1))
        p_yz = self._grid_sample_points(self.density_planes[2], torch.stack([yg, zg], dim=-1))  # note: (y,z) plane has W=Ny, H=Nz

        # Lines
        l_x = self._sample_line_1d(self.density_lines[0], xg)  # (B, R)
        l_y = self._sample_line_1d(self.density_lines[1], yg)
        l_z = self._sample_line_1d(self.density_lines[2], zg)

        # VM combine (sum over rank)
        sigma_raw = (p_xy * l_z).sum(dim=-1, keepdim=True) \
                  + (p_xz * l_y).sum(dim=-1, keepdim=True) \
                  + (p_yz * l_x).sum(dim=-1, keepdim=True)  # (B,1)
        return sigma_raw

    def _query_appfeat(self, xyz_grid):
        """
        xyz_grid: (B,3) in [-1,1]; returns (B, app_dim)
        """
        B = xyz_grid.shape[0]
        xg, yg, zg = xyz_grid[:, 0], xyz_grid[:, 1], xyz_grid[:, 2]

        # Sample planes: flatten (C_app, R, H, W) -> (C_app*R, H, W) for sampling
        def sample_app_plane(plane, uv):
            C, R, H, W = plane.shape
            plane_flat = plane.view(C * R, H, W)                          # (C*R, H, W)
            samp = self._grid_sample_points(plane_flat, uv)               # (B, C*R)
            return samp.view(B, C, R)                                     # (B, C_app, R)

        ap_xy = sample_app_plane(self.app_planes[0], torch.stack([xg, yg], dim=-1))
        ap_xz = sample_app_plane(self.app_planes[1], torch.stack([xg, zg], dim=-1))
        ap_yz = sample_app_plane(self.app_planes[2], torch.stack([yg, zg], dim=-1))

        # Sample lines: flatten (C_app, R, N) -> (C_app*R, N)
        def sample_app_line(line, u):
            C, R, N = line.shape
            line_flat = line.view(C * R, N)                               # (C*R, N)
            samp = self._sample_line_1d(line_flat, u)                     # (B, C*R)
            return samp.view(B, C, R)                                     # (B, C_app, R)

        al_x = sample_app_line(self.app_lines[0], xg)
        al_y = sample_app_line(self.app_lines[1], yg)
        al_z = sample_app_line(self.app_lines[2], zg)

        # VM combine per appearance channel: sum over rank
        app_feat = (ap_xy * al_z).sum(dim=-1) \
                 + (ap_xz * al_y).sum(dim=-1) \
                 + (ap_yz * al_x).sum(dim=-1)                              # (B, C_app)
        return app_feat

    # -------- Forward --------
    def forward(self, x):
        """
        x: (B, 6) with [xyz_world, view_dir]
        returns: (B, 4) = [R,G,B, sigma]
        """
        xyz = x[:, :3]
        d = F.normalize(x[:, 3:], dim=-1)  # normalize view dir
        xyz_grid, valid = self._normalize_xyz(xyz)  # [-1,1], valid mask (B,1)

        sigma_raw = self._query_sigma(xyz_grid)                    # (B,1)
        sigma = self.sigma_act(sigma_raw) * valid                  # zero out-of-aabb

        app_feat = self._query_appfeat(xyz_grid)                   # (B, app_dim)
        rgb = self.color_mlp(torch.cat([app_feat, d], dim=-1))     # (B,3)

        return torch.cat([rgb, sigma], dim=-1)

if __name__ == "__main__":
    model = TensoRF_VM(
        aabb=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
        grid_res=(64, 64, 64),
        sigma_rank=8,
        app_rank=16,
        app_dim=27,
        hidden_color=32
    )
    
    # Prepare datasets
    train_dataset = RayDataset('./data/nerf_synthetic/lego', mode='train')
    val_dataset = RayDataset('./data/nerf_synthetic/lego', mode='val')

    # Train it!
    train_nerf(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10000,
        batch_size=1024,
        val_every=10,
        output_dir='./val_output_tensorf',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume_path = "./val_output_tensorf/checkpoints/model_epoch_0400.pt"
    )