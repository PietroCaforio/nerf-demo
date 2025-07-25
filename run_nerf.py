import torch
from torch import nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
    

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
    model = NaiveNERF()
    inputs = torch.tensor([
        [0.0, 1.0, 2.0, 0.5, 1.2],
        [0.2, 0.9, 2.1, 0.4, 1.0],
        [0.1, 1.1, 1.8, 0.6, 1.4],
    ], dtype=torch.float32)
    output = model(inputs)
    print("Output shape:", output.shape)  # (3, 4)
    print("Output (R, G, B, σ):\n", output)
