import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(Wi, softplus_ci):  # L-inf norm
    absrowsum = torch.sum(torch.abs(Wi), dim=1, keepdim=True)  # Shape: (out_dim, 1)
    scale = torch.minimum(
        torch.tensor(1.0, device=Wi.device),
        F.softplus(softplus_ci).unsqueeze(1) / absrowsum,
    )
    return Wi * scale

class LFQQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_uniform_(self.codebook)

    def forward(self, z_e):
        batch_size, seq_len, latent_dim = z_e.shape
        z_e_expanded = z_e.unsqueeze(2)  # Shape: [B, S, 1, D]
        codebook_expanded = self.codebook.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        distances = torch.norm(z_e_expanded - codebook_expanded, dim=-1)
        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook[indices]  # [B, S, D]
        return z_q, indices

class LFQ_VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_codes=1024, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.quantizer = LFQQuantizer(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        x_flat = x.reshape(-1, feature_dim)
        h = self.encoder(x_flat).view(batch_size, seq_len, -1)
        z_e = self.to_latent(h)
        z_q, indices = self.quantizer(z_e)
        z_latent = z_q.clone().detach()
        recon = self.decoder(z_q).view(batch_size, seq_len, -1)
        x_recon = self.to_output(recon)
        
        # Loss computation
        recon_loss = F.mse_loss(x_recon, x)
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        loss = recon_loss + 0.25 * commitment_loss + 0.25 * codebook_loss
        return z_latent, loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    seq_len = 256
    feature_dim = 256
    latent_dim = 256
    num_codes = 128
    model = LipFQ_VAE(feature_dim, latent_dim, num_codes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(batch_size, seq_len, feature_dim).to(device)

    for epoch in range(1000):
        optimizer.zero_grad()
        z_latent, loss = model(data)
        print(f"Epoch {epoch}: Latent Shape {data.shape}, Loss {loss.item():.4f}")
        loss.backward()
        optimizer.step()
