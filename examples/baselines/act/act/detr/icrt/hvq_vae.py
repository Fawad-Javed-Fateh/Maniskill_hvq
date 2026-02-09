import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# import faiss

# ----------------------------
# LFQ Quantizer WITH EMA
# ----------------------------
def normalization(Wi, softplus_ci):  # L-inf norm
    absrowsum = torch.sum(torch.abs(Wi), dim=1, keepdim=True)  # Shape: (out_dim, 1)
    scale = torch.minimum(
        torch.tensor(1.0, device=Wi.device),
        F.softplus(softplus_ci).unsqueeze(1) / absrowsum,
    )
    return Wi * scale  # Broadcasting should now work


class LFQQuantizerEMA_KMeans(nn.Module):
    """
    LFQ quantizer with:
      • K-Means initialization (first batch)
      • EMA codebook updates
      • Codebook usage tracking
      • Dead-code replacement
      • L-inf Lipschitz ML layer compatibility
    """

    def __init__(
        self,
        num_codes,
        code_dim,
        decay=0.99,
        epsilon=1e-5,
        dead_threshold=5,  # minimum usage before marking as dead
        replace_strategy="nearest",  # or "random"
    ):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold
        self.replace_strategy = replace_strategy
        self.training = True

        # K-Means will overwrite codebook on first forward pass
        self.initialized = False

        # Main codebook + EMA buffers
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_normal_(self.codebook)

        self.ema_cluster_size = nn.Parameter(
            torch.zeros(num_codes), requires_grad=False
        )
        self.ema_codebook = nn.Parameter(
            torch.randn(num_codes, code_dim), requires_grad=False
        )

        # Tracking utilization
        self.register_buffer("usage_counts", torch.zeros(num_codes))
        self.register_buffer("usage_ma", torch.zeros(num_codes))  # moving average
        self.register_buffer("entropy_ma", torch.tensor(0.0))

    # -------------------------------------------------------------
    # K-Means initialization from first batch
    # -------------------------------------------------------------
    def kmeans_init(self, z_e):
        """
        z_e: [B, S, D]
        Initializes codebook using token-level KMeans
        """
        B, S, D = z_e.shape

        # -------------------------------------------------
        # Flatten tokens: [B*S, D]
        # -------------------------------------------------
        z_e_flat = z_e.reshape(B * S, D)

        # Cap number of samples (token-level)
        n_samples = min(20000, B * S)

        # Random token sampling
        sample_idx = torch.randperm(B * S, device=z_e.device)[:n_samples]
        sample = z_e_flat[sample_idx].detach().cpu().numpy()

        # -------------------------------------------------
        # KMeans
        # -------------------------------------------------
        kmeans = KMeans(
            n_clusters=self.num_codes,
            n_init="auto",
            max_iter=50,
        )
        centers = kmeans.fit(sample).cluster_centers_

        centers = torch.tensor(
            centers,
            dtype=z_e.dtype,
            device=z_e.device,
        )

        # -------------------------------------------------
        # Initialize codebook + EMA buffers
        # -------------------------------------------------
        self.codebook.data.copy_(centers)
        self.ema_codebook.data.copy_(centers.clone())
        self.ema_cluster_size.data.fill_(1.0)

        self.initialized = True

    # -------------------------------------------------------------
    # Forward quantization (LFQ + codebook lookup)
    # -------------------------------------------------------------
    def forward(self, z_e):
        """
        z_e: [B, S, D]
        Returns:
            z_q:     [B, S, D]
            indices: [B, S]
        """
        B, S, D = z_e.shape
        N = self.num_codes

        # -------------------------------------------------
        # KMeans init (once)
        # -------------------------------------------------
        if self.training and not self.initialized:
            self.kmeans_init(z_e)

        # -------------------------------------------------
        # Nearest neighbor search (same semantics as reference)
        # -------------------------------------------------
        # [B, S, 1, D]
        z_e_expanded = z_e.unsqueeze(2)

        # [1, 1, N, D]
        codebook_expanded = self.codebook.unsqueeze(0).unsqueeze(0)

        # [B, S, N]
        distances = torch.norm(z_e_expanded - codebook_expanded, dim=-1)

        # [B, S]
        indices = torch.argmin(distances, dim=-1)

        # [B, S, D]
        z_q = self.codebook[indices]

        # -------------------------------------------------
        # EMA updates (TRAINING ONLY)
        # -------------------------------------------------
        if self.training:
            with torch.no_grad():
                # Flatten tokens: [B*S, D]
                z_e_flat = z_e.reshape(B * S, D)
                indices_flat = indices.reshape(B * S)

                # [B*S, N]
                one_hot = F.one_hot(indices_flat, N).float()

                # [N]
                cluster_size = one_hot.sum(dim=0)

                # EMA cluster size
                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1 - self.decay
                )

                # EMA codebook sum
                embed_sum = one_hot.T @ z_e_flat  # [N, D]
                self.ema_codebook.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )

                # Normalize
                n = self.ema_cluster_size.sum()
                cluster_size_norm = (self.ema_cluster_size + self.epsilon) / (
                    n + N * self.epsilon
                )

                new_codebook = self.ema_codebook / cluster_size_norm.unsqueeze(1)
                self.codebook.data.copy_(new_codebook)

            # -------------------------------------------------
            # Utilization tracking
            # -------------------------------------------------
            with torch.no_grad():
                self.usage_counts.add_(cluster_size)
                self.usage_ma.mul_(0.99).add_(cluster_size > 0, alpha=0.01)

                p = cluster_size / (cluster_size.sum() + 1e-8)
                entropy = -(p * (p + 1e-8).log()).sum()
                self.entropy_ma.mul_(0.99).add_(entropy, alpha=0.01)

            # -------------------------------------------------
            # Dead code replacement
            # -------------------------------------------------
            dead = self.usage_counts < self.dead_threshold
            if dead.any():
                dead_idx = dead.nonzero(as_tuple=True)[0]

                if self.replace_strategy == "nearest":
                    alive = (~dead).nonzero(as_tuple=True)[0]
                    if len(alive) > 0:
                        alive_codes = self.codebook[alive]      # [Na, D]
                        dead_codes = self.codebook[dead_idx]    # [Nd, D]

                        # [Nd, Na]
                        dead_norm = (dead_codes ** 2).sum(dim=1, keepdim=True)
                        alive_norm = (alive_codes ** 2).sum(dim=1).unsqueeze(0)
                        dists = dead_norm + alive_norm - 2 * dead_codes @ alive_codes.T

                        nearest = alive[dists.argmin(dim=1)]
                        self.codebook.data[dead_idx] = self.codebook.data[nearest]
                else:
                    rand_ids = torch.randint(
                        0, B * S, (dead_idx.numel(),), device=z_e.device
                    )
                    self.codebook.data[dead_idx] = z_e_flat[rand_ids].detach()

        return z_q, indices

class LipschitzMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(out_dim))
        self.ci = torch.nn.Parameter(torch.ones(out_dim))  # Learnable ci parameter

    def forward(self, x):
        W_norm = normalization(self.W, self.ci)
        return torch.sigmoid(torch.matmul(x, W_norm.T) + self.b)


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




# ============================================================
# NEW HIERARCHICAL HVQ-VAE USING EITHER Z-LEVEL AND Q-LEVEL LFQ
# ============================================================

class TimeMLP(nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.fc1 = nn.Linear(q_dim, 2 * q_dim)
        self.fc2 = nn.Linear(2 * q_dim, q_dim)
        self.fc_last = nn.Linear(q_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        t_hat = self.fc_last(x)
        return t_hat


class HierarchicalLFQHVQVAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        z_dim,
        q_dim,
        num_z_codes=1024,
        num_q_codes=512,
        hidden_dim=128,
    ):
        super().__init__()

        # -------------------------------
        # Encoder (shared LLFQVAE_V4)
        # -------------------------------

        self.encoder = LFQ_VAE(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).encoder  # USE ONLY encoder part

        self.to_z_latent = LFQ_VAE(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).to_latent  # Lipschitz mapping

        # -------------------------------
        # Quantizers (Z then Q)
        # -------------------------------
        # self.z_quantizer = LFQQuantizerEMA(num_z_codes, z_dim)

        self.z_quantizer = LFQQuantizerEMA_KMeans(num_z_codes, z_dim, dead_threshold=3)
        self.q_quantizer = LFQQuantizerEMA_KMeans(num_q_codes, q_dim, dead_threshold=1)
        self.q_encoder = LipschitzMLP(z_dim, q_dim)
        # self.q_quantizer = LFQQuantizerEMA(num_q_codes, q_dim)

        # -------------------------------
        # Decoder (shared LLFQVAE_V4)
        # -------------------------------
        self.decoder = LFQ_VAE(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).decoder

        self.to_output = LFQ_VAE(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).to_output
        self.seq_len=None
        self.time_mlp=TimeMLP(q_dim=q_dim)
        # t = torch.linspace(0, 1, steps=self.seq_len)
        # self.register_buffer("time_buffer", t)  # (T,)

        # self.q_time_head = nn.Sequential(
        #         nn.Linear(q_dim, q_dim // 2),
        #         nn.GELU(),
        #         nn.Linear(q_dim // 2, 1),
        #         nn.Sigmoid()
        #     )
    # ------------------------------------------------------------
    # Helper for losses
    # ------------------------------------------------------------
    def vq_loss(self, z_e, z_q):
        """
        Standard VQ losses adapted to LFQ (detached STE).
        """
        commit = F.mse_loss(z_e, z_q.detach())
        codebook = F.mse_loss(z_q, z_e.detach())
        return commit, codebook

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        B = batch_size
        self.seq_len = seq_len
        x_flat = x.reshape(-1, feature_dim)
        
        # ============================
        # 1) Z-level
        # ============================
        h = self.encoder(x_flat).view(batch_size, seq_len, -1)
        z_e = self.to_z_latent(h)
        z_q, z_idx = self.z_quantizer(z_e)
        
        # FIXED: Clamp the quantization losses
        commit_z = F.mse_loss(z_e, z_q.detach())
        codebook_z = F.mse_loss(z_q, z_e.detach())
        
        # ============================
        # 2) Q-level + TimeMLP
        # ============================
        # CRITICAL FIX: Use stop-gradient properly
        q_cont = self.q_encoder(z_q.detach())
        
        # Time prediction
        t_per_seq = torch.linspace(0, 1, steps=self.seq_len, device=x.device)
        t = t_per_seq.unsqueeze(0).expand(B, self.seq_len)
        t_flat = t.reshape(B * self.seq_len)
        q_cont_flat = q_cont.reshape(B * self.seq_len, -1)
        time_pred = self.time_mlp(q_cont_flat)
        time_loss = F.mse_loss(time_pred.squeeze(-1), t_flat)
        
        # CRITICAL FIX: Detach q_cont before quantization to prevent explosion
        q_q, q_idx = self.q_quantizer(q_cont.detach())
        
        # FIXED: Use straight-through estimator properly
        # The codebook should learn to match q_cont, not the other way around
        commit_q = F.mse_loss(q_cont, q_q.detach())
        # Don't include codebook_q loss - let EMA handle codebook updates
        
        # ============================
        # Reconstruction - use STE
        # ============================
        # Straight-through: backprop through q_cont, forward through q_q
        q_for_decoder = q_cont + (q_q - q_cont).detach()
        
        dec_h = self.decoder(q_for_decoder).view(batch_size, seq_len, -1)
        x_recon = self.to_output(dec_h)
        recon_loss = F.mse_loss(x_recon, x)
        
        # ============================
        # Total loss - REBALANCED
        # ============================
        loss = (
            recon_loss
            + 0.03 * commit_z  # Reduced from 0.03
            + 0.05 * commit_q  # Reduced from 0.05
            + 0.02 * time_loss  # Reduced from 0.02
        )
        print('recon_loss', recon_loss)
        print('z_loss',( 0.03 * (commit_z)))
        print('q_loss', (0.05 * (commit_q )))
        print('time_loss', 0.02*time_loss)
        
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "time_loss": time_loss,
            "z_commit": commit_z,
            "z_codebook": codebook_z,
            "q_commit": commit_q,
            "q_codebook": torch.tensor(0.0),  # Not used with EMA
            "x_recon": x_recon,
            "z_q": z_q,
            "q_q": q_q,
            "z_indices": z_idx,
            "q_indices": q_idx,
        }


 