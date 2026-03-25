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
        B, D = z_e.shape
        n_samples = min(20000, B)  # cap for memory

        sample_idx = torch.randperm(B)[:n_samples]
        sample = z_e[sample_idx].detach().cpu().numpy()

        kmeans = KMeans(n_clusters=self.num_codes, n_init="auto", max_iter=50)
        centers = kmeans.fit(sample).cluster_centers_
        centers = torch.tensor(centers, dtype=z_e.dtype, device=z_e.device)

        self.codebook.data.copy_(centers)
        self.ema_codebook.data.copy_(centers.clone())
        self.initialized = True

    # -------------------------------------------------------------
    # Forward quantization (LFQ + codebook lookup)
    # -------------------------------------------------------------
    def forward(self, z_e):
        B, D = z_e.shape

        # ---- Run KMEANS on first forward ----
        if self.training and not self.initialized:
            self.kmeans_init(z_e)

        # ---- Nearest neighbor search (GPU-accelerated) ----
        # Compute L2 distances directly on GPU
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        z_e_norm = (z_e**2).sum(dim=1, keepdim=True)  # [B, 1]
        cb_norm = (self.codebook**2).sum(dim=1, keepdim=True)  # [num_codes, 1]
        dots = z_e @ self.codebook.T  # [B, num_codes]
        distances = z_e_norm + cb_norm.T - 2 * dots  # [B, num_codes]
        indices = distances.argmin(dim=1)  # [B]

        # Get quantized embeddings
        z_q = self.codebook[indices].clone()

        # ---------------------------------------------------------
        # EMA updates (TRAINING ONLY)
        # ---------------------------------------------------------
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()  # [B, num_codes]
                cluster_size = one_hot.sum(0, keepdim=True)  # [1, num_codes]

                # Update EMA
                cluster_size_squeezed = cluster_size.squeeze(0)
                self.ema_cluster_size.copy_(
                    self.decay * self.ema_cluster_size
                    + (1 - self.decay) * cluster_size_squeezed
                )

                embed_sum = one_hot.T @ z_e  # [num_codes, D]
                self.ema_codebook.copy_(
                    self.decay * self.ema_codebook + (1 - self.decay) * embed_sum
                )

                # -----------------------------
                # FIX: Proper Laplace smoothing
                # -----------------------------
                n = self.ema_cluster_size.sum()

                # cluster_size = (
                #     (self.ema_cluster_size + self.epsilon)
                #     / (n + self.num_codes * self.epsilon)
                # ) * n
                cluster_size_norm = (self.ema_cluster_size + self.epsilon) / (
                    n + self.num_codes * self.epsilon)*n
                new_codebook = self.ema_codebook / cluster_size_norm.unsqueeze(1)
                self.codebook.data.copy_(new_codebook)
            # ---------------------------------------------------------
            # Utilization tracking (TRAINING ONLY)
            # ---------------------------------------------------------
            with torch.no_grad():
                cluster_size_squeezed = cluster_size.squeeze(0)
                self.usage_counts.add_(cluster_size_squeezed)
                self.usage_ma.mul_(0.99).add_(cluster_size_squeezed > 0, alpha=0.01)

                p = cluster_size_squeezed / (cluster_size_squeezed.sum() + 1e-8)
                entropy = -(p * (p + 1e-8).log()).sum()
                self.entropy_ma.mul_(0.99).add_(entropy, alpha=0.01)

            # ---------------------------------------------------------
            # Dead code replacement (TRAINING ONLY)
            # ---------------------------------------------------------
            dead = self.usage_counts < self.dead_threshold
            if dead.any():
                dead_idx = dead.nonzero(as_tuple=True)[0]

                if self.replace_strategy == "nearest":
                    alive = (~dead).nonzero(as_tuple=True)[0]
                    if len(alive) > 0:
                        alive_codes = self.codebook[alive]  # [num_alive, D]
                        dead_codes = self.codebook[dead_idx]  # [num_dead, D]

                        # Compute distances efficiently: [num_dead, num_alive]
                        dead_norm = (dead_codes**2).sum(dim=1, keepdim=True)
                        alive_norm = (alive_codes**2).sum(dim=1, keepdim=True).T
                        dists = (
                            dead_norm + alive_norm - 2 * (dead_codes @ alive_codes.T)
                        )
                        nearest = alive[dists.argmin(dim=1)]

                        self.codebook.data[dead_idx] = self.codebook.data[nearest]
                else:
                    rand_ids = torch.randint(
                        0, B, (dead_idx.shape[0],), device=z_e.device
                    )
                    self.codebook.data[dead_idx] = z_e[rand_ids].detach()


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
        self.codebook = nn.Parameter(
            torch.randn(num_codes, code_dim)
        )  # Learnable codebook
        nn.init.kaiming_uniform_(self.codebook)  # Proper initialization

    def forward(self, z_e):
        batch_size, latent_dim = z_e.shape  # Ensure shape consistency
        z_e_sign = (2 * torch.sign(z_e) + 1).unsqueeze(1)  # Shape: [B, 1, latent_dim]
        z_e_sign = torch.clamp(z_e_sign, max=1)
        z_e_expanded = z_e.unsqueeze(1)  # Shape: [B, 1, D]
        codebook_expanded = self.codebook.unsqueeze(0)  # Shape: [1, num_codes, D]
        distances = torch.norm(
            z_e_sign * (z_e_expanded - codebook_expanded), dim=-1
        )  # Compute L2 distances
        indices = torch.argmin(distances, dim=-1)  # Get closest code
        z_q = self.codebook[indices]  # Retrieve quantized values
        return z_q, indices


class LLFQVAE_V4(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_codes=1024, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_latent = LipschitzMLP(hidden_dim, latent_dim)
        self.quantizer = LFQQuantizer(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)  # Correct shape: [B, hidden_dim]
        z_e = self.to_latent(h)  # Shape: [B, latent_dim]
        z_q, indices = self.quantizer(z_e)  # Shape: [B, latent_dim]
        z_latent = z_q.clone().detach()
        recon = self.decoder(z_q)  # Correct shape: [B, hidden_dim]
        x_recon = self.to_output(recon)  # Shape: [B, feature_dim]

        # Compute losses
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        commitment_loss = F.mse_loss(z_q.detach(), z_e)  # Commitment loss
        codebook_loss = F.mse_loss(z_q, z_e.detach())  # Codebook loss

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


class HVQ(nn.Module):
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

        self.encoder = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).encoder  # USE ONLY encoder part

        self.to_z_latent = LLFQVAE_V4(
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
        self.decoder = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).decoder

        self.to_output = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).to_output
        self.seq_len=16
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
        
        """
        x: (B*T, 12) flattened input
        T: sequence length
        """
        # print(x.shape)
        # input("")
        batch_size, seq_len, feature_dim = x.shape
        self.seq_len=seq_len
        x = x.reshape(batch_size * seq_len, feature_dim)
        B = x.shape[0] // self.seq_len  # compute batch size
        
        # ============================
        # 1) Z-level
        # ============================
        h = self.encoder(x)
        z_e = self.to_z_latent(h)
        z_q, z_idx = self.z_quantizer(z_e)
        commit_z = F.mse_loss(z_e, z_q.detach())
        codebook_z = F.mse_loss(z_q, z_e.detach())

        # ============================
        # 2) Q-level + TimeMLP
        # ============================
        q_cont = self.q_encoder(z_q)

        # Generate time automatically
        t_per_seq = torch.linspace(0, 1, steps=self.seq_len, device=x.device)
        t = t_per_seq.unsqueeze(0).expand(B, self.seq_len)   # (B, T)
        t_flat = t.reshape(B*self.seq_len)

        time_pred = self.time_mlp(q_cont)                 # (B*T, 1)
        time_loss = F.mse_loss(time_pred.squeeze(-1), t_flat)

        # ============================
        # Quantization (unchanged)
        # ============================
        q_q, q_idx = self.q_quantizer(q_cont)
        commit_q = F.mse_loss(q_cont, q_q.detach())
        codebook_q = F.mse_loss(q_q, q_cont.detach())

        # ============================
        # Reconstruction
        # ============================
        dec_h = self.decoder(q_q)
        x_recon = self.to_output(dec_h)
        recon_loss = F.mse_loss(x_recon, x)

        # ============================
        # Total loss
        # ============================
        loss = (
             0.00001*recon_loss
            + 1.0 * (commit_z + codebook_z)
            + 1.0 * (commit_q + codebook_q)
            + 0.00001 * time_loss
        )
        # commit_z   = F.mse_loss(z_e, z_q.detach()) / z_e.shape[-1]
        # codebook_z = F.mse_loss(z_q, z_e.detach()) / z_e.shape[-1]
        # commit_q   = F.mse_loss(q_cont, q_q.detach()) / q_cont.shape[-1]
        # codebook_q = F.mse_loss(q_q, q_cont.detach()) / q_cont.shape[-1]
        commit = 1 * (commit_z + codebook_z)+ 1* (commit_q + codebook_q)
        # print("z_e mean:", z_e.abs().mean())
        # print("z_q mean:", z_q.abs().mean())
        # print("per-dim commit_z:", commit_z / 208)
        # print("per-dim commit_q:", commit_q / 208)
        print('\n')
        print("commit",commit)
        print('time',0.00001*time_loss)
        print("recon",0.00001*recon_loss)
        print('commit_z',commit_z)
        print('codebook_z',codebook_z)
        print('commit_q',commit_q)
        print('codebook_q',codebook_q)
        q_q=q_q.reshape(batch_size, seq_len, feature_dim)
        return q_q,loss

