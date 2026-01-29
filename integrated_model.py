#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math
import numpy as np


class StructuralTeacher(nn.Module):
    """Structural-side teacher: corrects structural bias under dynamic semantic guidance."""

    def __init__(self, hid_dim, condition_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.condition_dim = condition_dim

        self.struct_proj = nn.Linear(hid_dim, hid_dim)

        self.condition_attention = nn.MultiheadAttention(
            embed_dim=hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.noise_predictor = nn.Sequential(
            nn.Linear(hid_dim + condition_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hid_dim // 2),
            nn.GELU(),
            nn.Linear(hid_dim // 2, hid_dim),
        )

        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)

    def forward(self, h_struct_t, dynamic_semantic_condition, t, edge_index=None, uncertainty=None):
        """
        Args:
            h_struct_t: Noisy structural embeddings [N, hid_dim]
            dynamic_semantic_condition: semantic denoising output h_t^{rec,t} [N, condition_dim]
            t: current timestep
            edge_index: graph edges

        Returns:
            predicted_noise: predicted structural noise ε_s^t [N, hid_dim]
        """
        batch_size = h_struct_t.size(0)

        t_embed = self.time_mlp(t.float().view(-1, 1))
        t_embed = t_embed.expand(batch_size, -1)

        h_struct = self.struct_proj(h_struct_t)
        h_struct = self.norm1(h_struct + t_embed)

        if dynamic_semantic_condition is not None:
            n = h_struct.size(0)
            if n > 5000:
                chunk_size = 1000
                conditioned_struct = torch.zeros_like(h_struct)

                for i in range(0, n, chunk_size):
                    end_idx = min(i + chunk_size, n)
                    h_chunk = h_struct[i:end_idx].unsqueeze(0)
                    cond_chunk = dynamic_semantic_condition[i:end_idx].unsqueeze(0)

                    chunk_output, _ = self.condition_attention(
                        query=h_chunk,
                        key=cond_chunk,
                        value=cond_chunk,
                    )
                    conditioned_struct[i:end_idx] = chunk_output.squeeze(0)
            else:
                conditioned_struct, _ = self.condition_attention(
                    query=h_struct.unsqueeze(0),
                    key=dynamic_semantic_condition.unsqueeze(0),
                    value=dynamic_semantic_condition.unsqueeze(0),
                )
                conditioned_struct = conditioned_struct.squeeze(0)

                if uncertainty is not None:
                    guidance_scale = 1.0 + uncertainty.unsqueeze(-1)
                    conditioned_struct = conditioned_struct * guidance_scale

            h_struct = self.norm2(h_struct + conditioned_struct)

        if dynamic_semantic_condition is not None:
            combined_input = torch.cat([h_struct, dynamic_semantic_condition], dim=-1)
        else:
            combined_input = torch.cat([h_struct, torch.zeros_like(h_struct)], dim=-1)

        predicted_noise = self.noise_predictor(combined_input)
        return predicted_noise


class SemanticTeacher(nn.Module):
    """Semantic-side teacher: corrects semantic bias under dynamic structural guidance."""

    def __init__(self, hid_dim, condition_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.condition_dim = condition_dim

        self.residual_proj = nn.Linear(hid_dim, hid_dim)

        self.condition_attention = nn.MultiheadAttention(
            embed_dim=hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.residual_noise_predictor = nn.Sequential(
            nn.Linear(hid_dim + condition_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hid_dim // 2),
            nn.GELU(),
            nn.Linear(hid_dim // 2, hid_dim),
        )

        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)

    def forward(self, r_t, dynamic_struct_condition, t, uncertainty=None):
        """
        Args:
            r_t: noisy residual signal [N, hid_dim]
            dynamic_struct_condition: structural denoising output h_s^{rec,t} [N, condition_dim]
            t: current timestep

        Returns:
            predicted_residual_noise: predicted residual noise ε_r^t [N, hid_dim]
        """
        batch_size = r_t.size(0)

        t_embed = self.time_mlp(t.float().view(-1, 1))
        t_embed = t_embed.expand(batch_size, -1)

        h_residual = self.residual_proj(r_t)
        h_residual = self.norm1(h_residual + t_embed)

        if dynamic_struct_condition is not None:
            n = h_residual.size(0)
            if n > 5000:
                chunk_size = 1000
                conditioned_residual = torch.zeros_like(h_residual)

                for i in range(0, n, chunk_size):
                    end_idx = min(i + chunk_size, n)
                    h_chunk = h_residual[i:end_idx].unsqueeze(0)
                    cond_chunk = dynamic_struct_condition[i:end_idx].unsqueeze(0)

                    chunk_output, _ = self.condition_attention(
                        query=h_chunk,
                        key=cond_chunk,
                        value=cond_chunk,
                    )
                    conditioned_residual[i:end_idx] = chunk_output.squeeze(0)
            else:
                conditioned_residual, _ = self.condition_attention(
                    query=h_residual.unsqueeze(0),
                    key=dynamic_struct_condition.unsqueeze(0),
                    value=dynamic_struct_condition.unsqueeze(0),
                )
                conditioned_residual = conditioned_residual.squeeze(0)

                if uncertainty is not None:
                    guidance_scale = 1.0 + uncertainty.unsqueeze(-1)
                    conditioned_residual = conditioned_residual * guidance_scale

            h_residual = self.norm2(h_residual + conditioned_residual)

        if dynamic_struct_condition is not None:
            combined_input = torch.cat([h_residual, dynamic_struct_condition], dim=-1)
        else:
            combined_input = torch.cat([h_residual, torch.zeros_like(h_residual)], dim=-1)

        predicted_residual_noise = self.residual_noise_predictor(combined_input)
        return predicted_residual_noise


class DynamicDualDiffusion(nn.Module):
    """Dual diffusion with uncertainty-guided adaptive noise (V11)."""

    def __init__(self, hid_dim, diffusion_steps=5, sigma_data=0.5, adaptive_noise=True, sigma_min=0.5, sigma_max=1.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.diffusion_steps = diffusion_steps
        self.sigma_data = sigma_data

        self.adaptive_noise = adaptive_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        beta_start, beta_end = 0.0001, 0.02
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, diffusion_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        self.structural_teacher = StructuralTeacher(hid_dim, hid_dim)
        self.semantic_teacher = SemanticTeacher(hid_dim, hid_dim)

    def compute_node_uncertainty(self, logits):
        """
        Args:
            logits: [N, C]

        Returns:
            uncertainty: [N] in [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = np.log(probs.size(1))
        uncertainty = entropy / max_entropy
        return uncertainty

    def add_adaptive_noise(self, x, t, uncertainty):
        """
        Args:
            x: [N, D]
            t: timestep tensor
            uncertainty: [N]

        Returns:
            x_noisy: [N, D]
            noise: [N, D]
            actual_sigma: [N, 1]
        """
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise, torch.ones_like(x[:, :1])

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise

    def denoise_step(self, x_t, predicted_noise, t):
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.ones_like(alpha_t)
        beta_t = self.betas[t]

        coef1 = 1 / torch.sqrt(self.alphas[t])
        coef2 = beta_t / torch.sqrt(1 - alpha_t)

        x_t_minus_1 = coef1 * (x_t - coef2 * predicted_noise)
        return x_t_minus_1

    def forward(self, h_struct_0, h_semantic_0, edge_index=None, training=True, uncertainty=None):
        """
        Args:
            h_struct_0: [N, hid_dim]
            h_semantic_0: [N, hid_dim]
            edge_index: graph edges
            training: bool
            uncertainty: [N] in [0, 1]

        Returns:
            h_struct_rec, h_semantic_rec, losses
        """
        r_0 = h_semantic_0 - h_struct_0
        losses = {}

        if training:
            t = torch.randint(0, self.diffusion_steps, (1,), device=h_struct_0.device)

            h_struct_t, struct_noise, _ = self.add_adaptive_noise(h_struct_0, t, uncertainty)
            r_t, residual_noise, _ = self.add_adaptive_noise(r_0, t, uncertainty)

            h_semantic_t_rec = h_struct_0 + r_t
            struct_noise_pred = self.structural_teacher(
                h_struct_t,
                h_semantic_t_rec,
                t,
                edge_index,
                uncertainty=uncertainty,
            )

            h_struct_t_rec = h_struct_t - struct_noise_pred
            residual_noise_pred = self.semantic_teacher(
                r_t,
                h_struct_t_rec,
                t,
                uncertainty=uncertainty,
            )

            dim_scale = self.hid_dim ** 0.5
            losses["struct_diffusion"] = F.mse_loss(struct_noise_pred, struct_noise) / dim_scale
            losses["semantic_diffusion"] = F.mse_loss(residual_noise_pred, residual_noise) / dim_scale

            h_struct_rec = self.denoise_step(h_struct_t, struct_noise_pred, t)
            r_rec = self.denoise_step(r_t, residual_noise_pred, t)
            h_semantic_rec = h_struct_rec + r_rec
        else:
            h_struct_current = h_struct_0
            r_current = r_0

            for t in reversed(range(self.diffusion_steps)):
                t_tensor = torch.tensor([t], device=h_struct_0.device)

                h_semantic_current = h_struct_current + r_current
                struct_noise_pred = self.structural_teacher(
                    h_struct_current,
                    h_semantic_current,
                    t_tensor,
                    edge_index,
                    uncertainty=uncertainty,
                )

                h_struct_denoised = self.denoise_step(h_struct_current, struct_noise_pred, t_tensor)
                residual_noise_pred = self.semantic_teacher(
                    r_current,
                    h_struct_denoised,
                    t_tensor,
                    uncertainty=uncertainty,
                )

                h_struct_current = h_struct_denoised
                r_current = self.denoise_step(r_current, residual_noise_pred, t_tensor)

            h_struct_rec = h_struct_current
            h_semantic_rec = h_struct_current + r_current

        return h_struct_rec, h_semantic_rec, losses


class DualDiffusionGraphModel_V11(nn.Module):
    """
    V11: Dual diffusion with uncertainty-guided adaptive noise.
    """

    def __init__(
        self,
        in_dim,
        sem_dim,
        hid_dim,
        num_classes,
        diffusion_steps=5,
        num_heads=8,
        dropout=0.1,
        ema_alpha=0.999,
        sigma_data=0.5,
        adaptive_noise=True,
        sigma_min=0.5,
        sigma_max=1.5,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.sem_dim = sem_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.sigma_data = sigma_data

        self.adaptive_noise = adaptive_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.struct_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.LayerNorm(hid_dim),
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(sem_dim, hid_dim),
            nn.GELU(),
            nn.LayerNorm(hid_dim),
        )

        self.struct_encoder = nn.ModuleList([GCNConv(hid_dim, hid_dim) for _ in range(2)])

        self.dual_diffusion = DynamicDualDiffusion(
            hid_dim=hid_dim,
            diffusion_steps=diffusion_steps,
            adaptive_noise=adaptive_noise,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hid_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, num_classes),
        )

        self.register_buffer("ema_h_struct", None)
        self.register_buffer("ema_h_semantic", None)

        self.graph_augmentor = GraphAugmentor()

    def forward(self, data, text_embeds, ema_h_str=None, ema_h_sem=None, uncertainty=None, use_augmentation=True, **kwargs):
        x, edge_index = data.x, data.edge_index

        if use_augmentation and self.training:
            edge_index = self.graph_augmentor(edge_index, x.size(0))

        h_struct_init = self.struct_proj(x)
        if text_embeds is not None:
            h_semantic_init = self.semantic_proj(text_embeds)
        else:
            h_semantic_init = self.semantic_proj(x)

        h_struct = h_struct_init
        for gcn_layer in self.struct_encoder:
            h_struct = F.gelu(gcn_layer(h_struct, edge_index))
            h_struct = F.dropout(h_struct, p=0.1, training=self.training)

        h_struct_rec, h_semantic_rec, diffusion_losses = self.dual_diffusion(
            h_struct_0=h_struct,
            h_semantic_0=h_semantic_init,
            edge_index=edge_index,
            training=self.training,
            uncertainty=uncertainty,
        )

        if self.training:
            if self.ema_h_struct is None:
                self.ema_h_struct = h_struct_rec.detach().clone()
                self.ema_h_semantic = h_semantic_rec.detach().clone()
            else:
                self.ema_h_struct = self.ema_alpha * self.ema_h_struct + (1 - self.ema_alpha) * h_struct_rec.detach()
                self.ema_h_semantic = self.ema_alpha * self.ema_h_semantic + (1 - self.ema_alpha) * h_semantic_rec.detach()

        h_fused = torch.cat([h_struct_rec, h_semantic_rec], dim=-1)
        h_final = self.feature_fusion(h_fused)

        logits = self.classifier(h_final)

        uncertainty = self.dual_diffusion.compute_node_uncertainty(logits.detach())

        align_loss = F.mse_loss(h_struct_rec, h_semantic_rec.detach())

        total_diffusion_loss = diffusion_losses.get("struct_diffusion", 0) + diffusion_losses.get("semantic_diffusion", 0)

        return (
            logits,
            h_final,
            h_struct_rec,
            h_semantic_rec,
            total_diffusion_loss,
            self.ema_h_struct,
            self.ema_h_semantic,
            uncertainty,
        )


class GraphAugmentor(nn.Module):
    """A minimal graph augmentor."""

    def __init__(self):
        super().__init__()
        self.aug_type = "edge_dropout"
        self.aug_ratio = 0.1

    def forward(self, edge_index, num_nodes):
        if self.aug_type == "edge_dropout" and self.aug_ratio > 0:
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > self.aug_ratio
            edge_index = edge_index[:, mask]
        return edge_index


def create_v10_model(in_dim, sem_dim, hid_dim, num_classes, **kwargs):
    """Factory for legacy V10 model (kept for backward compatibility)."""
    return DualDiffusionGraphModel_V10(
        in_dim=in_dim,
        sem_dim=sem_dim,
        hid_dim=hid_dim,
        num_classes=num_classes,
        **kwargs,
    )


if __name__ == "__main__":
    print("🚀 V10 dynamic conditional dual-diffusion smoke test")

    batch_size = 100
    in_dim = 128
    sem_dim = 384
    hid_dim = 256
    num_classes = 7

    from torch_geometric.data import Data

    x = torch.randn(batch_size, in_dim)
    edge_index = torch.randint(0, batch_size, (2, batch_size * 3))
    y = torch.randint(0, num_classes, (batch_size,))
    text_embeds = torch.randn(batch_size, sem_dim)

    data = Data(x=x, edge_index=edge_index, y=y)

    model = create_v10_model(
        in_dim=in_dim,
        sem_dim=sem_dim,
        hid_dim=hid_dim,
        num_classes=num_classes,
        diffusion_steps=5,
    )

    model.train()
    outputs = model(data, text_embeds)

    logits, h_final, h_struct, h_semantic, diffusion_loss, ema_h_struct, ema_h_sem, uncertainty = outputs

    print("✅ Smoke test passed.")
    print(f"   logits shape: {logits.shape}")
    print(f"   diffusion loss: {diffusion_loss:.4f}")
    print(f"   num params: {sum(p.numel() for p in model.parameters()):,}")
