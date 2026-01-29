#!/usr/bin/env python3
"""
Usage:
python train.py \
    --split balanced --epochs 400 --sigma_min 0.5 --sigma_max 1.5 --seed 42 \
    --gpu_id 0 --patience 150 \
    --hid_dim 256 --diffusion_steps 5 --num_heads 8 --dropout 0.4 \
    --contrast_weight 0.001 --contrast_temp 0.5 --lap_weight 0.01 \
    --lr 0.0003 --weight_decay 0.003 --label_smoothing 0.2 \
    --t_0 50 --t_mult 2 \
    --target_ratio 0.2 --sigma_data 0.5 --ema_alpha 0.999 \
    --adaptive_noise True \
    --aug_type edge_dropout --aug_ratio 0.1
"""

import os
import sys
import argparse
import numpy as np
import torch

# ============================================================
# Compatibility patch: PyTorch 2.1 vs Transformers pytree API
# Must be placed before importing transformers / torch_geometric
# ============================================================
import torch.utils._pytree as pytree


def safe_register_pytree_node(typ, flatten_func, unflatten_func, serialized_type_name=None):
    # Ignore serialized_type_name passed by newer transformers versions
    return pytree._register_pytree_node(typ, flatten_func, unflatten_func)


if not hasattr(pytree, "register_pytree_node"):
    pytree.register_pytree_node = safe_register_pytree_node
# ============================================================

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_dataset import get_cora_casestudy, get_raw_text_cora
from integrated_model import DualDiffusionGraphModel_V11

# SentenceTransformer support
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ Warning: sentence-transformers is not installed; will fall back to raw features.")


class EMA:
    """Exponential Moving Average."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_contrastive_loss(h_struct, h_semantic, labels, temperature=0.07):
    """
    Contrastive loss (InfoNCE-style).

    Args:
        h_struct: Structural embeddings [N, D]
        h_semantic: Semantic embeddings [N, D]
        labels: Node labels [N]
        temperature: Temperature

    Returns:
        contrast_loss: scalar
    """
    h_struct_norm = F.normalize(h_struct, p=2, dim=-1)
    h_semantic_norm = F.normalize(h_semantic, p=2, dim=-1)

    sim_matrix = torch.mm(h_struct_norm, h_semantic_norm.t()) / temperature

    labels_expand = labels.unsqueeze(1)
    pos_mask = (labels_expand == labels_expand.t()).float()

    identity_mask = torch.eye(pos_mask.size(0), device=pos_mask.device)
    pos_mask = pos_mask * (1 - identity_mask)

    pos_counts = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_counts.squeeze()
    contrast_loss = -pos_log_prob.mean()

    return contrast_loss


def off_diagonal(x):
    """Return all off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_barlow_twins_loss(z1, z2, lambd=0.005):
    """
    Barlow Twins loss (redundancy reduction via cross-correlation).

    Args:
        z1: View 1 features [N, D] (e.g., structural)
        z2: View 2 features [N, D] (e.g., semantic)
        lambd: Weight for off-diagonal penalty

    Returns:
        loss: scalar
    """
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)

    N, D = z1_norm.size()

    c = torch.mm(z1_norm.T, z2_norm) / N

    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()

    loss = on_diag + lambd * off_diag
    return loss


def compute_laplacian_loss(h, edge_index):
    """
    Graph Laplacian regularization (feature smoothness over edges).

    Args:
        h: Node features [N, D]
        edge_index: Edge indices [2, E]

    Returns:
        loss: scalar
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    h_src = h[src_idx]
    h_dst = h[dst_idx]

    squared_diff = (h_src - h_dst).pow(2).sum(dim=1)
    loss = squared_diff.mean()
    return loss


def train_epoch(
    model,
    data,
    optimizer,
    epoch,
    total_epochs,
    text_embeds,
    target_ratio=0.2,
    ema_h_str=None,
    ema_h_sem=None,
    uncertainty=None,
    contrast_weight=0.0,
    contrast_temp=0.5,
    label_smoothing=0.2,
    lap_weight=0.01,
):
    """
    One training epoch (V11): classification + diffusion + (optional) Barlow Twins + Laplacian.
    """
    model.train()
    optimizer.zero_grad()

    out, h_final, h_str, h_sem, diffusion_loss, ema_h_str, ema_h_sem, uncertainty = model(
        data,
        text_embeds,
        ema_h_str=ema_h_str,
        ema_h_sem=ema_h_sem,
        uncertainty=uncertainty,
        use_augmentation=True,
        epoch=epoch,
    )

    cls_loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask],
        label_smoothing=label_smoothing,
    )

    base_weight = 0.5
    progress = epoch / total_epochs
    diffusion_weight = base_weight * (1.0 + 0.5 * progress)

    bt_loss = torch.tensor(0.0, device=out.device)
    if contrast_weight > 0:
        bt_loss = compute_barlow_twins_loss(
            h_str[data.train_mask],
            h_sem[data.train_mask],
            lambd=0.005,
        )

    lap_loss = compute_laplacian_loss(h_final, data.edge_index)

    total_loss = (
        cls_loss
        + (diffusion_weight * diffusion_loss)
        + (contrast_weight * bt_loss)
        + (lap_weight * lap_loss)
    )

    total_loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()

    diagnostics = {
        "grad_norm": total_norm,
        "diffusion_contribution": (diffusion_weight * diffusion_loss).item(),
        "bt_loss": bt_loss.item(),
        "lap_loss": lap_loss.item(),
        "lap_contribution": (lap_weight * lap_loss).item(),
        "bt_contribution": (contrast_weight * bt_loss).item(),
    }

    with torch.no_grad():
        diagnostics.update(
            {
                "h_str_mean": h_str.mean().item(),
                "h_str_std": h_str.std().item(),
                "h_sem_mean": h_sem.mean().item(),
                "h_sem_std": h_sem.std().item(),
                "h_final_mean": h_final.mean().item(),
                "h_final_std": h_final.std().item(),
                "out_mean": out.mean().item(),
                "out_std": out.std().item(),
                "out_min": out.min().item(),
                "out_max": out.max().item(),
                "ema_available": (ema_h_str is not None) and (ema_h_sem is not None),
            }
        )

    return (
        total_loss.item(),
        diffusion_loss.item(),
        cls_loss.item(),
        train_acc,
        diffusion_weight,
        1.0,
        ema_h_str,
        ema_h_sem,
        uncertainty,
        diagnostics,
        bt_loss.item(),
        lap_loss.item(),
    )


@torch.no_grad()
def evaluate(model, data, text_embeds, mask, ema_h_str=None, ema_h_sem=None, uncertainty=None):
    """Model evaluation."""
    model.eval()

    out, h_final, h_str, h_sem, diffusion_loss, ema_h_str, ema_h_sem, uncertainty = model(
        data,
        text_embeds,
        ema_h_str=ema_h_str,
        ema_h_sem=ema_h_sem,
        uncertainty=uncertainty,
        use_augmentation=False,
    )

    pred = out.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, precision, recall, f1, ema_h_str, ema_h_sem, uncertainty


def get_text_embeddings(texts, device="cuda", cache_file="text_embeddings_cache_cora_v10.pt"):
    """Generate or load cached text embeddings."""
    if os.path.exists(cache_file):
        print(f"📂 Loading cached text embeddings: {cache_file}")
        text_embeds = torch.load(cache_file, map_location="cpu")
        return text_embeds.to(device)

    if not SBERT_AVAILABLE:
        print("❌ Cannot generate embeddings: sentence-transformers is not installed.")
        return None

    print("🔄 Loading SentenceTransformer model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("🔄 Encoding texts...")
    text_embeds = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=True, device=device)
    text_embeds = text_embeds.to(torch.float32)

    print(f"💾 Saving embeddings cache: {cache_file}")
    torch.save(text_embeds.cpu(), cache_file)

    return text_embeds.to(device)


def create_balanced_split(data, seed=42):
    """Create a class-balanced 60/20/20 split."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_nodes = data.y.size(0)
    num_classes = int(data.y.max()) + 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for class_idx in range(num_classes):
        class_nodes = (data.y == class_idx).nonzero(as_tuple=False).squeeze()
        class_nodes = class_nodes[torch.randperm(len(class_nodes))]

        n_class = len(class_nodes)
        n_train = int(0.6 * n_class)
        n_val = int(0.2 * n_class)

        train_mask[class_nodes[:n_train]] = True
        val_mask[class_nodes[n_train : n_train + n_val]] = True
        test_mask[class_nodes[n_train + n_val :]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("🚀 V11 Training (Full SBERT) - Cora")
    print("=" * 80)

    print("🧠 V11 core: uncertainty-guided adaptive noise")
    print(f"   ✓ Adaptive noise: {'ON' if args.adaptive_noise else 'OFF (V10-compatible)'}")
    if args.adaptive_noise:
        print(f"   ✓ Noise range: σ_min={args.sigma_min}, σ_max={args.sigma_max}")
        print("   ✓ Policy: hard nodes -> larger noise, easy nodes -> smaller noise")

    print("\n📊 Loss configuration:")
    print(f"   ✓ Classification: CrossEntropy (label_smoothing={args.label_smoothing})")
    print("   ✓ Diffusion: dynamic weight (base=0.5 with progress scaling)")
    if args.contrast_weight > 0:
        print(f"   ✓ Barlow Twins: weight={args.contrast_weight}, temp={args.contrast_temp}")
        lap_weight = getattr(args, "lap_weight", 0.01)
        print(f"   ✓ Laplacian: weight={lap_weight}")
    else:
        print("   ✓ Barlow Twins: OFF (weight=0)")
        print("   ✓ Laplacian: ON (default)")

    print("\n🔧 Model configuration:")
    print(f"   ✓ Hidden dim: {args.hid_dim}")
    print(f"   ✓ Diffusion steps: {args.diffusion_steps}")
    print(f"   ✓ Attention heads: {args.num_heads}")
    print(f"   ✓ Dropout: {args.dropout}")
    print(f"   ✓ EMA alpha: {args.ema_alpha}")
    print(f"   ✓ Sigma_data: {args.sigma_data}")

    print("\n⚙️ Training configuration:")
    print(f"   ✓ LR: {args.lr}")
    print(f"   ✓ Weight decay: {args.weight_decay}")
    print(f"   ✓ Cosine restarts: T_0={args.t_0}, T_mult={args.t_mult}")
    print(f"   ✓ Early-stop patience: {args.patience} epochs")

    print("\n🔄 Graph augmentation:")
    print(f"   ✓ Type: {args.aug_type}")
    print(f"   ✓ Ratio: {args.aug_ratio}")

    print("\n🎯 Expected performance (reference only):")
    print("   ✓ V10 baseline (standard split): 93.36%")
    if args.adaptive_noise:
        if args.sigma_max <= 1.5:
            print("   ✓ V11 conservative target: 93.5–93.8%")
        else:
            print("   ✓ V11 aggressive target: 93.6–94.0%")
    else:
        print("   ✓ V11 compatible mode: ~93.3–93.4%")

    print("\n📂 Loading Cora dataset...")
    data, texts = get_raw_text_cora(use_text=True, seed=args.seed)

    print("🔄 [Structural branch] Preparing Full SBERT features (Title + Abstract)...")
    full_sbert = get_text_embeddings(texts, device=device, cache_file="text_embeddings_cache_cora_v10.pt")

    llm_feat_path = "cora_llm_keywords.pt"
    if os.path.exists(llm_feat_path):
        print(f"🔥 [Semantic branch] Loading LLM-refined features ({llm_feat_path})...")
        llm_keywords = torch.load(llm_feat_path, map_location=device)
    else:
        raise RuntimeError("❌ Missing LLM keyword feature file. Run preprocessing first.")

    data.x = full_sbert.clone()
    current_in_dim = full_sbert.shape[1]
    text_embeds = llm_keywords.clone()

    print("✅ Feature assignment (asymmetric):")
    print(f"   GCN input (data.x): Full SBERT {data.x.shape}")
    print(f"   Teacher input (text_embeds): LLM Keywords {text_embeds.shape}")
    print("   Goal: use clean LLM semantics to refine aggregation over full text")

    num_classes = int(data.y.max().item()) + 1

    if args.split == "balanced":
        data = create_balanced_split(data, seed=args.seed)

    data = data.to(device)
    text_embeds = text_embeds.to(device)

    print("\n🚀 Initializing V11 model...")
    model = DualDiffusionGraphModel_V11(
        in_dim=current_in_dim,
        sem_dim=text_embeds.shape[1],
        hid_dim=args.hid_dim,
        num_classes=num_classes,
        diffusion_steps=args.diffusion_steps,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ema_alpha=args.ema_alpha,
        sigma_data=args.sigma_data,
        adaptive_noise=args.adaptive_noise,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("📊 Parameter stats:")
    print(f"   ✓ Total params: {total_params:,}")
    print(f"   ✓ Trainable params: {trainable_params:,}")
    print(f"   ✓ Input dims: structural={current_in_dim}, semantic={text_embeds.shape[1]}")
    print(f"   ✓ Classes: {num_classes}")

    model.graph_augmentor.aug_type = args.aug_type
    model.graph_augmentor.aug_ratio = args.aug_ratio

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t_0,
        T_mult=args.t_mult,
        eta_min=args.lr * 0.01,
    )

    print(f"\n🚀 Start training ({'adaptive noise' if args.adaptive_noise else 'V10-compatible'} mode)...")
    print("=" * 80)

    print("🔍 Environment check:")
    print(f"   ✓ Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(args.gpu_id)
        gpu_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3
        print(f"   ✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    print(f"   ✓ Dataset: Cora ({data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges)")
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()
    print(f"   ✓ Split: train={train_count}, val={val_count}, test={test_count}")
    print("=" * 80)

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    ema_h_str, ema_h_sem, uncertainty = None, None, None

    for epoch in range(1, args.epochs + 1):
        lap_weight = getattr(args, "lap_weight", 0.01)
        (
            train_loss,
            diffusion_loss,
            cls_loss,
            train_acc,
            diffusion_weight_used,
            adjust_factor,
            ema_h_str,
            ema_h_sem,
            uncertainty,
            diagnostics,
            bt_loss_val,
            lap_loss_val,
        ) = train_epoch(
            model,
            data,
            optimizer,
            epoch,
            args.epochs,
            text_embeds,
            target_ratio=args.target_ratio,
            ema_h_str=ema_h_str,
            ema_h_sem=ema_h_sem,
            uncertainty=uncertainty,
            contrast_weight=args.contrast_weight,
            contrast_temp=args.contrast_temp,
            label_smoothing=args.label_smoothing,
            lap_weight=lap_weight,
        )

        val_acc, val_p, val_r, val_f1, ema_h_str, ema_h_sem, uncertainty = evaluate(
            model, data, text_embeds, data.val_mask, ema_h_str, ema_h_sem, uncertainty
        )
        test_acc, test_p, test_r, test_f1, ema_h_str, ema_h_sem, uncertainty = evaluate(
            model, data, text_embeds, data.test_mask, ema_h_str, ema_h_sem, uncertainty
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "ema_h_str": ema_h_str,
                    "ema_h_sem": ema_h_sem,
                    "uncertainty": uncertainty,
                },
                "best_cora_v10_optimized.pt",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            diffusion_ratio = diffusion_loss / (cls_loss + 1e-8)
            diffusion_contribution = diffusion_weight_used * diffusion_loss
            progress_percent = epoch / args.epochs * 100

            if train_acc < 0.95:
                state_emoji = "🌱"
                state_msg = "learning"
            elif train_acc >= 0.99:
                state_emoji = "⚠️"
                state_msg = "very high train acc"
            else:
                state_emoji = "✅"
                state_msg = "good"

            title = "V11 Cora (adaptive noise)" if args.adaptive_noise else "V11 Cora (V10-compatible)"
            print(f"\nEpoch {epoch:03d}/{args.epochs} - {title}:")
            print(f"  {state_emoji} Status: {state_msg} (progress: {progress_percent:.1f}%)")

            if args.contrast_weight > 0:
                bt_contribution = args.contrast_weight * bt_loss_val
                lap_contribution = lap_weight * lap_loss_val
                print(f"  Loss: total={train_loss:.4f} | cls={cls_loss:.4f} | diff={diffusion_loss:.4f}")
                print(
                    f"  Reg: BT={bt_loss_val:.4f}(×{args.contrast_weight:.3f}) | "
                    f"Lap={lap_loss_val:.4f}(×{lap_weight:.3f})"
                )
                print(f"  LR: {current_lr:.2e} | diff_weight: {diffusion_weight_used:.3f}")
                print(
                    f"  Contribution vs cls: diff={diffusion_contribution/cls_loss*100:.0f}%, "
                    f"BT={bt_contribution/cls_loss*100:.0f}%, "
                    f"Lap={lap_contribution/cls_loss*100:.0f}%"
                )
                if bt_loss_val > 1.0:
                    print(f"  ⚠️ BT loss is large ({bt_loss_val:.2f}); consider lowering contrast_weight")
            else:
                print(
                    f"  Loss: total={train_loss:.4f} | cls={cls_loss:.4f} | "
                    f"diff={diffusion_loss:.4f} | lap={lap_loss_val:.4f}"
                )
                print(
                    f"  LR: {current_lr:.2e} | diff_weight: 0.5×(1+0.5×{progress_percent/100:.2f}) = "
                    f"{diffusion_weight_used:.3f}"
                )
                print(
                    f"  Contribution: diff={diffusion_contribution/cls_loss:.2f}x, "
                    f"lap={lap_weight * lap_loss_val/cls_loss:.2f}x"
                )

            if args.adaptive_noise and uncertainty is not None and hasattr(uncertainty, "mean"):
                unc_mean = uncertainty.mean().item()
                unc_std = uncertainty.std().item()
                unc_min = uncertainty.min().item()
                unc_max = uncertainty.max().item()
                print(
                    f"  Uncertainty: mean={unc_mean:.3f}±{unc_std:.3f}, "
                    f"range=[{unc_min:.3f}-{unc_max:.3f}]"
                )

            print(f"  Acc: train={train_acc:.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            print(f"  Val metrics: P={val_p:.3f}, R={val_r:.3f}, F1={val_f1:.3f}")
            print(f"  Best val: {best_val_acc:.4f} @ epoch {best_epoch}")
            print(f"  Ratio: diff/cls={diffusion_ratio:.1f}, effective={diffusion_contribution/cls_loss:.2f}x")

            if epoch % 30 == 0 or epoch == 1:
                print("  🔍 Diagnostics:")
                print(
                    f"     Feature stats: h_str={diagnostics['h_str_mean']:.3f}±{diagnostics['h_str_std']:.3f}, "
                    f"h_sem={diagnostics['h_sem_mean']:.3f}±{diagnostics['h_sem_std']:.3f}, "
                    f"h_final={diagnostics['h_final_mean']:.3f}±{diagnostics['h_final_std']:.3f}"
                )
                print(
                    f"     Output stats: mean={diagnostics['out_mean']:.3f}±{diagnostics['out_std']:.3f}, "
                    f"range=[{diagnostics['out_min']:.3f}, {diagnostics['out_max']:.3f}]"
                )
                print(
                    f"     Grad norm: {diagnostics['grad_norm']:.3f}, EMA available: {diagnostics['ema_available']}"
                )
                if args.adaptive_noise:
                    print(
                        f"     V11: BT={diagnostics['bt_loss']:.4f}, Lap={diagnostics['lap_loss']:.4f}"
                    )
                    print(
                        f"     Reg contrib: BT={diagnostics.get('bt_contribution', 0):.4f}, "
                        f"Lap={diagnostics.get('lap_contribution', 0):.4f}"
                    )

        if val_acc > best_val_acc - 0.001:
            if test_acc >= 0.93:
                print(f"  🎯 Hit 93% target! test_acc={test_acc:.4f}")
            if test_acc >= 0.95:
                print(f"  🏆 Break 95%! test_acc={test_acc:.4f}")

        if patience_counter >= args.patience:
            print(f"\n⏹️ Early stop @ epoch {epoch}")
            break

    print("\n📊 Loading best checkpoint for final test...")
    checkpoint = torch.load("best_cora_v10_optimized.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_h_str = checkpoint["ema_h_str"]
    ema_h_sem = checkpoint["ema_h_sem"]
    uncertainty = checkpoint["uncertainty"]

    print(f"✅ Loaded best model from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.4f})")

    final_test_acc, final_test_p, final_test_r, final_test_f1, _, _, _ = evaluate(
        model, data, text_embeds, data.test_mask, ema_h_str, ema_h_sem, uncertainty
    )

    model.eval()
    with torch.no_grad():
        out, _, _, _, _, _, _, _ = model(
            data,
            text_embeds,
            ema_h_str=ema_h_str,
            ema_h_sem=ema_h_sem,
            uncertainty=uncertainty,
            use_augmentation=False,
        )
        pred = out[data.test_mask].argmax(dim=1)
        cm = confusion_matrix(data.y[data.test_mask].cpu(), pred.cpu())

    print("\n" + "=" * 60)
    print(f"🏆 Final Result - Cora ({args.split.upper()} split)")
    print("=" * 60)
    print(f"Test Acc: {final_test_acc:.4f}")
    print(f"Precision: {final_test_p:.4f} | Recall: {final_test_r:.4f} | F1: {final_test_f1:.4f}")
    print(f"Best Val: {best_val_acc:.4f} @ epoch {best_epoch}")

    print("\nV11 config summary:")
    if args.adaptive_noise:
        print(f"  ✓ Adaptive noise: ON (σ range: {args.sigma_min}-{args.sigma_max})")
        print("  ✓ Policy: uncertainty-guided (hard->big noise, easy->small noise)")
    else:
        print("  ✓ Adaptive noise: OFF (V10-compatible)")

    print(
        f"  ✓ Loss: cls + diff + BT({args.contrast_weight}) + Lap({getattr(args, 'lap_weight', 0.01)})"
    )
    print(f"  ✓ Optim: AdamW(lr={args.lr}, wd={args.weight_decay}) + cosine restarts")
    print(f"  ✓ Reg: dropout({args.dropout}) + label_smoothing({args.label_smoothing})")
    print(f"  ✓ Aug: {args.aug_type}({args.aug_ratio})")

    print("\nPerformance note:")
    if args.split == "standard":
        print("  Reference V10 (standard split): 93.36%")
        print(
            f"  V11 ({'adaptive' if args.adaptive_noise else 'compatible'}): "
            f"{final_test_acc:.4f} ({final_test_acc:.2%})"
        )
        improvement = (final_test_acc - 0.9336) * 100
        if improvement > 0.2:
            print(f"  🎉 Significant gain: +{improvement:.2f}%")
        elif improvement > 0:
            print(f"  ✅ Small gain: +{improvement:.2f}%")
        elif improvement > -0.2:
            print(f"  ⚖️ Roughly tied: {improvement:.2f}%")
        else:
            print(f"  ⚠️ Drop: {improvement:.2f}%")

        if args.adaptive_noise:
            noise_range = args.sigma_max - args.sigma_min
            if noise_range >= 1.0:
                print(f"  Noise policy: aggressive range ({noise_range:.1f})")
            else:
                print(f"  Noise policy: conservative range ({noise_range:.1f})")
    else:
        print("  Reference V10 (standard split): 93.36%")
        print(f"  V11 (balanced split): {final_test_acc:.4f} ({final_test_acc:.2%})")
        print("  Note: different split, not directly comparable.")

    print(f"\nConfusion matrix:\n{cm}")
    print("=" * 60)

    print("\nPer-class accuracy:")
    for i in range(num_classes):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Class {i}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11 training script - Cora (uncertainty-guided adaptive noise)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400, help="Total epochs")
    parser.add_argument("--patience", type=int, default=150, help="Early-stop patience")
    parser.add_argument("--split", type=str, default="balanced")

    parser.add_argument("--hid_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--diffusion_steps", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")

    parser.add_argument("--contrast_weight", type=float, default=0.0, help="Barlow Twins loss weight (0=off)")
    parser.add_argument("--contrast_temp", type=float, default=0.5, help="Temperature for contrast (kept for compat)")
    parser.add_argument("--lap_weight", type=float, default=0.01, help="Laplacian regularization weight")

    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=3e-3, help="Weight decay (L2)")
    parser.add_argument("--label_smoothing", type=float, default=0.2, help="Label smoothing (0=off)")

    parser.add_argument("--t_0", type=int, default=50, help="Cosine warm restart T_0")
    parser.add_argument("--t_mult", type=int, default=2, help="Cosine warm restart T_mult")

    parser.add_argument("--target_ratio", type=float, default=0.2, help="Target diffusion/classification loss ratio")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Sigma_data for diffusion")
    parser.add_argument("--ema_alpha", type=float, default=0.999, help="EMA alpha for internal states")

    parser.add_argument(
        "--adaptive_noise",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable uncertainty-guided adaptive noise",
    )
    parser.add_argument("--sigma_min", type=float, default=0.5, help="Minimum noise scale for confident nodes")
    parser.add_argument("--sigma_max", type=float, default=1.5, help="Maximum noise scale for uncertain nodes")

    parser.add_argument(
        "--diffusion_weight",
        type=float,
        default=1.0,
        help="[Deprecated] diffusion weight (dynamic weighting is used now)",
    )

    parser.add_argument(
        "--aug_type",
        type=str,
        default="edge_dropout",
        choices=["edge_dropout", "node_dropout", "subgraph_sampling", "edge_addition"],
        help="Graph augmentation type",
    )
    parser.add_argument("--aug_ratio", type=float, default=0.1, help="Graph augmentation ratio")

    parser.add_argument("--use_ema", action="store_true", default=False, help="[Deprecated] kept for compatibility")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="[Deprecated] use --ema_alpha instead")

    args = parser.parse_args()
    main(args)
