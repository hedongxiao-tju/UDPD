#!/usr/bin/env python3
"""
使用方法：
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
# 🚑【紧急修复】PyTorch 2.1 vs Transformers 兼容性补丁 (全局生效)
# 必须放在 import transformers 或 import torch_geometric 之前！
# ============================================================
import torch.utils._pytree as pytree

def safe_register_pytree_node(typ, flatten_func, unflatten_func, serialized_type_name=None):
    # 忽略新版 transformers 传进来的 serialized_type_name 参数
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
# 🆕 使用V11模型：不确定性引导的自适应噪声
from integrated_model import DualDiffusionGraphModel_V11

# 添加SentenceTransformer支持
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ 警告：未安装sentence-transformers，将使用原始特征")


class EMA:
    """指数移动平均"""
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
    对比学习损失 - 增强类内一致性，对抗过拟合
    
    核心思想：
    - 同类节点的结构表征和语义表征应该更接近
    - 异类节点应该更远离
    - 使用InfoNCE损失（对比学习标准损失）
    
    Args:
        h_struct: 结构表征 [N, hid_dim]
        h_semantic: 语义表征 [N, hid_dim]
        labels: 节点标签 [N]
        temperature: 温度参数（控制分布锐度）
        
    Returns:
        contrast_loss: 对比学习损失（标量）
    """
    # L2归一化（投影到单位球面）
    h_struct_norm = F.normalize(h_struct, p=2, dim=-1)
    h_semantic_norm = F.normalize(h_semantic, p=2, dim=-1)
    
    # 计算结构-语义相似度矩阵 [N, N]
    sim_matrix = torch.mm(h_struct_norm, h_semantic_norm.t()) / temperature
    
    # 构建正样本mask（同类为1，异类为0）
    labels_expand = labels.unsqueeze(1)  # [N, 1]
    pos_mask = (labels_expand == labels_expand.t()).float()  # [N, N]
    
    # 排除自身（对角线）
    identity_mask = torch.eye(pos_mask.size(0), device=pos_mask.device)
    pos_mask = pos_mask * (1 - identity_mask)
    
    # 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
    
    # InfoNCE损失
    # log(exp(sim_pos) / sum(exp(sim_all)))
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # 只对正样本计算损失
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_counts.squeeze()
    contrast_loss = -pos_log_prob.mean()
    
    return contrast_loss
# ==========================================
# 🆕 新增：Barlow Twins 损失函数相关代码
# ==========================================

def off_diagonal(x):
    """提取矩阵的非对角线元素"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_barlow_twins_loss(z1, z2, lambd=0.005):
    """
    Barlow Twins 损失计算: 互相关冗余削减
    Args:
        z1: 结构特征 [N, D]
        z2: 语义特征 [N, D]
        lambd: 冗余削减项的权重 (默认0.005)
    Returns:
        loss: 标量
    """
    # 1. 维度归一化 (Batch Normalization style)
    # Barlow Twins 要求沿着 Batch 维度做归一化
    
    # 为了数值稳定性，加入 eps
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
    
    N, D = z1_norm.size()

    # 2. 计算互相关矩阵 [D, D]
    # 对应公式中的 C matrix
    c = torch.mm(z1_norm.T, z2_norm) / N

    # 3. 损失计算
    # 3.1 不变性项 (Invariance): 对角线元素尽可能接近 1
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    
    # 3.2 冗余削减项 (Redundancy Reduction): 非对角线元素尽可能接近 0
    off_diag = off_diagonal(c).pow(2).sum()

    loss = on_diag + lambd * off_diag
    return loss


def compute_laplacian_loss(h, edge_index):
    """
    图拉普拉斯正则化 (Graph Laplacian Regularization)
    
    核心思想：
    - 显式强迫相邻节点的特征在潜空间中保持一致。
    - "橡皮筋理论"：相连节点拉得越远，惩罚越大。
    
    Args:
        h: 节点特征矩阵 [N, D] (建议使用 h_final)
        edge_index: 图的边索引 [2, E]
    
    Returns:
        loss: 标量
    """
    # 1. 获取源节点 (Source) 和 目标节点 (Target) 的索引
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    # 2. 查表获取对应的特征向量
    h_src = h[src_idx]  # [E, D]
    h_dst = h[dst_idx]  # [E, D]

    # 3. 计算成对欧氏距离的平方 ||h_i - h_j||^2
    # (h_src - h_dst).pow(2) -> [E, D]
    # .sum(dim=1) -> [E] (每条边的距离平方)
    squared_diff = (h_src - h_dst).pow(2).sum(dim=1)

    # 4. 求平均
    # 使用 mean() 而不是 sum() 是为了避免边数过多导致 Loss 数值过大，方便调参
    loss = squared_diff.mean()
    
    return loss


def train_epoch(model, data, optimizer, epoch, total_epochs, text_embeds, target_ratio=0.2,
                ema_h_str=None, ema_h_sem=None, uncertainty=None, contrast_weight=0.0, contrast_temp=0.5,
                label_smoothing=0.2, lap_weight=0.01): # <--- 新增 lap_weight 参数
    """
    V11训练epoch - 集成方案1 (Barlow Twins) 和 方案2 (Laplacian)
    """
    model.train()
    optimizer.zero_grad()
    
    # 1. 前向传播
    out, h_final, h_str, h_sem, diffusion_loss, ema_h_str, ema_h_sem, uncertainty = model(
        data, text_embeds,
        ema_h_str=ema_h_str,
        ema_h_sem=ema_h_sem,
        uncertainty=uncertainty,
        use_augmentation=True, # 训练时开启图增强
        epoch=epoch
    )
    
    # 2. 基础损失计算
    # 分类损失 (CrossEntropy)
    cls_loss = F.cross_entropy(
        out[data.train_mask], 
        data.y[data.train_mask],
        label_smoothing=label_smoothing
    )
    
    # 扩散损失权重 (动态调整)
    base_weight = 0.5 
    progress = epoch / total_epochs
    diffusion_weight = base_weight * (1.0 + 0.5 * progress)
    
    # ============================================================
    # 🔄 [方案1] Barlow Twins (替代 InfoNCE)
    # ============================================================
    bt_loss = torch.tensor(0.0, device=out.device)
    if contrast_weight > 0:
        # 复用 contrast_weight 参数
        # 计算结构特征(h_str)和语义特征(h_sem)的解耦与对齐
        bt_loss = compute_barlow_twins_loss(
            h_str[data.train_mask], 
            h_sem[data.train_mask],
            lambd=0.005
        )

    # ============================================================
    # 🔄 [方案2] Graph Laplacian Regularization (图平滑)
    # ============================================================
    # 对最终融合特征 h_final 施加全图平滑约束
    # 即使是无标签节点(Test/Val)的边，也能提供结构监督信号
    lap_loss = compute_laplacian_loss(h_final, data.edge_index)

    # ============================================================
    # 总损失聚合
    # ============================================================
    total_loss = cls_loss + \
                 (diffusion_weight * diffusion_loss) + \
                 (contrast_weight * bt_loss) + \
                 (lap_weight * lap_loss) # <--- 加入拉普拉斯损失
    
    total_loss.backward()
    
    # 梯度裁剪与优化
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 计算准确率
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    
    # 收集详细诊断信息
    diagnostics = {
        'grad_norm': total_norm,
        'diffusion_contribution': (diffusion_weight * diffusion_loss).item(),
        # 记录 Barlow Twins 损失
        'bt_loss': bt_loss.item(),
        # 记录 Laplacian 损失
        'lap_loss': lap_loss.item(),
        # 记录 Laplacian 对总 Loss 的贡献
        'lap_contribution': (lap_weight * lap_loss).item(),
        # 记录 Barlow Twins 对总 Loss 的贡献
        'bt_contribution': (contrast_weight * bt_loss).item()
    }
        # ===== 额外统计信息，给日志用 =====
    with torch.no_grad():
        diagnostics.update({
            'h_str_mean': h_str.mean().item(),
            'h_str_std':  h_str.std().item(),
            'h_sem_mean': h_sem.mean().item(),
            'h_sem_std':  h_sem.std().item(),
            'h_final_mean': h_final.mean().item(),
            'h_final_std':  h_final.std().item(),
            'out_mean': out.mean().item(),
            'out_std':  out.std().item(),
            'out_min':  out.min().item(),
            'out_max':  out.max().item(),
            'ema_available': (ema_h_str is not None) and (ema_h_sem is not None),
        })

    
    # 在返回值中增加 lap_loss 的数值，方便 main 函数打印
    # 返回顺序: total, diff, cls, acc, diff_w, adjust, ema_str, ema_sem, uncert, diag, bt, lap
    return (total_loss.item(), diffusion_loss.item(), cls_loss.item(), train_acc,
            diffusion_weight, 1.0, ema_h_str, ema_h_sem, uncertainty, diagnostics, 
            bt_loss.item(), lap_loss.item())

@torch.no_grad()
def evaluate(model, data, text_embeds, mask, 
             ema_h_str=None, ema_h_sem=None, uncertainty=None):
    """V10模型评估 - 完全复刻93%版本"""
    model.eval()
    
    out, h_final, h_str, h_sem, diffusion_loss, ema_h_str, ema_h_sem, uncertainty = model(
        data, text_embeds,
        ema_h_str=ema_h_str,
        ema_h_sem=ema_h_sem,
        uncertainty=uncertainty,
        use_augmentation=False
    )
    
    pred = out.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, precision, recall, f1, ema_h_str, ema_h_sem, uncertainty


def get_text_embeddings(texts, device='cuda', cache_file='text_embeddings_cache_cora_v10.pt'):
    """生成或加载文本嵌入"""
    # 检查缓存
    if os.path.exists(cache_file):
        print(f"📂 从缓存加载文本嵌入: {cache_file}")
        text_embeds = torch.load(cache_file, map_location='cpu')
        return text_embeds.to(device)
    
    if not SBERT_AVAILABLE:
        print("❌ 无法生成文本嵌入：sentence-transformers未安装")
        return None
    
    print(f"🔄 正在加载 SentenceTransformer 模型...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("🔄 正在生成文本嵌入...")
    text_embeds = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=True, device=device)
    text_embeds = text_embeds.to(torch.float32)
    
    # 保存缓存
    print(f"💾 保存文本嵌入到缓存: {cache_file}")
    torch.save(text_embeds.cpu(), cache_file)
    
    return text_embeds.to(device)


def create_balanced_split(data, seed=42):
    """创建类别平衡的60/20/20划分"""
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
        n_train = int(0.6 * n_class)  # 60%训练
        n_val = int(0.2 * n_class)    # 20%验证
        # 剩余20%测试
        
        train_mask[class_nodes[:n_train]] = True
        val_mask[class_nodes[n_train:n_train+n_val]] = True
        test_mask[class_nodes[n_train+n_val:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def main(args):
    # 设置种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("🚀 V11 [全SBERT升级版] 训练 - Cora数据集")
    print("="*80)
    
    # 🆕 V11核心信息展示
    print(f"🧠 V11核心创新: 不确定性引导的自适应噪声")
    print(f"   ✓ 自适应噪声: {'启用' if args.adaptive_noise else '禁用 (V10兼容模式)'}")
    if args.adaptive_noise:
        print(f"   ✓ 噪声范围: σ_min={args.sigma_min}, σ_max={args.sigma_max}")
        print(f"   ✓ 策略: 困难节点→大噪声，简单节点→小噪声")
    
    # 🆕 损失函数配置信息
    print(f"\n📊 损失函数配置:")
    print(f"   ✓ 分类损失: CrossEntropy (标签平滑: {args.label_smoothing})")
    print(f"   ✓ 扩散损失: 动态权重 (基础0.5 → 进度调整)")
    if args.contrast_weight > 0:
        print(f"   ✓ Barlow Twins: 权重={args.contrast_weight}, 温度={args.contrast_temp}")
        # 🆕 添加lap_weight参数显示（注意这里需要添加参数）
        lap_weight = getattr(args, 'lap_weight', 0.01)  # 默认值
        print(f"   ✓ Laplacian正则: 权重={lap_weight} (图平滑约束)")
    else:
        print(f"   ✓ Barlow Twins: 禁用 (权重=0)")
        print(f"   ✓ Laplacian正则: 默认启用")
    
    # 🆕 模型架构详细信息
    print(f"\n🔧 V11模型架构:")
    print(f"   ✓ 隐藏维度: {args.hid_dim} (V10最佳: 256)")
    print(f"   ✓ 扩散步数: {args.diffusion_steps} (V10最佳: 5)")
    print(f"   ✓ 注意力头数: {args.num_heads} (V10最佳: 8)")
    print(f"   ✓ Dropout: {args.dropout} (V10最佳: 0.4)")
    print(f"   ✓ EMA平滑: α={args.ema_alpha}")
    print(f"   ✓ 数据标准差: σ_data={args.sigma_data}")
    
    # 🆕 训练配置信息
    print(f"\n⚙️ 训练配置:")
    print(f"   ✓ 学习率: {args.lr} (V10最佳: 0.0003)")
    print(f"   ✓ 权重衰减: {args.weight_decay} (V10最佳: 3e-3)")
    print(f"   ✓ 余弦退火: T_0={args.t_0}, T_mult={args.t_mult}")
    print(f"   ✓ 早停耐心: {args.patience} epochs")
    
    # 🆕 图增强配置
    print(f"\n🔄 图增强配置:")
    print(f"   ✓ 增强策略: {args.aug_type}")
    print(f"   ✓ 增强比例: {args.aug_ratio}")
    
    # 🆕 性能目标预告
    print(f"\n🎯 性能预期:")
    print(f"   ✓ V10基线: 93.36% (标准划分)")
    if args.adaptive_noise:
        if args.sigma_max <= 1.5:
            print(f"   ✓ V11保守目标: 93.5-93.8% (+0.1~0.4%)")
        else:
            print(f"   ✓ V11激进目标: 93.6-94.0% (+0.2~0.6%)")
    else:
        print(f"   ✓ V11兼容模式: 93.3-93.4% (接近V10)")
    
    # 1. 加载数据和文本嵌入
    print(f"\n📂 加载Cora数据集...")
    data, texts = get_raw_text_cora(use_text=True, seed=args.seed)

    # --- 步骤 A: 准备结构路特征 (GCN 输入) ---
    # 策略：使用 Full SBERT (标题+摘要)，保留最大信息量用于聚合
    print("🔄 [结构路] 准备 Full SBERT 特征 (Title + Abstract)...")
    # 注意：使用 cache_file 区分文件名
    full_sbert = get_text_embeddings(texts, device=device, cache_file='text_embeddings_cache_cora_v10.pt')
    
    # ============================================================
    # 🚀 [升级] 加载 LLM 提纯后的特征
    # ============================================================
    llm_feat_path = "cora_llm_keywords.pt"
    
    if os.path.exists(llm_feat_path):
        print(f"🔥 [语义路] 准备 LLM 提纯特征 ({llm_feat_path})...")
        llm_keywords = torch.load(llm_feat_path, map_location=device)
    else:
        raise RuntimeError("❌ 找不到 LLM 关键词文件！请先运行预处理脚本。")
        
    # --- 步骤 C: 分配特征 ---
    
    # 1. 结构路 -> data.x (给 GCN 用)
    data.x = full_sbert.clone()
    current_in_dim = full_sbert.shape[1] # 384
    
    # 2. 语义路 -> text_embeds (给 Teacher 用)
    text_embeds = llm_keywords.clone()

    print(f"✅ 特征分配完成 (非对称架构):")
    print(f"   GCN 输入 (data.x): Full SBERT {data.x.shape} (富含细节)")
    print(f"   Teacher 输入 (text_embeds): LLM Keywords {text_embeds.shape} (纯净语义)")
    print("   👉 目标: 利用 LLM 的精华语义来'提纯' GCN 聚合的全文信息")
    # ============================================================
    
    num_classes = int(data.y.max().item()) + 1
    
    # 数据划分
    if args.split == 'balanced':
        data = create_balanced_split(data, seed=args.seed)
    
    # 移动到设备
    data = data.to(device)
    text_embeds = text_embeds.to(device)

    
    # 创建模型
    print(f"\n🚀 正在初始化V11模型...")
    print(f"📊 模型参数统计:")
    model = DualDiffusionGraphModel_V11(
        in_dim=current_in_dim,      # <--- 注意这里用了新的维度 (384)
        sem_dim=text_embeds.shape[1],                # 语义路维度不变
        hid_dim=args.hid_dim,
        num_classes=num_classes,
        diffusion_steps=args.diffusion_steps,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ema_alpha=args.ema_alpha,
        sigma_data=args.sigma_data,
        adaptive_noise=args.adaptive_noise,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max
    ).to(device)
    
    # 计算并显示模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ 总参数: {total_params:,}")
    print(f"   ✓ 可训练参数: {trainable_params:,}")
    print(f"   ✓ 输入维度: 结构路={current_in_dim}, 语义路={text_embeds.shape[1]}")
    print(f"   ✓ 输出类别: {num_classes}")
    
    # ... (后面的优化器设置、训练循环代码完全不用变) ...
    # 为了完整性，下面接上原来的代码逻辑
    
    model.graph_augmentor.aug_type = args.aug_type
    model.graph_augmentor.aug_ratio = args.aug_ratio
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t_0,
        T_mult=args.t_mult,
        eta_min=args.lr * 0.01
    )


    
    print(f"\n🚀 开始V11训练 ({'自适应噪声' if args.adaptive_noise else 'V10兼容'}模式)...")
    print("="*80)
    
    # 🆕 训练前的环境检查
    print(f"🔍 训练环境检查:")
    print(f"   ✓ 设备: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(args.gpu_id)
        gpu_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3
        print(f"   ✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    print(f"   ✓ 数据集: Cora ({data.x.shape[0]}节点, {data.edge_index.shape[1]}边)")
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()
    print(f"   ✓ 划分: 训练={train_count}, 验证={val_count}, 测试={test_count}")
    print("="*80)
    
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    # 记录最佳模型
    best_model_state = None
    
    # V10关键：初始化EMA状态（跨epoch传递）
    ema_h_str, ema_h_sem, uncertainty = None, None, None
    
    for epoch in range(1, args.epochs + 1):
        # 训练 - 传递并更新EMA状态
        lap_weight = getattr(args, 'lap_weight', 0.01)  # 获取lap_weight参数
        train_loss, diffusion_loss, cls_loss, train_acc, diffusion_weight_used, adjust_factor, ema_h_str, ema_h_sem, uncertainty, diagnostics, bt_loss_val, lap_loss_val = train_epoch(
            model, data, optimizer, epoch, args.epochs, text_embeds, 
            target_ratio=args.target_ratio,
            ema_h_str=ema_h_str, ema_h_sem=ema_h_sem, uncertainty=uncertainty,
            contrast_weight=args.contrast_weight,
            contrast_temp=args.contrast_temp,
            label_smoothing=args.label_smoothing,
            lap_weight=lap_weight  # 🆕 传递lap_weight参数
        )
        
        # 评估 - 使用相同的EMA状态
        val_acc, val_p, val_r, val_f1, ema_h_str, ema_h_sem, uncertainty = evaluate(
            model, data, text_embeds, data.val_mask,
            ema_h_str, ema_h_sem, uncertainty
        )
        test_acc, test_p, test_r, test_f1, ema_h_str, ema_h_sem, uncertainty = evaluate(
            model, data, text_embeds, data.test_mask,
            ema_h_str, ema_h_sem, uncertainty
        )
        
        # 学习率调度（93%版本）
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            # 深拷贝模型状态
            best_model_state = copy.deepcopy(model.state_dict())
            # 保存到磁盘（包含EMA状态）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'ema_h_str': ema_h_str,
                'ema_h_sem': ema_h_sem,
                'uncertainty': uncertainty
            }, 'best_cora_v10_optimized.pt')
        else:
            patience_counter += 1
        
        # 输出进度（简化版）
        if epoch % 10 == 0 or epoch == 1:
            diffusion_ratio = diffusion_loss / (cls_loss + 1e-8)
            diffusion_contribution = diffusion_weight_used * diffusion_loss
            progress_percent = epoch / args.epochs * 100
            
            # 判断训练状态
            if train_acc < 0.95:
                state_emoji = "🌱"
                state_msg = "学习中"
            elif train_acc >= 0.99:
                state_emoji = "⚠️"
                state_msg = "高训练准确率"
            else:
                state_emoji = "✅"
                state_msg = "良好"
            
            # 标题显示V11信息
            if args.adaptive_noise:
                print(f"\nEpoch {epoch:03d}/{args.epochs} - V11 Cora (自适应噪声):")
            else:
                print(f"\nEpoch {epoch:03d}/{args.epochs} - V11 Cora (V10兼容模式):")
            
            print(f"  {state_emoji} 状态: {state_msg} (进度: {progress_percent:.1f}%)")
            
            # 🆕 更新的损失信息显示
            if args.contrast_weight > 0:
                bt_contribution = args.contrast_weight * bt_loss_val
                lap_contribution = lap_weight * lap_loss_val
                print(f"  总损失: {train_loss:.4f} | 分类: {cls_loss:.4f} | 扩散: {diffusion_loss:.4f}")
                print(f"  正则化: BT={bt_loss_val:.4f}(×{args.contrast_weight:.3f}) | Lap={lap_loss_val:.4f}(×{lap_weight:.3f})")
                print(f"  学习率: {current_lr:.2e} | 扩散权重: {diffusion_weight_used:.3f}")
                print(f"  贡献比例: 扩散={diffusion_contribution/cls_loss*100:.0f}%, BT={bt_contribution/cls_loss*100:.0f}%, Lap={lap_contribution/cls_loss*100:.0f}%")
                # Barlow Twins损失监控
                if bt_loss_val > 1.0:
                    print(f"  ⚠️  BT损失过大({bt_loss_val:.2f})，建议降低contrast_weight")
            else:
                print(f"  总损失: {train_loss:.4f} | 分类: {cls_loss:.4f} | 扩散: {diffusion_loss:.4f} | Lap: {lap_loss_val:.4f}")
                print(f"  学习率: {current_lr:.2e} | 扩散权重: 0.5×(1+0.5×{progress_percent/100:.2f}) = {diffusion_weight_used:.3f}")
                print(f"  贡献: 扩散={diffusion_contribution/cls_loss:.2f}x, Lap={lap_weight * lap_loss_val/cls_loss:.2f}x")
            
            # 🆕 V11自适应噪声状态监控（详细统计）
            if args.adaptive_noise and uncertainty is not None and hasattr(uncertainty, 'mean'):
                unc_mean = uncertainty.mean().item()
                unc_std = uncertainty.std().item()
                unc_min = uncertainty.min().item()
                unc_max = uncertainty.max().item()
                print(f"  🆕 不确定性: 平均={unc_mean:.3f}±{unc_std:.3f}, 范围=[{unc_min:.3f}-{unc_max:.3f}]")
            
            print(f"  准确率: 训练={train_acc:.4f} | 验证={val_acc:.4f} | 测试={test_acc:.4f}")
            print(f"  验证指标: P={val_p:.3f}, R={val_r:.3f}, F1={val_f1:.3f}")
            print(f"  当前最佳: {best_val_acc:.4f} @ Epoch {best_epoch}")
            print(f"  🧠 损失比: diff/cls={diffusion_ratio:.1f}, 实际贡献={diffusion_contribution/cls_loss:.2f}x")
            
            # 🆕 更新诊断信息显示（每30个epoch详细显示一次）
            if epoch % 30 == 0 or epoch == 1:
                print(f"  🔍 诊断信息:")
                print(f"     特征统计: h_str={diagnostics['h_str_mean']:.3f}±{diagnostics['h_str_std']:.3f}, "
                      f"h_sem={diagnostics['h_sem_mean']:.3f}±{diagnostics['h_sem_std']:.3f}, "
                      f"h_final={diagnostics['h_final_mean']:.3f}±{diagnostics['h_final_std']:.3f}")
                print(f"     输出统计: mean={diagnostics['out_mean']:.3f}±{diagnostics['out_std']:.3f}, "
                      f"range=[{diagnostics['out_min']:.3f}, {diagnostics['out_max']:.3f}]")
                print(f"     梯度范数: {diagnostics['grad_norm']:.3f}, EMA可用: {diagnostics['ema_available']}")
                # 🆕 添加V11特有的诊断信息
                if args.adaptive_noise:
                    print(f"     🆕 V11状态: BT损失={diagnostics['bt_loss']:.4f}, Lap损失={diagnostics['lap_loss']:.4f}")
                    print(f"     🆕 正则化贡献: BT={diagnostics.get('bt_contribution', 0):.4f}, Lap={diagnostics.get('lap_contribution', 0):.4f}")
        
        # 特殊里程碑
        if val_acc > best_val_acc - 0.001:
            if test_acc >= 0.93:
                print(f"  🎯 达到93%目标! 测试准确率: {test_acc:.4f}")
            if test_acc >= 0.95:
                print(f"  🏆 突破95%! 测试准确率: {test_acc:.4f}")
        
        # 早停（但设置较大的patience）
        if patience_counter >= args.patience:
            print(f"\n⏹️ 早停 @ Epoch {epoch}")
            break
    
    # 加载最佳模型进行最终测试（包括EMA状态）
    print(f"\n📊 加载最佳模型进行最终测试...")
    checkpoint = torch.load('best_cora_v10_optimized.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_h_str = checkpoint['ema_h_str']
    ema_h_sem = checkpoint['ema_h_sem']
    uncertainty = checkpoint['uncertainty']
    
    print(f"✅ 已加载Epoch {checkpoint['epoch']}的最佳模型（验证准确率: {checkpoint['val_acc']:.4f}）")
    
    # 最终测试评估（使用EMA状态）
    final_test_acc, final_test_p, final_test_r, final_test_f1, _, _, _ = evaluate(
        model, data, text_embeds, data.test_mask,
        ema_h_str, ema_h_sem, uncertainty
    )
    
    # 混淆矩阵
    model.eval()
    with torch.no_grad():
        out, _, _, _, _, _, _, _ = model(
            data, text_embeds,
            ema_h_str=ema_h_str,
            ema_h_sem=ema_h_sem,
            uncertainty=uncertainty,
            use_augmentation=False
        )
        pred = out[data.test_mask].argmax(dim=1)
        cm = confusion_matrix(data.y[data.test_mask].cpu(), pred.cpu())
    
    print("\n" + "="*60)
    print(f"🏆 V11 Cora 最终结果 ({args.split.upper()}划分):")
    print("="*60)
    print(f"测试准确率: {final_test_acc:.4f}")
    print(f"精确率: {final_test_p:.4f} | 召回率: {final_test_r:.4f} | F1: {final_test_f1:.4f}")
    print(f"最佳验证: {best_val_acc:.4f} @ Epoch {best_epoch}")
    
    # 🆕 V11详细配置总结
    print(f"\n🆕 V11配置总结:")
    if args.adaptive_noise:
        print(f"  ✓ 自适应噪声: 启用 (σ范围: {args.sigma_min}-{args.sigma_max})")
        print(f"  ✓ 策略: 不确定性引导 (困难节点→大噪声, 简单节点→小噪声)")
    else:
        print(f"  ✓ 自适应噪声: 禁用 (V10兼容模式)")
    
    print(f"  ✓ 损失配置: 分类+扩散+BT({args.contrast_weight})+Lap({getattr(args, 'lap_weight', 0.01)})")
    print(f"  ✓ 优化配置: AdamW(lr={args.lr}, wd={args.weight_decay}) + 余弦退火")
    print(f"  ✓ 正则配置: Dropout({args.dropout}) + 标签平滑({args.label_smoothing})")
    print(f"  ✓ 图增强: {args.aug_type}({args.aug_ratio})")
    
    print(f"\n📈 性能对比分析:")
    if args.split == 'standard':
        print(f"  📊 V10基线 (标准划分): 93.36%")
        print(f"  🆕 V11结果 ({'自适应' if args.adaptive_noise else '兼容'}): {final_test_acc:.4f} ({final_test_acc:.2%})")
        improvement = (final_test_acc - 0.9336)*100
        if improvement > 0.2:
            print(f"  🎉 显著提升: +{improvement:.2f}% ✅✅")
        elif improvement > 0:
            print(f"  ✅ 轻微提升: +{improvement:.2f}% ✅")
        elif improvement > -0.2:
            print(f"  ⚖️ 基本持平: {improvement:.2f}%")
        else:
            print(f"  ⚠️ 性能下降: {improvement:.2f}%")
        
        # 🆕 V11配置效果评估
        if args.adaptive_noise:
            noise_range = args.sigma_max - args.sigma_min
            if noise_range >= 1.0:
                print(f"  🧠 噪声策略: 激进范围({noise_range:.1f}) - 适合复杂数据")
            else:
                print(f"  🧠 噪声策略: 保守范围({noise_range:.1f}) - 适合稳定训练")
    else:
        print(f"  📊 参考V10 (标准划分): 93.36%")
        print(f"  🆕 V11结果 (平衡划分): {final_test_acc:.4f} ({final_test_acc:.2%})")
        print(f"  📝 注意: 不同划分方式，性能不直接可比")
    print(f"\n混淆矩阵:\n{cm}")
    print("="*60)
    
    # 分析每个类别的性能
    print("\n📊 类别性能分析:")
    for i in range(num_classes):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  类别{i}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V11训练脚本 - Cora数据集（不确定性引导自适应噪声）')
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数（93%版本用400）')
    parser.add_argument('--patience', type=int, default=150, help='早停patience（最佳：150）')
    parser.add_argument('--split', type=str, default='balanced')
    
    # V11模型架构参数（基于V10最佳配置）
    parser.add_argument('--hid_dim', type=int, default=256, 
                        help='隐藏层维度（V10最佳：256，影响模型容量）')
    parser.add_argument('--diffusion_steps', type=int, default=5, 
                        help='扩散步数（V10最佳：5，影响去噪精度）')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='注意力头数（V10最佳：8，影响特征融合）')
    parser.add_argument('--dropout', type=float, default=0.4, 
                        help='Dropout比例（V10最佳：0.4，防止过拟合）')
    
    # 🆕 V11正则化参数（新增损失函数）
    parser.add_argument('--contrast_weight', type=float, default=0.0, 
                        help='Barlow Twins损失权重（0=禁用，推荐0.1-0.5）')
    parser.add_argument('--contrast_temp', type=float, default=0.5,
                        help='Barlow Twins温度参数（控制分布锐度，推荐0.5-1.0）')
    parser.add_argument('--lap_weight', type=float, default=0.01,
                        help='🆕 图拉普拉斯正则化权重（默认0.01，推荐0.005-0.05）')
    
    # V11优化参数（基于V10最佳配置）
    parser.add_argument('--lr', type=float, default=0.0003, 
                        help='学习率（V10最佳：0.0003，平衡收敛速度与稳定性）')
    parser.add_argument('--weight_decay', type=float, default=3e-3, 
                        help='L2正则化（V10最佳：3e-3，防止权重过大）')
    parser.add_argument('--label_smoothing', type=float, default=0.2, 
                        help='标签平滑（V10最佳：0.2，提升泛化能力，0=禁用）')
    
    # V11学习率调度器参数
    parser.add_argument('--t_0', type=int, default=50, 
                        help='余弦退火初始周期（影响学习率变化频率）')
    parser.add_argument('--t_mult', type=int, default=2, 
                        help='余弦退火周期倍数（每次重启后周期翻倍）')
    
    # V11扩散系统参数
    parser.add_argument('--target_ratio', type=float, default=0.2, 
                        help='目标扩散/分类损失比（动态权重调节目标，推荐0.1-0.5）')
    parser.add_argument('--sigma_data', type=float, default=0.5, 
                        help='扩散过程数据标准差（影响噪声尺度）')
    parser.add_argument('--ema_alpha', type=float, default=0.999, 
                        help='指数移动平均平滑系数（影响特征稳定性）')
    
    # 🆕 V11自适应噪声参数（核心创新）
    parser.add_argument('--adaptive_noise', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='🆕 启用不确定性引导的自适应噪声（V11核心创新，默认True）')
    parser.add_argument('--sigma_min', type=float, default=0.5,
                        help='🆕 确定节点的最小噪声系数（推荐0.3-0.7，默认0.5）')
    parser.add_argument('--sigma_max', type=float, default=1.5,
                        help='🆕 不确定节点的最大噪声系数（推荐1.0-2.0，默认1.5）')
    
    # ⚠️ 废弃参数（保留向后兼容）
    parser.add_argument('--diffusion_weight', type=float, default=1.0, 
                        help='⚠️ [废弃] 现在使用动态自适应权重（0.5→0.75渐进）')
    
    # V11图增强参数
    parser.add_argument('--aug_type', type=str, default='edge_dropout',
                        choices=['edge_dropout', 'node_dropout', 'subgraph_sampling', 'edge_addition'],
                        help='图增强策略（训练时随机扰动图结构，提升泛化）')
    parser.add_argument('--aug_ratio', type=float, default=0.1, 
                        help='图增强比例（扰动强度，推荐0.05-0.2）')
    
    # ⚠️ 废弃的EMA参数（V11使用内置的EMA状态）
    parser.add_argument('--use_ema', action='store_true', default=False, 
                        help='⚠️ [废弃] V11使用内置EMA状态管理，无需手动设置')
    parser.add_argument('--ema_decay', type=float, default=0.999, 
                        help='⚠️ [废弃] 请使用--ema_alpha参数')
    
    args = parser.parse_args()
    main(args)
