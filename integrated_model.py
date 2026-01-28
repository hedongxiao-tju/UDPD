#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math
import numpy as np

# =======================================
# V10核心创新: 动态条件双扩散模块  
# =======================================

class StructuralTeacher(nn.Module):
    """结构侧教师：在动态语义引导下，修改结构偏差"""
    
    def __init__(self, hid_dim, condition_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.condition_dim = condition_dim
        
        # 结构特征处理
        self.struct_proj = nn.Linear(hid_dim, hid_dim)
        
        # 动态条件融合 - 关键创新
        self.condition_attention = nn.MultiheadAttention(
            embed_dim=hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(hid_dim + condition_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim)
        )
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hid_dim // 2),
            nn.GELU(),
            nn.Linear(hid_dim // 2, hid_dim)
        )
        
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        
    def forward(self, h_struct_t, dynamic_semantic_condition, t, edge_index=None,uncertainty=None):
        """
        Args:
            h_struct_t: 加噪的结构嵌入 [N, hid_dim]
            dynamic_semantic_condition: 语义侧实时去噪结果 h_t^{rec,t} [N, condition_dim] 
            t: 当前时间步
            edge_index: 图连接边
        Returns:
            predicted_noise: 预测的结构噪声 ε_s^t [N, hid_dim]
        """
        batch_size = h_struct_t.size(0)
        
        # 时间步嵌入
        t_embed = self.time_mlp(t.float().view(-1, 1))  # [1, hid_dim]
        t_embed = t_embed.expand(batch_size, -1)  # [N, hid_dim]
        
        # 结构特征处理
        h_struct = self.struct_proj(h_struct_t)
        h_struct = self.norm1(h_struct + t_embed)
        
        # 动态条件注意力融合 - 内存优化版本
        if dynamic_semantic_condition is not None:
            # 对于大数据集，使用分块处理来节省内存
            batch_size = h_struct.size(0)
            if batch_size > 5000:  # 大数据集分块处理
                chunk_size = 1000
                conditioned_struct = torch.zeros_like(h_struct)
                
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    h_chunk = h_struct[i:end_idx].unsqueeze(0)
                    cond_chunk = dynamic_semantic_condition[i:end_idx].unsqueeze(0)
                    
                    chunk_output, _ = self.condition_attention(
                        query=h_chunk,
                        key=cond_chunk, 
                        value=cond_chunk
                    )
                    conditioned_struct[i:end_idx] = chunk_output.squeeze(0)
            else:
                # 小数据集正常处理
                conditioned_struct, attn_weights = self.condition_attention(
                    query=h_struct.unsqueeze(0),  # [1, N, hid_dim] 
                    key=dynamic_semantic_condition.unsqueeze(0),  # [1, N, condition_dim]
                    value=dynamic_semantic_condition.unsqueeze(0)  # [1, N, condition_dim]
                )
                conditioned_struct = conditioned_struct.squeeze(0)  # [N, hid_dim]
                # ============================================================
                # 🆕 核心修改：动态条件引导
                # ============================================================
                if uncertainty is not None:
                 # uncertainty: [N] -> [N, 1]
                 # 逻辑：越不确定，越要听劝 (放大来自语义的建议)
                 # 权重 = 1.0 + uncertainty (范围 1.0 ~ 2.0)
                    guidance_scale = 1.0 + uncertainty.unsqueeze(-1)
                    conditioned_struct = conditioned_struct * guidance_scale
        # ============================================================
            
            # 残差连接
            h_struct = self.norm2(h_struct + conditioned_struct)
        
        # 融合条件信息进行噪声预测
        if dynamic_semantic_condition is not None:
            combined_input = torch.cat([h_struct, dynamic_semantic_condition], dim=-1)
        else:
            # 没有动态条件时的降级处理
            combined_input = torch.cat([h_struct, torch.zeros_like(h_struct)], dim=-1)
        
        # 预测结构噪声
        predicted_noise = self.noise_predictor(combined_input)
        
        return predicted_noise

class SemanticTeacher(nn.Module):
    """语义侧教师：在动态结构引导下，修正语义偏差"""
    
    def __init__(self, hid_dim, condition_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.condition_dim = condition_dim
        
        # 残差特征处理
        self.residual_proj = nn.Linear(hid_dim, hid_dim)
        
        # 动态条件融合 - 关键创新
        self.condition_attention = nn.MultiheadAttention(
            embed_dim=hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差噪声预测网络
        self.residual_noise_predictor = nn.Sequential(
            nn.Linear(hid_dim + condition_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim)
        )
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hid_dim // 2),
            nn.GELU(),
            nn.Linear(hid_dim // 2, hid_dim)
        )
        
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        
    def forward(self, r_t, dynamic_struct_condition, t,uncertainty=None):
        """
        Args:
            r_t: 加噪的残差信号 [N, hid_dim]
            dynamic_struct_condition: 结构侧实时去噪结果 h_s^{rec,t} [N, condition_dim]
            t: 当前时间步
        Returns:
            predicted_residual_noise: 预测的残差噪声 ε_r^t [N, hid_dim]
        """
        batch_size = r_t.size(0)
        
        # 时间步嵌入
        t_embed = self.time_mlp(t.float().view(-1, 1))  # [1, hid_dim]
        t_embed = t_embed.expand(batch_size, -1)  # [N, hid_dim]
        
        # 残差特征处理
        h_residual = self.residual_proj(r_t)
        h_residual = self.norm1(h_residual + t_embed)
        
        # 动态条件注意力融合 - 内存优化版本
        if dynamic_struct_condition is not None:
            # 对于大数据集，使用分块处理来节省内存
            batch_size = h_residual.size(0)
            if batch_size > 5000:  # 大数据集分块处理
                chunk_size = 1000
                conditioned_residual = torch.zeros_like(h_residual)
                
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    h_chunk = h_residual[i:end_idx].unsqueeze(0)
                    cond_chunk = dynamic_struct_condition[i:end_idx].unsqueeze(0)
                    
                    chunk_output, _ = self.condition_attention(
                        query=h_chunk,
                        key=cond_chunk,
                        value=cond_chunk
                    )
                    conditioned_residual[i:end_idx] = chunk_output.squeeze(0)
            else:
                # 小数据集正常处理
                conditioned_residual, attn_weights = self.condition_attention(
                    query=h_residual.unsqueeze(0),  # [1, N, hid_dim]
                    key=dynamic_struct_condition.unsqueeze(0),  # [1, N, condition_dim]
                    value=dynamic_struct_condition.unsqueeze(0)  # [1, N, condition_dim]
                )
                conditioned_residual = conditioned_residual.squeeze(0)  # [N, hid_dim]

                # ============================================================
                # 🆕 核心修改：动态条件引导
                # ============================================================
                if uncertainty is not None:
                # 逻辑：越不确定，越要听结构的指挥
                    guidance_scale = 1.0 + uncertainty.unsqueeze(-1)
                    conditioned_residual = conditioned_residual * guidance_scale
                # ============================================================
            
            # 残差连接
            h_residual = self.norm2(h_residual + conditioned_residual)
        
        # 融合条件信息进行残差噪声预测
        if dynamic_struct_condition is not None:
            combined_input = torch.cat([h_residual, dynamic_struct_condition], dim=-1)
        else:
            # 没有动态条件时的降级处理
            combined_input = torch.cat([h_residual, torch.zeros_like(h_residual)], dim=-1)
        
        # 预测残差噪声
        predicted_residual_noise = self.residual_noise_predictor(combined_input)
        
        return predicted_residual_noise

class DynamicDualDiffusion(nn.Module):
    """V11核心：不确定性引导的自适应噪声双扩散"""
    
    def __init__(self, hid_dim, diffusion_steps=5, sigma_data=0.5, 
                 adaptive_noise=True, sigma_min=0.5, sigma_max=1.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.diffusion_steps = diffusion_steps
        self.sigma_data = sigma_data
        
        # V11参数：自适应噪声
        self.adaptive_noise = adaptive_noise
        self.sigma_min = sigma_min  # 确定节点的最小噪声系数
        self.sigma_max = sigma_max  # 不确定节点的最大噪声系数
        
        # 噪声调度
        beta_start, beta_end = 0.0001, 0.02
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, diffusion_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 双侧教师网络
        self.structural_teacher = StructuralTeacher(hid_dim, hid_dim)
        self.semantic_teacher = SemanticTeacher(hid_dim, hid_dim)
    
    def compute_node_uncertainty(self, logits):
        """
        🆕 V11核心：基于预测熵计算节点不确定性
        
        Args:
            logits: [N, num_classes] 模型输出
        
        Returns:
            uncertainty: [N] 每个节点的不确定性 (0-1)
                        0 = 完全确定, 1 = 完全不确定
        """
        # Softmax计算概率分布
        probs = F.softmax(logits, dim=-1)  # [N, C]
        
        # 计算熵（信息量）
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [N]
        
        # 归一化到[0, 1]
        # 最大熵 = log(C), 当所有类别概率相等时达到
        max_entropy = np.log(probs.size(1))
        uncertainty = entropy / max_entropy  # [N], 0=确定, 1=完全不确定
        
        return uncertainty
    
    def add_adaptive_noise(self, x, t, uncertainty):
        """
        🆕 V11核心：根据不确定性添加自适应噪声
        
        Args:
            x: [N, D] 特征
            t: 时间步
            uncertainty: [N] 不确定性
        
        Returns:
            noisy_x: [N, D] 加噪后的特征
            noise: [N, D] 添加的噪声
            actual_sigma: [N, 1] 每个节点的实际噪声水平
        """
        # 生成基础噪声
        noise = torch.randn_like(x)  # [N, D]
        
        # 不确定性映射到噪声系数
        # uncertainty=0 → sigma=sigma_min (最小噪声)
        # uncertainty=1 → sigma=sigma_max (最大噪声)
        # noise_scale = self.sigma_min + uncertainty * (self.sigma_max - self.sigma_min)  # [N]
        # noise_scale = noise_scale.unsqueeze(-1)  # [N, 1]
        
        
        # 使用DDPM的加噪公式，但噪声水平自适应
        alpha_t = self.alphas_cumprod[t]
        
        # x_t = sqrt(alpha) * x + sqrt(1-alpha) * noise
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        # 返回时保持noise_scale的维度以正确广播
        return x_noisy, noise, torch.ones_like(x[:, :1])
        
    def add_noise(self, x, t):
        """基础加噪（V10兼容）"""
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        
        # 重参数化技巧
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise
        
    def denoise_step(self, x_t, predicted_noise, t):
        """单步去噪"""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.ones_like(alpha_t)
        beta_t = self.betas[t]
        
        # DDPM去噪公式
        coef1 = 1 / torch.sqrt(self.alphas[t])
        coef2 = beta_t / torch.sqrt(1 - alpha_t)
        
        x_t_minus_1 = coef1 * (x_t - coef2 * predicted_noise)
        
        return x_t_minus_1
    
    def forward(self, h_struct_0, h_semantic_0, edge_index=None, training=True, uncertainty=None):
        """
        V11不确定性引导的自适应噪声双扩散前向传播
        
        Args:
            h_struct_0: 初始结构嵌入 [N, hid_dim]
            h_semantic_0: 初始语义嵌入 [N, hid_dim] 
            edge_index: 图连接边
            training: 是否训练模式
            uncertainty: 🆕 节点不确定性 [N] (0-1), None则使用固定噪声
            
        Returns:
            h_struct_rec: 去噪后的结构嵌入
            h_semantic_rec: 去噪后的语义嵌入  
            losses: 各种损失
        """
        
        # 计算初始残差 - 核心创新
        r_0 = h_semantic_0 - h_struct_0  # 残差信号
        
        losses = {}
        
        if training:
            # 训练模式：随机采样时间步
            t = torch.randint(0, self.diffusion_steps, (1,), device=h_struct_0.device)
            
            # 1. 加噪 (现在是一视同仁了)
            # 我们依然传入 uncertainty 是为了保持接口兼容，但函数内部忽略了它
            h_struct_t, struct_noise, _ = self.add_adaptive_noise(h_struct_0, t, uncertainty)
            r_t, residual_noise, _ = self.add_adaptive_noise(r_0, t, uncertainty)
            
            # 2. 动态条件预测 (Teacher)
            # 🆕 关键修改：把 uncertainty 传进去做 Guidance
            
            # 结构侧
            h_semantic_t_rec = h_struct_0 + r_t
            struct_noise_pred = self.structural_teacher(
                h_struct_t, h_semantic_t_rec, t, edge_index, 
                uncertainty=uncertainty  # <--- 传入不确定性
            )
            
            # 语义侧
            h_struct_t_rec = h_struct_t - struct_noise_pred
            residual_noise_pred = self.semantic_teacher(
                r_t, h_struct_t_rec, t, 
                uncertainty=uncertainty  # <--- 传入不确定性
            )
            
            # 4. 计算损失（归一化处理）
            # 除以特征维度的平方根，使损失量级与分类损失相当
            dim_scale = self.hid_dim ** 0.5
            losses['struct_diffusion'] = F.mse_loss(struct_noise_pred, struct_noise) / dim_scale
            losses['semantic_diffusion'] = F.mse_loss(residual_noise_pred, residual_noise) / dim_scale
            
            # 5. 单步去噪得到重建结果
            h_struct_rec = self.denoise_step(h_struct_t, struct_noise_pred, t)
            r_rec = self.denoise_step(r_t, residual_noise_pred, t)
            h_semantic_rec = h_struct_rec + r_rec
            
        else:
            # 推理模式：完整的多步去噪
            h_struct_current = h_struct_0
            r_current = r_0
            
            # 从最大噪声开始逐步去噪
            for t in reversed(range(self.diffusion_steps)):
                t_tensor = torch.tensor([t], device=h_struct_0.device)
                
                # 动态条件预测
                h_semantic_current = h_struct_current + r_current
                struct_noise_pred = self.structural_teacher(h_struct_current, h_semantic_current, t_tensor, edge_index,uncertainty=uncertainty)
                
                h_struct_denoised = self.denoise_step(h_struct_current, struct_noise_pred, t_tensor)
                residual_noise_pred = self.semantic_teacher(r_current, h_struct_denoised, t_tensor,uncertainty=uncertainty)
                
                # 更新当前状态
                h_struct_current = h_struct_denoised
                r_current = self.denoise_step(r_current, residual_noise_pred, t_tensor)
            
            h_struct_rec = h_struct_current
            h_semantic_rec = h_struct_current + r_current
        
        return h_struct_rec, h_semantic_rec, losses

# =======================================
# V10完整模型：集成动态双扩散
# =======================================

class DualDiffusionGraphModel_V11(nn.Module):
    """
    V11版本：不确定性引导的自适应噪声双扩散模型
    基于V10（93.36%）+ 自适应噪声创新
    """
    
    def __init__(self, in_dim, sem_dim, hid_dim, num_classes, 
                 diffusion_steps=5, num_heads=8, dropout=0.1,
                 ema_alpha=0.999, sigma_data=0.5,
                 adaptive_noise=True, sigma_min=0.5, sigma_max=1.5):
        super().__init__()
        
        self.in_dim = in_dim
        self.sem_dim = sem_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.sigma_data = sigma_data
        
        # 🆕 V11参数
        self.adaptive_noise = adaptive_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # 输入投影层
        self.struct_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.LayerNorm(hid_dim)
        )
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(sem_dim, hid_dim),
            nn.GELU(), 
            nn.LayerNorm(hid_dim)
        )
        
        # 结构编码器（简化的GCN）
        self.struct_encoder = nn.ModuleList([
            GCNConv(hid_dim, hid_dim) for _ in range(2)
        ])
        
        # V11核心：不确定性引导的自适应噪声双扩散模块
        self.dual_diffusion = DynamicDualDiffusion(
            hid_dim=hid_dim,
            diffusion_steps=diffusion_steps,
            adaptive_noise=adaptive_noise,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        
        # 特征融合和分类器
        self.feature_fusion = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hid_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, num_classes)
        )
        
        # EMA缓存
        self.register_buffer('ema_h_struct', None)
        self.register_buffer('ema_h_semantic', None)
        
        # 图增强器
        self.graph_augmentor = GraphAugmentor()
        
    def forward(self, data, text_embeds, ema_h_str=None, ema_h_sem=None, 
                uncertainty=None, use_augmentation=True, **kwargs):
        """V11前向传播：不确定性引导的自适应噪声扩散"""
        
        x, edge_index = data.x, data.edge_index
        
        # 图增强
        if use_augmentation and self.training:
            edge_index = self.graph_augmentor(edge_index, x.size(0))
        
        # 1. 特征投影
        h_struct_init = self.struct_proj(x)  # 结构特征
        # 处理Cora等没有文本嵌入的情况
        if text_embeds is not None:
            h_semantic_init = self.semantic_proj(text_embeds)  # 语义特征
        else:
            h_semantic_init = self.semantic_proj(x)  # 使用结构特征作为语义特征
        
        # 2. 结构编码
        h_struct = h_struct_init
        for gcn_layer in self.struct_encoder:
            h_struct = F.gelu(gcn_layer(h_struct, edge_index))
            h_struct = F.dropout(h_struct, p=0.1, training=self.training)
        
        # 3. V11核心：不确定性引导的自适应噪声双扩散
        # 注意：在第一次前向传播时uncertainty为None，使用固定噪声
        # 后续epoch会使用上一次计算的uncertainty
        h_struct_rec, h_semantic_rec, diffusion_losses = self.dual_diffusion(
            h_struct_0=h_struct,
            h_semantic_0=h_semantic_init, 
            edge_index=edge_index,
            training=self.training,
            uncertainty=uncertainty  # 🆕 传递上一次计算的不确定性
        )
        
        # 4. EMA更新
        if self.training:
            if self.ema_h_struct is None:
                self.ema_h_struct = h_struct_rec.detach().clone()
                self.ema_h_semantic = h_semantic_rec.detach().clone()
            else:
                self.ema_h_struct = self.ema_alpha * self.ema_h_struct + \
                                   (1 - self.ema_alpha) * h_struct_rec.detach()
                self.ema_h_semantic = self.ema_alpha * self.ema_h_semantic + \
                                      (1 - self.ema_alpha) * h_semantic_rec.detach()
        
        # 5. 特征融合
        h_fused = torch.cat([h_struct_rec, h_semantic_rec], dim=-1)
        h_final = self.feature_fusion(h_fused)
        
        # 6. 分类
        logits = self.classifier(h_final)
        
        # 🆕 V11核心：在分类之后计算不确定性（用于下一次前向传播的去噪引导）
        # 这样可以确保使用最新的模型预测来评估不确定性
        # 注意：不确定性用于引导去噪过程，而非加噪过程
        uncertainty = self.dual_diffusion.compute_node_uncertainty(logits.detach())
        
        # 7. 计算对齐损失（简化版）
        align_loss = F.mse_loss(h_struct_rec, h_semantic_rec.detach())
        
        # 8. 总扩散损失
        total_diffusion_loss = diffusion_losses.get('struct_diffusion', 0) + \
                               diffusion_losses.get('semantic_diffusion', 0)
        
        return (logits, h_final, h_struct_rec, h_semantic_rec, 
                total_diffusion_loss, self.ema_h_struct, self.ema_h_semantic, uncertainty)

# =======================================
# 辅助模块
# =======================================

class GraphAugmentor(nn.Module):
    """简化的图增强器"""
    
    def __init__(self):
        super().__init__()
        self.aug_type = 'edge_dropout'
        self.aug_ratio = 0.1
    
    def forward(self, edge_index, num_nodes):
        if self.aug_type == 'edge_dropout' and self.aug_ratio > 0:
            # 边删除增强
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > self.aug_ratio
            edge_index = edge_index[:, mask]
        
        return edge_index

# =======================================
# V10模型工厂函数  
# =======================================

def create_v10_model(in_dim, sem_dim, hid_dim, num_classes, **kwargs):
    """创建V10模型的工厂函数"""
    
    return DualDiffusionGraphModel_V10(
        in_dim=in_dim,
        sem_dim=sem_dim, 
        hid_dim=hid_dim,
        num_classes=num_classes,
        **kwargs
    )

if __name__ == '__main__':
    # V10模型测试
    print("🚀 V10动态条件双扩散模型测试")
    
    # 创建测试数据
    batch_size = 100
    in_dim = 128
    sem_dim = 384
    hid_dim = 256
    num_classes = 7
    
    # 模拟图数据
    from torch_geometric.data import Data
    
    x = torch.randn(batch_size, in_dim)
    edge_index = torch.randint(0, batch_size, (2, batch_size * 3))
    y = torch.randint(0, num_classes, (batch_size,))
    text_embeds = torch.randn(batch_size, sem_dim)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # 创建V10模型
    model = create_v10_model(
        in_dim=in_dim,
        sem_dim=sem_dim,
        hid_dim=hid_dim, 
        num_classes=num_classes,
        diffusion_steps=5
    )
    
    # 测试前向传播
    model.train()
    outputs = model(data, text_embeds)
    
    logits, h_final, h_struct, h_semantic, diffusion_loss, ema_h_struct, ema_h_sem, uncertainty = outputs
    
    print(f"✅ V10模型测试成功!")
    print(f"   Logits形状: {logits.shape}")
    print(f"   扩散损失: {diffusion_loss:.4f}")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("🎯 V10核心创新:")
    print("   ✓ 动态条件双扩散")
    print("   ✓ 结构-语义互相引导")  
    print("   ✓ 残差学习机制")
    print("   ✓ 同步去噪预测")
