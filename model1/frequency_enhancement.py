# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 只使用你指定的GPU1
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from typing import Tuple, Optional


# # 统一使用单GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class FrequencyDomainEnhancementAndPurification(nn.Module):
#     """频域特征增强与纯化单元（单GPU版+局部窗口注意力）"""
    
#     def __init__(
#         self,
#         num_subbands: int = 4,
#         img_size: Tuple[int, int] = (512, 512),
#         attention_heads: int = 4,
#         hidden_channels: int = 32,
#         window_size: int = 8  # 局部窗口大小，8×8是精度和速度的最优平衡
#     ):
#         super().__init__()
#         self.num_subbands = num_subbands
#         self.H, self.W = img_size
#         self.attention_heads = attention_heads
#         self.hidden_channels = hidden_channels
#         self.window_size = window_size

#         # ====================== 1. 局部窗口跨频率结构注意力 ======================
#         self.attention_qkv = nn.Conv2d(num_subbands, 3 * hidden_channels, kernel_size=1)
#         self.attention_proj = nn.Conv2d(hidden_channels, num_subbands, kernel_size=1)
#         self.attention_norm = nn.LayerNorm([num_subbands, self.H, self.W])
        
#         # ====================== 2. 组织自适应软频域门控 ======================
#         self.gate_predictor = nn.Sequential(
#             nn.Conv2d(num_subbands + 3, hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(hidden_channels),
#             nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(hidden_channels),
#             nn.Conv2d(hidden_channels, num_subbands, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
        
#         # ====================== 3. 多频率语义一致性校正 ======================
#         self.consistency_correction = nn.Conv2d(num_subbands, num_subbands, kernel_size=3, padding=1)
#         self.correction_norm = nn.LayerNorm([num_subbands, self.H, self.W])
        
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
#         """将特征图划分为不重叠的局部窗口"""
#         B, C, H, W = x.shape
#         x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
#         windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
#         windows = windows.view(-1, self.window_size, self.window_size, C)
#         return windows

#     def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         """将局部窗口合并回特征图"""
#         B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
#         x = windows.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, -1)
#         x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
#         x = x.view(B, -1, H, W)
#         return x

#     def _cross_frequency_structural_attention(
#         self,
#         x: torch.Tensor,
#         structure_prior: torch.Tensor
#     ) -> torch.Tensor:
#         """局部窗口跨频率结构注意力（内存占用<100MB）"""
#         B, K, H, W = x.shape
        
#         # 1. 生成QKV
#         qkv = self.attention_qkv(x)
#         q, k, v = torch.chunk(qkv, 3, dim=1)
        
#         # 2. 划分为局部窗口
#         q_windows = self._window_partition(q)
#         k_windows = self._window_partition(k)
#         v_windows = self._window_partition(v)
        
#         # 3. 重塑为多头注意力格式
#         C_per_head = self.hidden_channels // self.attention_heads
#         q = q_windows.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
#         k = k_windows.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
#         v = v_windows.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
        
#         # 4. 计算注意力分数（每个窗口内计算，N=64，内存可忽略）
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C_per_head ** 0.5)
        
#         # 5. 加入结构先验偏置
#         structure_prior_windows = self._window_partition(structure_prior)
#         structure_bias = structure_prior_windows.view(-1, 1, self.window_size*self.window_size, 1)
#         attn_scores = attn_scores + structure_bias
        
#         # 6. 计算注意力权重并应用
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         out = torch.matmul(attn_weights, v)
        
#         # 7. 重塑并合并窗口
#         out = out.transpose(1, 2).contiguous().view(-1, self.window_size, self.window_size, self.hidden_channels)
#         out = self._window_reverse(out, H, W)
#         out = self.attention_proj(out)
        
#         # 8. 残差连接+归一化
#         out = self.attention_norm(x + out)
        
#         return out

#     def _adaptive_soft_frequency_gate(
#         self,
#         x: torch.Tensor,
#         attenuation_prior: torch.Tensor,
#         scattering_prior: torch.Tensor,
#         structure_prior: torch.Tensor
#     ) -> torch.Tensor:
#         gate_input = torch.cat([x, attenuation_prior, scattering_prior, structure_prior], dim=1)
#         gate_values = self.gate_predictor(gate_input)
        
#         for k in range(self.num_subbands):
#             # 高衰减区域降低高频权重
#             freq_penalty = 1.0 - (k / (self.num_subbands - 1)) * attenuation_prior
#             gate_values[:, k:k+1] *= freq_penalty
#             # 细散斑区域降低高频权重
#             speckle_penalty = 1.0 - (k / (self.num_subbands - 1)) * scattering_prior
#             gate_values[:, k:k+1] *= speckle_penalty
#             # 结构区域提高所有频带权重
#             gate_values[:, k:k+1] *= (0.5 + 0.5 * structure_prior)
        
#         return x * gate_values

#     def _semantic_consistency_correction(self, x: torch.Tensor) -> torch.Tensor:
#         residual = self.consistency_correction(x)
#         out = self.correction_norm(x + residual)
#         return out

#     def forward(
#         self,
#         img: torch.Tensor,
#         adaptive_filters_fft: torch.Tensor,
#         attenuation_prior: torch.Tensor,
#         scattering_prior: torch.Tensor,
#         structure_prior: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         B = img.shape[0]
        
#         # ====================== 步骤1：初始频域分解 ======================
#         img_fft = torch.fft.fft2(img, dim=(-2, -1))
#         initial_subbands = []
#         for k in range(self.num_subbands):
#             subband_fft = img_fft * adaptive_filters_fft[:, k:k+1, :, :]
#             subband = torch.fft.ifft2(subband_fft, dim=(-2, -1)).real
#             initial_subbands.append(subband)
#         initial_subbands = torch.cat(initial_subbands, dim=1)
        
#         # ====================== 步骤2：跨频率结构注意力 ======================
#         aligned_subbands = self._cross_frequency_structural_attention(initial_subbands, structure_prior)
        
#         # ====================== 步骤3：组织自适应软频域门控 ======================
#         purified_subbands = self._adaptive_soft_frequency_gate(aligned_subbands, attenuation_prior, scattering_prior, structure_prior)
        
#         # ====================== 步骤4：语义一致性校正 ======================
#         final_subbands = self._semantic_consistency_correction(purified_subbands)
        
#         return final_subbands, initial_subbands, aligned_subbands, purified_subbands

#     def visualize_subbands(
#         self,
#         img: torch.Tensor,
#         initial_subbands: torch.Tensor,
#         aligned_subbands: torch.Tensor,
#         purified_subbands: torch.Tensor,
#         final_subbands: torch.Tensor,
#         save_path: Optional[str] = None
#     ):
#         K = self.num_subbands
#         B = img.shape[0]
        
#         img_np = img[0, 0].cpu().numpy()
#         initial_np = initial_subbands[0].cpu().numpy()
#         aligned_np = aligned_subbands[0].cpu().numpy()
#         purified_np = purified_subbands[0].cpu().numpy()
#         final_np = final_subbands[0].cpu().numpy()
        
#         fig, axes = plt.subplots(5, K, figsize=(4*K, 20))
        
#         for k in range(K):
#             if k == 0:
#                 axes[0, k].imshow(img_np, cmap='gray')
#                 axes[0, k].set_title('Original Image')
#             else:
#                 axes[0, k].axis('off')
#             axes[0, k].axis('off')
        
#         for k in range(K):
#             axes[1, k].imshow(initial_np[k], cmap='gray')
#             axes[1, k].set_title(f'Initial Subband {k+1}')
#             axes[1, k].axis('off')
        
#         for k in range(K):
#             axes[2, k].imshow(aligned_np[k], cmap='gray')
#             axes[2, k].set_title(f'Aligned Subband {k+1}')
#             axes[2, k].axis('off')
        
#         for k in range(K):
#             axes[3, k].imshow(purified_np[k], cmap='gray')
#             axes[3, k].set_title(f'Purified Subband {k+1}')
#             axes[3, k].axis('off')
        
#         for k in range(K):
#             axes[4, k].imshow(final_np[k], cmap='gray')
#             axes[4, k].set_title(f'Final Subband {k+1}')
#             axes[4, k].axis('off')
        
#         plt.tight_layout()
#         if save_path is not None:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"子图可视化结果已保存到: {save_path}")
#         else:
#             plt.show()
#         plt.close()


# # ------------------------------
# # 单GPU完整运行示例
# # ------------------------------
# if __name__ == "__main__":
#     # 导入前两个模块
#     from model1.physical_prior import GlobalPhysicalPriorExtractor
#     from model1.adaptive_filter_bank import AdaptivePhysicalFilterBank
    
#     # 1. 初始化所有模块（统一放到单GPU）
#     prior_extractor = GlobalPhysicalPriorExtractor(window_size=7, device=device)
#     filter_bank = AdaptivePhysicalFilterBank(
#         num_filters=4,
#         img_size=(512, 512),
#         device=device
#     ).to(device)
    
#     enhancement_unit = FrequencyDomainEnhancementAndPurification(
#         num_subbands=4,
#         img_size=(512, 512),
#         window_size=8
#     ).to(device) 

#     image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"
    
#     try:
#         # 2. 先验提取
#         img = prior_extractor.read_image(image_path).to(device)
#         print(f"成功读取图像，尺寸: {img.shape}")
        
#         A, S, T = prior_extractor.extract_all_priors(img)
#         print(f"先验提取完成")
        
#         # 3. 生成自适应滤波器组
#         with torch.no_grad():
#             adaptive_filters_fft, weight_maps = filter_bank(A, S, T)
#         print(f"滤波器生成完成")
        
#         # 4. 频域增强与纯化
#         with torch.no_grad():
#             final_subbands, initial_subbands, aligned_subbands, purified_subbands = enhancement_unit(
#                 img, adaptive_filters_fft, A, S, T
#             )
#         print(f"频域增强与纯化完成")
#         print(f"最终子图shape: {final_subbands.shape}")
        
#         # 5. 可视化结果
#         enhancement_unit.visualize_subbands(
#             img, initial_subbands, aligned_subbands, purified_subbands, final_subbands,
#             save_path="subbands_visualization.png"
#         )
        
#         # 6. 保存结果
#         torch.save({
#             'final_subbands': final_subbands.cpu(),
#             'initial_subbands': initial_subbands.cpu(),
#             'aligned_subbands': aligned_subbands.cpu(),
#             'purified_subbands': purified_subbands.cpu()
#         }, "processed_subbands.pt")
#         print("子图数据已保存到: processed_subbands.pt")
        
#     except Exception as e:
#         print(f"处理出错: {str(e)}")
#         import traceback
#         traceback.print_exc()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class FrequencyDomainEnhancementAndPurification(nn.Module):
    """基于论文创新点3重重构的语义对齐与频域纯化单元（全标准全频谱fft2版）"""
    
    def __init__(
        self,
        num_subbands: int = 4,
        img_size: Tuple[int, int] = (512, 512),
        attention_heads: int = 4,
        hidden_channels: int = 32,
        window_size: int = 8
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.H, self.W = img_size
        self.attention_heads = attention_heads
        self.hidden_channels = hidden_channels
        self.window_size = window_size

        # 1. 局部窗口跨频率结构注意力
        self.attention_qkv = nn.Conv2d(num_subbands, 3 * hidden_channels, kernel_size=1)
        self.attention_proj = nn.Conv2d(hidden_channels, num_subbands, kernel_size=1)
        self.attention_norm = nn.LayerNorm([num_subbands, self.H, self.W])
        
        # 2. 组织自适应软频域门控
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(num_subbands + 3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, num_subbands, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 3. 多频率语义一致性校正
        self.consistency_correction = nn.Conv2d(num_subbands, num_subbands, kernel_size=3, padding=1)
        self.correction_norm = nn.LayerNorm([num_subbands, self.H, self.W])
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.LayerNorm)):
                m.weight.data.fill_(1.0) if isinstance(m, nn.LayerNorm) else nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        return x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size, self.window_size, C)

    def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, -1)
        return x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    def _cross_frequency_structural_attention(self, x: torch.Tensor, structure_prior: torch.Tensor) -> torch.Tensor:
        B, K, H, W = x.shape
        qkv = self.attention_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        q_w, k_w, v_w = map(self._window_partition, [q, k, v])
        C_per_head = self.hidden_channels // self.attention_heads
        
        q_w = q_w.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
        k_w = k_w.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
        v_w = v_w.view(-1, self.window_size*self.window_size, self.attention_heads, C_per_head).transpose(1, 2)
        
        attn_scores = torch.matmul(q_w, k_w.transpose(-2, -1)) / (C_per_head ** 0.5)
        
        # 融入组织结构先验偏置，引导跨频对齐
        sb_w = self._window_partition(structure_prior).view(-1, 1, self.window_size*self.window_size, 1)
        attn_scores = attn_scores + sb_w
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_w).transpose(1, 2).contiguous().view(-1, self.window_size, self.window_size, self.hidden_channels)
        out = self.attention_proj(self._window_reverse(out, H, W))
        
        return self.attention_norm(x + out)

    def _adaptive_soft_frequency_gate(self, x: torch.Tensor, A: torch.Tensor, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x, A, S, T], dim=1)
        gate_values = self.gate_predictor(gate_input)
        
        for k in range(self.num_subbands):
            # 概率化软调制：深度高衰减或强噪声细散斑区域，压制极高频成分，缓解频谱混叠
            gate_values[:, k:k+1] *= (1.0 - (k / (self.num_subbands - 1)) * A)
            gate_values[:, k:k+1] *= (1.0 - (k / (self.num_subbands - 1)) * S)
            gate_values[:, k:k+1] *= (0.5 + 0.5 * T)
            
        return x * gate_values

    def _semantic_consistency_correction(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_norm(x + self.consistency_correction(x))

    def forward(
        self,
        img: torch.Tensor,
        base_filters_fft: torch.Tensor,
        weight_maps: torch.Tensor,
        attenuation_prior: torch.Tensor,
        scattering_prior: torch.Tensor,
        structure_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整前向传播流程
        实现了：文字所述的“在频域中对原始超声图像进行加权分解，得到初始多频率子图表示”
        """
        B, _, H, W = img.shape
        
        # ====================== 步骤1：基础频域滤波 + 空间自适应权重调制 ======================
        img_fft = torch.fft.fft2(img, dim=(-2, -1))
        initial_subbands = []
        for k in range(self.num_subbands):
            # 严格频域滤波
            subband_fft = img_fft * base_filters_fft[k:k+1, :, :]
            base_subband = torch.fft.ifft2(subband_fft, dim=(-2, -1)).real
            # 落实空间域“组织自适应动态权重调制”
            initial_subband = base_subband * weight_maps[:, k:k+1, :, :]
            initial_subbands.append(initial_subband)
        initial_subbands = torch.cat(initial_subbands, dim=1)
        
        # ====================== 步骤2与3：跨频率语义对齐与频域纯化 ======================
        aligned_subbands = self._cross_frequency_structural_attention(initial_subbands, structure_prior)
        purified_subbands = self._adaptive_soft_frequency_gate(aligned_subbands, attenuation_prior, scattering_prior, structure_prior)
        final_subbands = self._semantic_consistency_correction(purified_subbands)
        
        return final_subbands, initial_subbands, aligned_subbands, purified_subbands

# ==========================================
# 模块三独立验证与多频率子图提取脚本 (Subband Extraction Script)
# ==========================================
if __name__ == "__main__":
    from physical_prior import GlobalPhysicalPriorExtractor
    from adaptive_filter_bank import AdaptivePhysicalFilterBank
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[验证] 正在初始化模块三多频率子图提取程序，使用设备: {device}")
    
    # 1. 实例化一整套前置解耦流水线
    prior_extractor = GlobalPhysicalPriorExtractor(window_size=7, device=device)
    filter_bank = AdaptivePhysicalFilterBank(num_filters=4, img_size=(512, 512), device=device).to(device)
    enhancement_unit = FrequencyDomainEnhancementAndPurification(num_subbands=4, img_size=(512, 512)).to(device)
    
    # 2. 加载图像并提取先验与权重
    image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"
    try:
        img = prior_extractor.read_image(image_path).to(device)
    except Exception as e:
        print(f"⚠️ 未能读取到指定路径图像，切换为模拟图案... Error: {e}")
        img = torch.randn(1, 1, 512, 512).to(device).abs()
        
    A, S, T = prior_extractor.extract_all_priors(img)
    
    with torch.no_grad():
        weight_maps = filter_bank(A, S, T)
        
        # 3. 核心前向传播：将单张超声图分解为 4 个纯化后的多频率表示子图
        # 抛弃一切下游去噪、重构和 Loss，只拿最核心的频域解耦结果
        final_subbands, _, _, _ = enhancement_unit(
            img, filter_bank.base_filters_fft, weight_maps, A, S, T
        )
        
    # 4. 转换数据至 CPU 用于量化打印与画图
    subbands_np = final_subbands.squeeze(0).cpu().numpy() # Shape: (4, 512, 512)
    img_np = img.squeeze(0).squeeze(0).cpu().numpy()
    
    print("\n[验证结果] 成功解耦并纯化多频率中间表示：")
    print(f" - 最终输出多频率表示矩阵维度: {final_subbands.shape}")
    for k in range(4):
        print(f"   -> 子图 {k+1} (频率由低到高) 像素值范围: [{subbands_np[k].min():.4f}, {subbands_np[k].max():.4f}]")
        
    # 5. 绘制多频率表示分解结果（直接用于放进论文插图）
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    
    # 原始图像
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Original B-mode Image", fontsize=12)
    axes[0].axis('off')
    
    # 4个独立解耦频率子图
    titles = [
        "Subband 1 (Low Freq - Coarse Texture)",
        "Subband 2 (Mid-Low Freq - Structure)",
        "Subband 3 (Mid-High Freq - Details)",
        "Subband 4 (High Freq - Fine Speckle)"
    ]
    cmaps = ['bone', 'bone', 'bone', 'bone'] # 使用标准临床超声骨色调，更具可读性
    
    for k in range(4):
        im_s = axes[k+1].imshow(subbands_np[k], cmap=cmaps[k])
        axes[k+1].set_title(titles[k], fontsize=10)
        axes[k+1].axis('off')
        fig.colorbar(im_s, ax=axes[k+1], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    save_path = "verification_subbands_output.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 论文多频率子图分解结果已保存至: {save_path} (请观察子图1到子图4是否成功实现了从宏观轮廓到微观散斑的物理隔离)")
    
    # 6. 【可选】将4个子图独立保存为单独的 .pt 文件，方便随时喂给后续任何你想尝试的第三方网络
    torch.save(final_subbands.cpu(), "purified_frequency_subbands.pt")
    print("💾 4个频率子图矩阵已打包保存为 'purified_frequency_subbands.pt'。")
    plt.close()