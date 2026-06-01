import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math 
from typing import Tuple, Optional

class AdaptivePhysicalFilterBank(nn.Module):
    """基于论文创新点2重构的组织自适应频带生成模块
    核心设计：标准物理标准Log-Gabor基础频带 + 空间像素级自适应权重预测
    统一采用标准全频谱 fft2 架构，完美解决维度冲突
    """
    
    def __init__(
        self,
        num_filters: int = 4,  # 标准配置K=4，对应临床超声4个频带
        img_size: Tuple[int, int] = (512, 512),  # 输入图像尺寸(H,W)
        base_frequencies: Tuple[float, ...] = (0.05, 0.15, 0.3, 0.45),
        bandwidths: Tuple[float, ...] = (0.08, 0.12, 0.15, 0.1),
        weight_net_channels: int = 32,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_filters = num_filters
        self.H, self.W = img_size
        self.base_frequencies = base_frequencies
        self.bandwidths = bandwidths
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预生成符合全频谱（fft2）维度的固定 Log-Gabor 基础滤波器组
        self.base_filters_fft = self._generate_log_gabor_filters().to(self.device)
        
        # 轻量级自适应权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(3, weight_net_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(weight_net_channels),
            nn.Conv2d(weight_net_channels, weight_net_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(weight_net_channels),
            nn.Conv2d(weight_net_channels, num_filters, kernel_size=3, padding=1),
            nn.Sigmoid()  # 严格限制权重在[0,1]
        ).to(self.device)
        
        self._initialize_weights()

    def _generate_log_gabor_filters(self) -> torch.Tensor:
        """生成标准全频谱全自适应 2D Log-Gabor 滤波器组"""
        # 使用 fftfreq 生成对称的正负频率网格，完美契合 torch.fft.fft2 维度要求
        y = torch.fft.fftfreq(self.H, device=self.device).view(self.H, 1)
        x = torch.fft.fftfreq(self.W, device=self.device).view(1, self.W)
        
        radius = torch.sqrt(y**2 + x**2)
        radius[0, 0] = 1e-8  # 规避 log(0)
        
        filters = []
        for f0, bw in zip(self.base_frequencies, self.bandwidths):
            sigma = f0 * bw
            log_sigma_over_f0 = math.log(sigma / f0)
            filter_fft = torch.exp(-(torch.log(radius / f0))**2 / (2 * log_sigma_over_f0**2))
            filter_fft[0, 0] = 0.0  # 剔除直流DC分量
            filters.append(filter_fft)
            
        # 返回 Shape: (K, self.H, self.W)
        return torch.stack(filters, dim=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)

    def forward(
        self,
        attenuation_prior: torch.Tensor,
        scattering_prior: torch.Tensor,
        structure_prior: torch.Tensor
    ) -> torch.Tensor:
        """前向传播预测空间动态加权图
        Returns:
            weight_maps: 空间域像素级响应加权图, shape: (B, K, H, W)
        """
        prior_input = torch.cat([attenuation_prior, scattering_prior, structure_prior], dim=1)
        weight_maps = self.weight_predictor(prior_input)
        
        # 统计物理先验驱动的动态加权调制约束
        for k in range(self.num_filters):
            # 约束1：高衰减区域(A大)降低高频子图的响应权重值
            freq_penalty = 1.0 - (k / (self.num_filters - 1)) * attenuation_prior
            weight_maps[:, k:k+1] *= freq_penalty
            
        # 约束2：稳态组织结构区域(T大)整体增强响应，保护边缘
        weight_maps *= (0.5 + 0.5 * structure_prior)
        
        return weight_maps
    

if __name__ == "__main__":
    from physical_prior import GlobalPhysicalPriorExtractor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[验证] 正在初始化模块二验证程序，使用设备: {device}")
    
    # 1. 初始化先验提取器与自适应滤波器组
    prior_extractor = GlobalPhysicalPriorExtractor(window_size=7, device=device)
    filter_bank = AdaptivePhysicalFilterBank(num_filters=4, img_size=(512, 512), device=device).to(device)
    
    # 2. 读取你的超声图像并提取物理先验
    image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"
    try:
        img = prior_extractor.read_image(image_path).to(device)
    except Exception as e:
        print(f"⚠️ 未能读取到指定路径图像，切换为生成模拟超声图案进行验证... Error: {e}")
        # 如果路径不存在，自动生成无损模拟图防止崩溃
        img = torch.randn(1, 1, 512, 512).to(device).abs()
        
    A, S, T = prior_extractor.extract_all_priors(img)
    
    # 3. 前向传播生成空间自适应权重图
    with torch.no_grad():
        weight_maps = filter_bank(A, S, T)
        
    # 4. 提取生成的频域基滤波器用于可视化 (转为中央低频对称形式 fftshift)
    base_filters = filter_bank.base_filters_fft.cpu()
    weights = weight_maps.squeeze(0).cpu() # Shape: (4, H, W)
    
    print("\n[验证结果] 成功生成自适应调制数据：")
    print(f" - 基础滤波器组尺寸: {base_filters.shape}")
    print(f" - 动态响应权重图尺寸: {weights.shape}")
    
    # 5. 绘图保存
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # 第一行：展示 4 个 Log-Gabor 基滤波器在频域的覆盖情况
    for k in range(4):
        # 使用 fftshift 将直流分量(低频)移到中心，方便论文可视化观感
        filter_centered = torch.fft.fftshift(base_filters[k])
        im_f = axes[0, k].imshow(filter_centered.numpy(), cmap='viridis')
        axes[0, k].set_title(f"Subband {k+1} Base Filter\n(Frequency Domain)", fontsize=10)
        axes[0, k].axis('off')
        fig.colorbar(im_f, ax=axes[0, k], fraction=0.046, pad=0.04)
        
    # 第二行：展示轻量网络预测的、受物理先验调制的空间像素级权重图
    for k in range(4):
        im_w = axes[1, k].imshow(weights[k].numpy(), cmap='magma', vmin=0, vmax=1)
        axes[1, k].set_title(f"Subband {k+1} Adaptive Weight\n(Spatial Domain)", fontsize=10)
        axes[1, k].axis('off')
        fig.colorbar(im_w, ax=axes[1, k], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    save_path = "verification_filter_bank.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 模块二验证图已成功保存至: {save_path} (请查看各个频带的权重响应是否合理)")
    plt.close()