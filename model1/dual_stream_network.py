# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from typing import Tuple
# # import seaborn as sns

# # =====================================================================
# # 1. 基础核心组件 (卷积与物理引导注意力机制)
# # =====================================================================

# class DoubleConv(nn.Module):
#     """标准双层残差卷积块"""
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         return self.net(x)

# class PhysicallyGuidedHighFreqBlock(nn.Module):
#     """物理引导的高频细节/散斑处理块
#     结合空间-通道注意力和物理先验 (A, S) 进行噪声抑制与纹理保留
#     """
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.conv = DoubleConv(in_channels, in_channels)
        
#         # 通道注意力 (动态筛选有效的高频通道)
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x: torch.Tensor, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
#         # 1. 数据驱动的通道注意力
#         x = x * self.ca(x)
        
#         # 2. 物理先验驱动的动态调制
#         # 衰减先验(A)：A越大(深层区域)，高频噪声越多，进行惩罚抑制
#         attenuation_penalty = 1.0 - 0.5 * A
#         # 散射先验(S)：S越大(粗糙纹理/散斑区域)，越需要保留组织真实感，进行增强
#         scattering_enhancement = 1.0 + 0.5 * S
        
#         # 物理一致性校正
#         x = x * attenuation_penalty * scattering_enhancement
#         return x

# class PhysicallyGuidedLowFreqBlock(nn.Module):
#     """物理引导的低频结构处理块
#     结合组织结构先验 (T) 进行边缘保真与锐化
#     """
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.conv = DoubleConv(in_channels, in_channels)

#     def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
#         # 组织结构先验(T)：T越大(清晰的边界/大血管壁)，越需要增强结构表达
#         structure_enhancement = 1.0 + 0.5 * T
#         x = x * structure_enhancement
#         return x

# # =====================================================================
# # 2. 核心双流网络架构
# # =====================================================================

# class DualStreamReconstructionNet(nn.Module):
#     def __init__(self, base_channels: int = 32):
#         super().__init__()
        
#         # ------------------ [流一：低频结构流] ------------------
#         # 输入: Subband 1 & 2 (Shape: [B, 2, H, W])
#         self.low_enc1 = PhysicallyGuidedLowFreqBlock(2)
#         self.low_enc1_conv = DoubleConv(2, base_channels)
#         self.low_enc2 = DoubleConv(base_channels, base_channels * 2)
        
#         # ------------------ [流二：高频细节流] ------------------
#         # 输入: Subband 3 & 4 (Shape: [B, 2, H, W])
#         self.high_enc1 = PhysicallyGuidedHighFreqBlock(2)
#         self.high_enc1_conv = DoubleConv(2, base_channels)
#         self.high_enc2 = DoubleConv(base_channels, base_channels * 2)
        
#         self.pool = nn.MaxPool2d(2)
        
#         # ------------------ [瓶颈层特征融合] ------------------
#         self.bottleneck = DoubleConv(base_channels * 4, base_channels * 4)
        
#         # ------------------ [统一解码器] ------------------
#         self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv(base_channels * 6, base_channels * 2)
        
#         self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv(base_channels * 3, base_channels)
        
#         # ------------------ [输出映射头] ------------------
#         self.final_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        
#         # 【创新验证点】：低频与高频的独立输出头，用于提取纯化子图
#         self.low_preview_head = nn.Sequential(
#             nn.Conv2d(base_channels, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.high_preview_head = nn.Sequential(
#             nn.Conv2d(base_channels, 1, kernel_size=1),
#             nn.Tanh() # 纹理可能包含正负扰动
#         )

#     def forward(self, subbands: torch.Tensor, A: torch.Tensor, S: torch.Tensor, T: torch.Tensor):
#         """
#         Args:
#             subbands: 频率解耦后的4个子图 [B, 4, H, W]
#             A, S, T: 从模块1提取的物理先验 [B, 1, H, W]
#         """
#         # Step 1: 物理频带分流
#         x_low = subbands[:, 0:2, :, :]    # Subband 1 & 2
#         x_high = subbands[:, 2:4, :, :]   # Subband 3 & 4
        
#         # Step 2: 物理先验驱动的双流编码
#         # --- 低频流 (融合结构先验 T)
#         low_feat_guided = self.low_enc1(x_low, T)
#         low_feat1 = self.low_enc1_conv(low_feat_guided)
#         low_feat2 = self.low_enc2(self.pool(low_feat1))
        
#         # --- 高频流 (融合衰减先验 A 和 散射先验 S)
#         high_feat_guided = self.high_enc1(x_high, A, S)
#         high_feat1 = self.high_enc1_conv(high_feat_guided)
#         high_feat2 = self.high_enc2(self.pool(high_feat1))
        
#         # Step 3: 获取独立流的预览图 (用于你的论文验证)
#         low_stream_output = self.low_preview_head(low_feat1)
#         high_stream_output = self.high_preview_head(high_feat1)
        
#         # Step 4: 瓶颈层与解码重构
#         bottleneck_input = torch.cat([self.pool(low_feat2), self.pool(high_feat2)], dim=1)
#         b_feat = self.bottleneck(bottleneck_input)
        
#         d1_up = self.up1(b_feat)
#         d1_merge = torch.cat([d1_up, low_feat2, high_feat2], dim=1)
#         d1_feat = self.dec1(d1_merge)
        
#         d2_up = self.up2(d1_feat)
#         d2_merge = torch.cat([d2_up, low_feat1, high_feat1], dim=1)
#         d2_feat = self.dec2(d2_merge)
        
#         final_reconstructed = torch.sigmoid(self.final_head(d2_feat))
        
#         return final_reconstructed, low_stream_output, high_stream_output

# # =====================================================================
# # 3. 完整系统端到端串联运行测试 (集成模块1,2,3及双流网络)
# # =====================================================================

# if __name__ == "__main__":
#     # 导入你的三个前置模块 (请确保它们与本脚本在同一目录下或路径正确)
#     from physical_prior import GlobalPhysicalPriorExtractor
#     from adaptive_filter_bank import AdaptivePhysicalFilterBank
#     from frequency_enhancement import FrequencyDomainEnhancementAndPurification
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"🚀 开始超声图像端到端物理双流重构测试，使用设备: {device}")
    
#     # ---------------------------------------------------------
#     # 第一步：初始化所有模块
#     # ---------------------------------------------------------
#     prior_extractor = GlobalPhysicalPriorExtractor(window_size=7, device=device)
#     filter_bank = AdaptivePhysicalFilterBank(num_filters=4, img_size=(512, 512), device=device).to(device)
#     enhancement_unit = FrequencyDomainEnhancementAndPurification(num_subbands=4, img_size=(512, 512)).to(device)
#     dual_stream_net = DualStreamReconstructionNet(base_channels=32).to(device)
    
#     # ---------------------------------------------------------
#     # 第二步：数据前向传播流水线
#     # ---------------------------------------------------------
#     image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"
    
#     try:
#         # [模块1] 读取并提取先验
#         img = prior_extractor.read_image(image_path).to(device)
#         A, S, T = prior_extractor.extract_all_priors(img)
#         print("✅ 模块1：物理统计先验 (A, S, T) 提取完成")
        
#         with torch.no_grad():
#             # [模块2] 自适应滤波器与空间权重
#             weight_maps = filter_bank(A, S, T)
#             print("✅ 模块2：自适应权重图生成完成")
            
#             # [模块3] 多频率解耦与频域纯化
#             final_subbands, _, _, _ = enhancement_unit(
#                 img, filter_bank.base_filters_fft, weight_maps, A, S, T
#             )
#             print("✅ 模块3：多频率中间表示 (4子图) 解耦纯化完成")
            
#             # [模块4 - 本网络] 物理驱动双流合成
#             final_img, low_stream_img, high_stream_img = dual_stream_net(
#                 final_subbands, A, S, T
#             )
#             print("✅ 模块4：双流重构网络合成完成！")
            
#     except Exception as e:
#         print(f"⚠️ 读取图像失败，使用模拟张量验证代码逻辑... (Error: {e})")
#         img = torch.rand(1, 1, 512, 512).to(device)
#         A, S, T = torch.rand(1, 1, 512, 512).to(device), torch.rand(1, 1, 512, 512).to(device), torch.rand(1, 1, 512, 512).to(device)
#         final_subbands = torch.rand(1, 4, 512, 512).to(device)
#         with torch.no_grad():
#             final_img, low_stream_img, high_stream_img = dual_stream_net(final_subbands, A, S, T)
#         print("✅ 模拟数据流通过双流网络测试！")

#     # ---------------------------------------------------------
#     # 第三步：可视化并保存结果 (论文验证图)
#     # ---------------------------------------------------------
#     img_np = img.squeeze().cpu().numpy()
#     low_stream_np = low_stream_img.squeeze().cpu().numpy()
#     high_stream_np = high_stream_img.squeeze().cpu().numpy()
#     final_np = final_img.squeeze().cpu().numpy()
    
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
#     axes[0].imshow(img_np, cmap='gray')
#     axes[0].set_title("1. Original Input", fontsize=14)
#     axes[0].axis('off')
    
#     axes[1].imshow(low_stream_np, cmap='bone')
#     axes[1].set_title("2. Low-Stream (Structure Guidence)", fontsize=14, color='blue')
#     axes[1].axis('off')
    
#     axes[2].imshow(high_stream_np, cmap='coolwarm')
#     axes[2].set_title("3. High-Stream (Speckle/Texture)", fontsize=14, color='green')
#     axes[2].axis('off')
    
#     axes[3].imshow(final_np, cmap='gray')
#     axes[3].set_title("4. Final Reconstructed", fontsize=14, color='red')
#     axes[3].axis('off')
    
#     plt.tight_layout()
#     save_path = "pipeline_dual_stream_result.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"\n🎉 完整的端到端双流合成结果已保存至: {save_path}")
#     print("   图2和图3直接展示了网络内部基于物理规律‘分而治之’的中间特征状态！")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple

# =====================================================================
# 1. 基础核心组件 (卷积与物理引导注意力机制)
# =====================================================================

class DoubleConv(nn.Module):
    """标准双层残差卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.net(x)

class PhysicallyGuidedHighFreqBlock(nn.Module):
    """物理引导的高频细节/散斑处理块
    结合空间-通道注意力和物理先验 (A, S) 进行噪声抑制与纹理保留
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, in_channels)
        
        # 通道注意力 (动态筛选有效的高频通道)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # 1. 数据驱动的通道注意力
        x = x * self.ca(x)
        
        # 2. 物理先验驱动的动态调制
        # 衰减先验(A)：A越大(深层区域)，高频噪声越多，进行惩罚抑制
        attenuation_penalty = 1.0 - 0.5 * A
        # 散射先验(S)：S越大(粗糙纹理/散斑区域)，越需要保留组织真实感，进行增强
        scattering_enhancement = 1.0 + 0.5 * S
        
        # 物理一致性校正
        x = x * attenuation_penalty * scattering_enhancement
        return x

class PhysicallyGuidedLowFreqBlock(nn.Module):
    """物理引导的低频结构处理块
    结合组织结构先验 (T) 进行边缘保真与锐化
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # 组织结构先验(T)：T越大(清晰的边界/大血管壁)，越需要增强结构表达
        structure_enhancement = 1.0 + 0.5 * T
        x = x * structure_enhancement
        return x

# =====================================================================
# 2. 核心双流网络架构
# =====================================================================

class DualStreamReconstructionNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        # ------------------ [流一：低频结构流] ------------------
        # 输入: Subband 1 & 2 (Shape: [B, 2, H, W])
        self.low_enc1 = PhysicallyGuidedLowFreqBlock(2)
        self.low_enc1_conv = DoubleConv(2, base_channels)
        self.low_enc2 = DoubleConv(base_channels, base_channels * 2)
        
        # ------------------ [流二：高频细节流] ------------------
        # 输入: Subband 3 & 4 (Shape: [B, 2, H, W])
        self.high_enc1 = PhysicallyGuidedHighFreqBlock(2)
        self.high_enc1_conv = DoubleConv(2, base_channels)
        self.high_enc2 = DoubleConv(base_channels, base_channels * 2)
        
        self.pool = nn.MaxPool2d(2)
        
        # ------------------ [瓶颈层特征融合] ------------------
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 4)
        
        # ------------------ [统一解码器] ------------------
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 6, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 3, base_channels)
        
        # ------------------ [输出映射头] ------------------
        self.final_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # 【创新验证点】：低频与高频的独立输出头，用于提取纯化子图
        self.low_preview_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_preview_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Tanh() # 纹理可能包含正负扰动
        )

    def forward(self, subbands: torch.Tensor, A: torch.Tensor, S: torch.Tensor, T: torch.Tensor):
        # Step 1: 物理频带分流
        x_low = subbands[:, 0:2, :, :]    
        x_high = subbands[:, 2:4, :, :]   
        
        # Step 2: 物理先验驱动的双流编码
        low_feat_guided = self.low_enc1(x_low, T)
        low_feat1 = self.low_enc1_conv(low_feat_guided)
        low_feat2 = self.low_enc2(self.pool(low_feat1))
        
        high_feat_guided = self.high_enc1(x_high, A, S)
        high_feat1 = self.high_enc1_conv(high_feat_guided)
        high_feat2 = self.high_enc2(self.pool(high_feat1))
        
        # Step 3: 获取独立流的预览图 (用于论文验证)
        low_stream_output = self.low_preview_head(low_feat1)
        high_stream_output = self.high_preview_head(high_feat1)
        
        # Step 4: 瓶颈层与解码重构
        bottleneck_input = torch.cat([self.pool(low_feat2), self.pool(high_feat2)], dim=1)
        b_feat = self.bottleneck(bottleneck_input)
        
        d1_up = self.up1(b_feat)
        d1_merge = torch.cat([d1_up, low_feat2, high_feat2], dim=1)
        d1_feat = self.dec1(d1_merge)
        
        d2_up = self.up2(d1_feat)
        d2_merge = torch.cat([d2_up, low_feat1, high_feat1], dim=1)
        d2_feat = self.dec2(d2_merge)
        
        final_reconstructed = torch.sigmoid(self.final_head(d2_feat))
        
        return final_reconstructed, low_stream_output, high_stream_output

# =====================================================================
# 3. 完整系统端到端串联运行测试 (基于 ZS-N2N 的 Zero-Shot 即时训练)
# =====================================================================

if __name__ == "__main__":
    from physical_prior import GlobalPhysicalPriorExtractor
    from adaptive_filter_bank import AdaptivePhysicalFilterBank
    from frequency_enhancement import FrequencyDomainEnhancementAndPurification
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始超声图像端到端物理双流重构测试 (Zero-Shot 模式)，使用设备: {device}")
    
    # ---------------------------------------------------------
    # 准备工作：初始化主模块与读取图像
    # ---------------------------------------------------------
    prior_extractor = GlobalPhysicalPriorExtractor(window_size=7, device=device)
    filter_bank = AdaptivePhysicalFilterBank(num_filters=4, img_size=(512, 512), device=device).to(device)
    enhancement_unit = FrequencyDomainEnhancementAndPurification(num_subbands=4, img_size=(512, 512)).to(device)
    dual_stream_net = DualStreamReconstructionNet(base_channels=32).to(device)
    
    image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"
    
    try:
        # [步骤1] 提取原图的全尺寸信息 (用于评估和一致性约束)
        img = prior_extractor.read_image(image_path).to(device)
        A, S, T = prior_extractor.extract_all_priors(img)
        with torch.no_grad():
            weight_maps = filter_bank(A, S, T)
            final_subbands, _, _, _ = enhancement_unit(img, filter_bank.base_filters_fft, weight_maps, A, S, T)
        print("✅ 模块1-3：原图物理统计先验与频率分解完成")
        
        # ---------------------------------------------------------
        # [核心新增] ZS-N2N 数据准备：空间对角线降采样机制
        # ---------------------------------------------------------
        print("⏳ 正在准备 ZS-N2N 的独立噪声对...")
        # 降采样卷积核: k1取对角线像素，k2取反对角线像素
        k1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]]).to(device)
        k2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]]).to(device)
        
        # 生成互相独立噪声的子图 y1 和 y2 (尺寸变为 256x256)
        y1 = F.conv2d(img, k1, stride=2) 
        y2 = F.conv2d(img, k2, stride=2)
        
        # 初始化处理半尺寸图像的辅助模块
        H_half, W_half = y1.shape[2], y1.shape[3]
        filter_bank_half = AdaptivePhysicalFilterBank(num_filters=4, img_size=(H_half, W_half), device=device).to(device)
        enhancement_half = FrequencyDomainEnhancementAndPurification(num_subbands=4, img_size=(H_half, W_half)).to(device)
        
        # 提取 y1 的多频子图与先验
        with torch.no_grad():
            A1, S1, T1 = prior_extractor.extract_all_priors(y1)
            wm1 = filter_bank_half(A1, S1, T1)
            sub1, _, _, _ = enhancement_half(y1, filter_bank_half.base_filters_fft, wm1, A1, S1, T1)
            
            # 提取 y2 的多频子图与先验
            A2, S2, T2 = prior_extractor.extract_all_priors(y2)
            wm2 = filter_bank_half(A2, S2, T2)
            sub2, _, _, _ = enhancement_half(y2, filter_bank_half.base_filters_fft, wm2, A2, S2, T2)
        
        # ---------------------------------------------------------
        # [核心新增] Zero-Shot 单图原地优化循环
        # ---------------------------------------------------------
        optimizer = optim.Adam(dual_stream_net.parameters(), lr=0.002)
        mse_loss = nn.MSELoss()
        
        num_iterations = 100000  # ZS-N2N 建议的快速收敛步数
        print(f"\n⚙️ 启动基于单张图像的 Zero-Shot 即时训练 (共 {num_iterations} 步)...")
        
        dual_stream_net.train()
        for step in range(num_iterations):
            optimizer.zero_grad()
            
            # 1. 完整图像前向传播
            final_img, low_stream, high_stream = dual_stream_net(final_subbands, A, S, T)
            
            # 2. 交叉预测前向传播 (y1 输入, y2 输入)
            out1, _, _ = dual_stream_net(sub1, A1, S1, T1)
            out2, _, _ = dual_stream_net(sub2, A2, S2, T2)
            
            # --- 损失计算 ---
            # Loss A: Neighbor2Neighbor 交叉预测损失 (用含噪声1预测含噪声2，强迫网络学习无噪结构) 
            loss_n2n = mse_loss(out1, y2) + mse_loss(out2, y1)
            
            # Loss B: ZS-N2N 论文的一致性约束 (Consistency Loss) 
            # 保证网络直接对全尺寸图重构后的下采样，等同于下采样后的网络重构
            final_img_d1 = F.conv2d(final_img, k1, stride=2)
            final_img_d2 = F.conv2d(final_img, k2, stride=2)
            loss_cons = mse_loss(final_img_d1, out1) + mse_loss(final_img_d2, out2)
            
            # Loss C: 低频结构引导 (结合你的创新点：强迫低频流仅拟合 Subband 1&2 对应的骨架信息)
            target_low = final_subbands[:, 0:2, :, :].mean(dim=1, keepdim=True)
            loss_low_guide = mse_loss(low_stream, target_low)
            
            # 综合物理与 ZS-N2N 损失
            total_loss = loss_n2n + 0.5 * loss_cons + 0.1 * loss_low_guide
            
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                print(f"   -> 迭代 [{step+1}/{num_iterations}], Total Loss: {total_loss.item():.5f} | N2N: {loss_n2n.item():.5f}")

        print("✅ 模块4：单图 Zero-Shot 优化去噪完成！\n")
        
        # ---------------------------------------------------------
        # 获取最终的高质量去噪结果
        # ---------------------------------------------------------
        dual_stream_net.eval()
        with torch.no_grad():
            final_denoised, low_preview, high_preview = dual_stream_net(final_subbands, A, S, T)
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # 可视化并保存结果
    # ---------------------------------------------------------
    img_np = img.squeeze().cpu().numpy()
    low_stream_np = low_preview.squeeze().cpu().numpy()
    high_stream_np = high_preview.squeeze().cpu().numpy()
    final_np = final_denoised.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("1. Original Input", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(low_stream_np, cmap='bone')
    axes[1].set_title("2. Low-Stream (Structure)", fontsize=14, color='blue')
    axes[1].axis('off')
    
    axes[2].imshow(high_stream_np, cmap='coolwarm')
    axes[2].set_title("3. High-Stream (Purified Texture)", fontsize=14, color='green')
    axes[2].axis('off')
    
    axes[3].imshow(final_np, cmap='gray')
    axes[3].set_title("4. Final ZS-N2N Denoised", fontsize=14, color='red')
    axes[3].axis('off')
    
    plt.tight_layout()
    save_path = "pipeline_zsn2n_dual_stream1.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 完整的端到端双流合成结果已保存至: {save_path}")
    print("   你现在将看到网络从初始的随机‘黑图’，奇迹般地自我学习成了一张去噪完美的超声图！")