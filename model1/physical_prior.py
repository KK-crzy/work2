import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class GlobalPhysicalPriorExtractor:
    """全局物理统计先验提取模块
    从单张超声B-mode图像中提取三个物理启发的统计先验：
    1. 相对衰减先验图 A(x,y)：表示当前位置高频信号的保留能力
    2. 局部散射统计先验图 S(x,y)：表示当前位置的散射特性（粗纹理/细散斑）
    3. 组织结构响应先验图 T(x,y)：表示当前位置是否属于稳定的组织结构
    """
    
    def __init__(
        self,
        window_size: int = 7,
        device: Optional[torch.device] = None,
        # 先验融合权重（根据临床数据优化）
        scattering_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        structure_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    ):
        """
        Args:
            window_size: 局部统计计算的窗口大小
            device: 计算设备，None则自动选择
            scattering_weights: 局部散射先验的三个特征权重
            structure_weights: 组织结构先验的三个特征权重
        """
        self.window_size = window_size
        self.padding = window_size // 2
        self.scattering_weights = scattering_weights
        self.structure_weights = structure_weights
        
        # 自动选择设备
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 预定义Sobel和Laplacian核
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

    def read_image(self, image_path: str) -> torch.Tensor:
        """读取超声图像，支持普通图像格式和DICOM格式
        Args:
            image_path: 图像文件路径
        Returns:
            归一化到[0,1]的灰度图像张量，shape: (1, 1, H, W)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # 读取DICOM格式
        if file_ext in ['.dcm', '.dicom']:
            ds = pydicom.dcmread(image_path)
            # 应用DICOM的rescale参数
            img = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            # 转换为灰度图并归一化
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
        # 读取普通图像格式
        else:
            img = Image.open(image_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0
        
        # 转换为PyTorch张量并添加batch和channel维度
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        return img_tensor

    def compute_relative_attenuation_prior(self, img: torch.Tensor) -> torch.Tensor:
        """计算相对衰减先验图 A(x,y)
        物理依据：超声高频分量随传播深度呈指数衰减
        Args:
            img: 输入超声图像，shape: (1, 1, H, W)
        Returns:
            相对衰减先验图，shape: (1, 1, H, W)，值∈[0,1]
        """
        # 1. 提取高频分量
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        high_freq_energy = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 2. 计算纵向（深度方向）平均高频能量曲线
        H, W = img.shape[2], img.shape[3]
        row_energy = torch.mean(high_freq_energy, dim=(1, 3)).squeeze()  # shape: (H,)
        
        # 3. 指数拟合衰减曲线
        y = torch.arange(H, dtype=torch.float32, device=self.device)
        # 避免log(0)
        row_energy_safe = row_energy + 1e-8
        log_energy = torch.log(row_energy_safe)
        
        # 最小二乘法拟合指数曲线 y = E0 * exp(-α * x)
        # 转换为线性拟合: log(y) = log(E0) - α * x
        X = torch.stack([y, torch.ones_like(y)], dim=1)
        coeffs = torch.linalg.lstsq(X, log_energy).solution
        alpha = -coeffs[0]
        E0 = torch.exp(coeffs[1])
        
        # 4. 计算每个像素的相对衰减值
        expected_energy = E0 * torch.exp(-alpha * y.view(-1, 1))
        attenuation = 1.0 - (high_freq_energy / (expected_energy.unsqueeze(0).unsqueeze(0) + 1e-8))
        
        # 5. 归一化到[0,1]并裁剪异常值
        attenuation = (attenuation - attenuation.min()) / (attenuation.max() - attenuation.min() + 1e-8)
        attenuation = torch.clamp(attenuation, 0.0, 1.0)
        
        return attenuation

    def compute_local_scattering_prior(self, img: torch.Tensor) -> torch.Tensor:
        """计算局部散射统计先验图 S(x,y)
        物理依据：不同尺寸的散射体产生不同的纹理特性
        Args:
            img: 输入超声图像，shape: (1, 1, H, W)
        Returns:
            局部散射统计先验图，shape: (1, 1, H, W)，值∈[0,1]
        """
        B, C, H, W = img.shape
        pad = self.padding
        
        # 1. 计算局部方差
        local_mean = F.avg_pool2d(img, self.window_size, stride=1, padding=pad)
        local_sq_mean = F.avg_pool2d(img**2, self.window_size, stride=1, padding=pad)
        local_var = local_sq_mean - local_mean**2
        
        # 2. 计算局部散斑密度（局部极大值数量）
        # 3x3窗口内的局部极大值
        max_filter = F.max_pool2d(img, 3, stride=1, padding=1)
        local_max = (img == max_filter).float()
        speckle_density = F.avg_pool2d(local_max, self.window_size, stride=1, padding=pad)
        
        # 3. 计算局部高频能量占比
        # 对每个局部窗口做FFT
        # 先将图像展开为窗口块
        patches = F.unfold(img, self.window_size, stride=1, padding=pad)  # (1, Ws*Ws, N)
        patches = patches.view(1, self.window_size, self.window_size, -1)  # (1, Ws, Ws, N)
        patches = patches.permute(0, 3, 1, 2)  # (1, N, Ws, Ws)
        
        # 计算每个窗口的FFT
        patches_fft = torch.fft.rfft2(patches)
        patches_fft_abs = torch.abs(patches_fft)
        
        # 计算高频能量占比（上半部分频率）
        h, w = patches_fft_abs.shape[2], patches_fft_abs.shape[3]
        high_freq_mask = torch.zeros_like(patches_fft_abs)
        high_freq_mask[:, :, h//2:, :] = 1.0
        
        total_energy = torch.sum(patches_fft_abs, dim=(2, 3))
        high_energy = torch.sum(patches_fft_abs * high_freq_mask, dim=(2, 3))
        high_freq_ratio = high_energy / (total_energy + 1e-8)
        
        # 重塑为图像尺寸
        high_freq_ratio = high_freq_ratio.view(1, 1, H, W)
        
        # 4. 归一化三个特征
        def normalize(tensor):
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        local_var_norm = normalize(local_var)
        speckle_density_norm = normalize(speckle_density)
        high_freq_ratio_norm = normalize(high_freq_ratio)
        
        # 5. 加权融合得到最终散射先验
        scattering_prior = (
            self.scattering_weights[0] * local_var_norm +
            self.scattering_weights[1] * speckle_density_norm +
            self.scattering_weights[2] * high_freq_ratio_norm
        )
        
        # 应用Sigmoid确保在[0,1]区间
        scattering_prior = torch.sigmoid(5 * (scattering_prior - 0.5))
        
        return scattering_prior

    def compute_structure_response_prior(self, img: torch.Tensor) -> torch.Tensor:
        """计算组织结构响应先验图 T(x,y)
        物理依据：稳定的组织结构具有清晰的边缘和一致的梯度方向
        Args:
            img: 输入超声图像，shape: (1, 1, H, W)
        Returns:
            组织结构响应先验图，shape: (1, 1, H, W)，值∈[0,1]
        """
        # 1. 计算梯度幅值
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 2. 计算Laplacian二阶导数（边缘锐度）
        laplacian = F.conv2d(img, self.laplacian, padding=1)
        laplacian_abs = torch.abs(laplacian)
        
        # 3. 计算结构张量的特征值比（各向异性）
        # 计算梯度外积
        IxIx = grad_x**2
        IxIy = grad_x * grad_y
        IyIy = grad_y**2
        
        # 高斯平滑结构张量
        # sigma = 1.0
        # kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        # IxIx_smooth = F.gaussian_blur(IxIx, kernel_size, sigma)
        # IxIy_smooth = F.gaussian_blur(IxIy, kernel_size, sigma)
        # IyIy_smooth = F.gaussian_blur(IyIy, kernel_size, sigma)
        sigma = 1.0
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)

        # 构造Gaussian Kernel
        coords = torch.arange(kernel_size, device=self.device).float()
        coords -= kernel_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # 2D Gaussian Kernel
        gaussian_kernel = torch.outer(g, g)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

        # Gaussian smoothing
        IxIx_smooth = F.conv2d(
            IxIx,
            gaussian_kernel,
            padding=kernel_size // 2
        )

        IxIy_smooth = F.conv2d(
            IxIy,
            gaussian_kernel,
            padding=kernel_size // 2
        )

        IyIy_smooth = F.conv2d(
            IyIy,
            gaussian_kernel,
            padding=kernel_size // 2
        )
        
        # 计算每个像素的结构张量特征值
        # 结构张量: [[IxIx, IxIy], [IxIy, IyIy]]
        # 特征值: λ1, λ2 = (trace ± sqrt(trace² - 4*det))/2
        trace = IxIx_smooth + IyIy_smooth
        det = IxIx_smooth * IyIy_smooth - IxIy_smooth**2
        sqrt_term = torch.sqrt(trace**2 - 4 * det + 1e-8)
        
        lambda1 = (trace + sqrt_term) / 2
        lambda2 = (trace - sqrt_term) / 2
        
        # 特征值比: 各向异性程度，值越大表示结构越明显
        anisotropy = lambda1 / (lambda2 + 1e-8)
        
        # 4. 归一化三个特征
        def normalize(tensor):
            return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        gradient_magnitude_norm = normalize(gradient_magnitude)
        laplacian_abs_norm = normalize(laplacian_abs)
        anisotropy_norm = normalize(anisotropy)
        
        # 5. 加权融合得到最终结构先验
        structure_prior = (
            self.structure_weights[0] * gradient_magnitude_norm +
            self.structure_weights[1] * laplacian_abs_norm +
            self.structure_weights[2] * anisotropy_norm
        )
        
        # 应用Sigmoid确保在[0,1]区间
        structure_prior = torch.sigmoid(5 * (structure_prior - 0.5))
        
        return structure_prior

    def extract_all_priors(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取所有三个物理统计先验
        Args:
            img: 输入超声图像，shape: (1, 1, H, W)
        Returns:
            A: 相对衰减先验图，shape: (1, 1, H, W)
            S: 局部散射统计先验图，shape: (1, 1, H, W)
            T: 组织结构响应先验图，shape: (1, 1, H, W)
        """
        A = self.compute_relative_attenuation_prior(img)
        S = self.compute_local_scattering_prior(img)
        T = self.compute_structure_response_prior(img)
        return A, S, T

    def visualize_priors(
        self,
        img: torch.Tensor,
        A: torch.Tensor,
        S: torch.Tensor,
        T: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """可视化原始图像和三个先验图
        Args:
            img: 原始超声图像
            A: 相对衰减先验图
            S: 局部散射统计先验图
            T: 组织结构响应先验图
            save_path: 保存路径，None则显示图像
        """
        # 转换为numpy数组
        img_np = img.squeeze().cpu().numpy()
        A_np = A.squeeze().cpu().numpy()
        S_np = S.squeeze().cpu().numpy()
        T_np = T.squeeze().cpu().numpy()
        
        # 创建子图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original Ultrasound Image')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(A_np, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Relative Attenuation Prior (A)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(S_np, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Local Scattering Prior (S)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        im3 = axes[3].imshow(T_np, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title('Structure Response Prior (T)')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"先验可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


# ------------------------------
# 示例用法
# ------------------------------
if __name__ == "__main__":
    # 初始化先验提取器
    extractor = GlobalPhysicalPriorExtractor(window_size=7)
    
    # 示例1：读取普通图像并提取先验
    image_path = "/data2/user24/cz/speckle2self-main/model1/result_0.png"  # 替换为你的图像路径
    try:
        # 读取图像
        img = extractor.read_image(image_path)
        print(f"成功读取图像，尺寸: {img.shape}")
        
        # 提取所有先验
        A, S, T = extractor.extract_all_priors(img)
        print(f"先验提取完成")
        print(f"相对衰减先验范围: [{A.min().item():.4f}, {A.max().item():.4f}]")
        print(f"局部散射先验范围: [{S.min().item():.4f}, {S.max().item():.4f}]")
        print(f"组织结构先验范围: [{T.min().item():.4f}, {T.max().item():.4f}]")
        
        # 可视化并保存结果
        extractor.visualize_priors(img, A, S, T, save_path="/data2/user24/cz/speckle2self-main/model1/physical_prior/priors_visualization.png")
        
        # 保存先验为numpy数组（用于后续训练）
        np.savez(
            "extracted_priors.npz",
            attenuation=A.cpu().numpy(),
            scattering=S.cpu().numpy(),
            structure=T.cpu().numpy()
        )
        print("先验数据已保存到: extracted_priors.npz")
        
    except Exception as e:
        print(f"处理出错: {str(e)}")