import numpy as np
import matplotlib.pyplot as plt

# 你保存的结果路径
result_path = "/data2/user24/cz/speckle2self-main/output_path1/results.npy"

# 加载数据
results = np.load(result_path)

# 打印基本信息（最重要！）
print("✅ 结果形状 (shape):", results.shape)
print("✅ 数据类型 (dtype):", results.dtype)
print("✅ 最小值:", results.min())
print("✅ 最大值:", results.max())

# 如果你想看第 0 张图
if len(results.shape) >= 3:
    img = results[0]  # 取第一张
    print("\n查看第 0 张图片，形状:", img.shape)

    # 保存成图片，不用弹窗
    plt.imsave("/data2/user24/cz/speckle2self-main/output_path1/result_1.png", img, cmap='gray')
    print("✅ 第 0 张图已保存为: result_0.png")