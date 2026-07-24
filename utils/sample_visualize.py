import torch
import matplotlib.pyplot as plt

def sample_timesteps_log_normal(
    batch_size: int,
    num_timesteps: int,
    mu: float = 1.5,
    sigma: float = 1.5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    根据类对数正态分布 (Log-Normal-inspired) 的策略采样时间步索引。

    这种方法会偏向性地采样索引，而不是纯粹的均匀随机。
    通过调整 mu 和 sigma，可以控制采样的中心和范围。

    参数:
        batch_size (int): 批量大小。
        num_timesteps (int): 总的时间步数量 (例如 len(scheduler.timesteps))。
        mu (float): 控制采样中心的参数。mu > 0 会使采样偏向末尾（低噪声）。
        sigma (float): 控制采样集中度的参数。值越小，采样越集中。
        device (str): 'cpu' 或 'cuda'。

    返回:
        torch.Tensor: 形状为 (batch_size,) 的时间步索引张量。
    """
    # 1. 从标准正态分布 N(0, 1) 中采样
    normal_samples = torch.randn(batch_size, device=device)

    # 2. 使用 mu 和 sigma 对正态样本进行缩放和移位
    transformed_samples = mu + sigma * normal_samples

    # 3. 使用 Sigmoid 函数将值域映射到 (0, 1) 区间
    #    这可以将无界的正态分布值转化为类似“百分比”的值
    p = torch.sigmoid(transformed_samples)

    # 4. 将 (0, 1) 区间的值缩放到总的时间步索引范围 [0, num_timesteps - 1]
    indices_float = p * (num_timesteps - 1)

    # 5. 转换为长整型，并使用 clamp 确保索引不会因浮点误差越界
    indices = torch.clamp(indices_float.long(), 0, num_timesteps - 1)

    return indices

# --- 如何在你的代码中使用 ---

# 假设这是你的设置
batch_size = 64
# self.pipeline.scheduler.timesteps 类似下面这个，通常是从大到小的
# 例如，对于 SD1.5，有 1000 个时间步
timesteps = torch.linspace(999, 0, 1000) 
num_timesteps = len(timesteps)
device = 'cpu' # 或者 'cuda'

# 你原来的代码：
# indices_uniform = torch.randint(0, num_timesteps, (batch_size,), device=device)

# 改造后的代码：
# 使用新策略生成索引
indices_new = sample_timesteps_log_normal(
    batch_size=batch_size,
    num_timesteps=num_timesteps,
    mu=1.5,       # 设置为正数，让采样偏向列表的末尾（低噪声区域）
    sigma=1.5,    # 控制分布的离散程度
    device=device
)


# --- (可选) 可视化对比两种采样策略的差异 ---
print("你的原始代码行:")
print(f"indices = torch.randint(0, {num_timesteps}, ({batch_size},), device='{device}')\n")

print("替换为:")
print(f"indices = sample_timesteps_log_normal(batch_size={batch_size}, num_timesteps={num_timesteps}, mu=1.5, sigma=1.5, device='{device}')\n")

# 生成大量样本以供可视化
large_batch = 10000
indices_uniform = torch.randint(0, num_timesteps, (large_batch,), device=device)
indices_new_visual = sample_timesteps_log_normal(large_batch, num_timesteps, mu=-0.5, sigma=1, device=device)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(indices_uniform.cpu().numpy(), bins=50, color='blue', alpha=0.7)
plt.title('Original Uniform Sampling')
plt.xlabel('Timestep Index')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(indices_new_visual.cpu().numpy(), bins=50, color='green', alpha=0.7)
plt.title('New Biased Sampling (mu=1.5, sigma=1.5)')
plt.xlabel('Timestep Index')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('sampling.png')