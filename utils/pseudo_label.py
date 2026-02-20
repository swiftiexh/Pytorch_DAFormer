# 伪标签生成

import torch
import numpy as np

def generate_pseudo_label(logits, threshold, ignore_top, ignore_bottom):
    device = logits.device

    # 1. Softmax 得到概率分布
    ema_softmax = torch.softmax(logits.detach(), dim=1)  # [B, C, H, W]

    # 2. 取最大概率及其对应的类别
    pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)  # [B, H, W]
    
    # 3. 生成置信度掩码: 概率 >= threshold 的像素
    ps_large_p = pseudo_prob.ge(threshold).long() == 1  # [B, H, W], bool
    
    # 4. 计算高置信度像素的比例 (用于统计)
    ps_size = np.size(np.array(pseudo_label.cpu()))  # 总像素数
    pseudo_ratio = torch.sum(ps_large_p).item() / ps_size
    
    # 5. 初始化权重: 所有高置信度像素权重为 pseudo_ratio
    # 这里权重值是统一的 pseudo_ratio,而不是每个像素的实际概率
    pseudo_weight = pseudo_ratio * torch.ones(
        pseudo_prob.shape, device=device
    )  # [B, H, W]
    
    # 6. 将低置信度像素的权重设为 0
    pseudo_weight[~ps_large_p] = 0
    
    # 7. 忽略顶部像素 (天空、建筑顶部等容易有伪影)
    if ignore_top > 0:
        pseudo_weight[:, :ignore_top, :] = 0
    
    # 8. 忽略底部像素 (车辆引擎盖、道路底部等容易有伪影)
    if ignore_bottom > 0:
        pseudo_weight[:, -ignore_bottom:, :] = 0
    
    return pseudo_label, pseudo_weight