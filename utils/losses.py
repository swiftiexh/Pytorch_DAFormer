# 损失相关的工具函数

import torch
import torch.nn as nn

# 下采样
def downscale_label_ratio(gt, scale_factor, min_ratio, n_classes, ignore_index=255):
    ignore_substitute = n_classes # 用于忽略标签的替代值
    # 克隆避免修改原始数据
    out = gt.clone()
    out[out == ignore_index] = ignore_substitute

    # One-hot 编码
    # (B,1,H,W) -> (B,H,W) - 对每个像素做独热编码 -> (B,H,W,n_classes+1) - 转置 -> (B,n_classes+1,H,W)
    out = nn.functional.one_hot(out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)

    # 平均池化下采样
    out = nn.functional.avg_pool2d(out.float(), kernel_size=scale_factor) # (B, n_classes+1, H/scale, W/scale)

    # gt_ritio：每个位置的最大占比值 [B,1,H,W]  主导类别 out [B,1,H,W]
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)

    # 恢复 ignore_index
    out[out == ignore_substitute] = ignore_index
    # 占比低于阈值的区域设为 ignore_index
    out[gt_ratio < min_ratio] = ignore_index

    return out
    
# 计算 thing class 上的特征距离
def masked_feature_distance(student_feat, teacher_feat, gt_seg, 
                            mask_classes=None, scale_min_ratio=0.75, 
                            num_classes=19, ignore_index=255):
    assert mask_classes is not None, "mask_classes must be specified（Thing Class）"
    # 创建掩码
    scale_factor = gt_seg.shape[-1] // student_feat.shape[-1] # 下采样比例
    # 将标签下采样到特征图尺寸
    gt_rescaled = downscale_label_ratio(
            gt_seg, 
            scale_factor=scale_factor,
            min_ratio=scale_min_ratio,
            n_classes=num_classes,
            ignore_index=ignore_index
        ).long().detach() # [B, 1, H/32, W/32]
    # 创建掩码：仅保留 mask_classes 中的类别
    fdclasses = torch.tensor(mask_classes, device=gt_seg.device, dtype=torch.long)
    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, dim=-1)  # [B, 1, H/32, W/32]
    # 计算特征距离（L2 范数）
    feat_diff = student_feat - teacher_feat  # [B, C, H/32, W/32]
    pw_feat_dist = torch.norm(feat_diff, p=2, dim=1) # [B, H/32, W/32]
    # 应用掩码
    masked_pw_feat_dist = pw_feat_dist[fdist_mask.squeeze(1)]  # [N]
    # 如果掩码区域为空，返回 0
    if fdist_mask.sum() == 0:
         return torch.tensor(0.0, device=student_feat.device, requires_grad=True)
    # 返回平均特征距离
    return torch.mean(masked_pw_feat_dist)