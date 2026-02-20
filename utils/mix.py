# 类别混合增强

import torch
import numpy as np
import torch.nn as nn
import kornia

# 生成指定类别的掩码
# 每个位置上，判断 label 是否属于 classes 中的某个类别
def generate_class_mask(label, classes):
    # label:[1,H,W] classes: [nclasses/2]
    label, classes = torch.broadcast_tensors(
        label,
        classes.unsqueeze(1).unsqueeze(2) # [nclasses/2,1,1]
    ) # 广播到相同尺寸：[nclasses/2,H,W]
    class_mask = label.eq(classes).sum(0, keepdims=True) # [1,H,W]
    return class_mask

# 为每个样本生成掩码
def get_class_masks(labels):
    class_masks = []
    # labels: [B,1,H,W]
    for label in labels:
        # label: [1,H,W]
        classes = torch.unique(label) # 获取图像中出现的类别 [nclasses]
        nclasses = classes.shape[0] # 类别数量
        # 随机选择一半的类别 (向上取整)
        class_choice = np.random.choice(
            nclasses, 
            int((nclasses + nclasses % 2) / 2),  # 选择一半,奇数时向上取整
            replace=False
        )
        classes = classes[torch.Tensor(class_choice).long()]

        # 生成类别掩码
        class_mask = generate_class_mask(label, classes).unsqueeze(0) # [1,1,H,W]
        class_masks.append(class_mask)

    return class_masks # List of [B tensors of shape [1,1,H,W]]

# 进行一次类别混合
def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    # mask: [1,1,H,W] -> 取出 [H,W]
    mask_2d = mask[0, 0]  # [H,W]
    if data is not None:
        # data: [2,3,H,W]
        mask_3d = mask_2d.unsqueeze(0)  # [1,H,W]
        data = (mask_3d * data[0] + (1 - mask_3d) * data[1]).unsqueeze(0)  # [1,3,H,W]
    if target is not None:
        # target: [2,H,W]
        target = (mask_2d * target[0] + (1 - mask_2d) * target[1]).unsqueeze(0)  # [1,H,W]
    return data, target

# 颜色抖动函数
def color_jitter(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.2):
    if data is None:
        return data, target
    if color_jitter < p: # 以概率 p 进行颜色抖动
        # 定义颜色抖动
        seq = nn.Sequential(
            kornia.augmentation.ColorJitter(
                brightness=s, contrast=s, saturation=s, hue=s
            )
        )
        # 反归一化
        data = data.mul(std).add(mean).div(255.0)
        # 应用抖动
        data = seq(data)
        # 重新归一化
        data = data.mul(255.0).sub(mean).div(std)
    return data, target

def gaussian_blur(blur, data=None, target=None):
    if data is None:
        return data, target
    if blur > 0.5:
        # 随机选择 sigma
        sigma = np.random.uniform(0.15, 1.15)
        # 根据图像大小计算 kernel size (约为图像尺寸的10%)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2
            )
        )
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2
            )
        )
        kernel_size = (kernel_size_y, kernel_size_x) 
        # 应用高斯模糊
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, 
                sigma=(sigma, sigma)
            )
        )
        data = seq(data)
    return data, target

# 强增强变换函数
def strong_transform(param, data=None, target=None):
    # 1. 类别混合
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    # 2. 颜色抖动
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target
    )
    # 3. 高斯模糊
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

# 混合增强函数
def apply_class_mix(src_img, src_seg, tgt_img, pseudo_label, 
                   pseudo_weight, mix_type, blur, 
                   color_jitter_s, color_jitter_p):
    assert mix_type == 'class'
    batch_size = src_img.shape[0]
    device = src_img.device

    # 1. 生成类别掩码 (每个样本随机选择一半类别)
    mix_masks = get_class_masks(src_seg)

    # 2. 对每个样本进行混合
    mixed_img_list = []
    mixed_lbl_list = []
    mixed_weight_list = []
    # 准备归一化参数 （同训练时的 norm_cfg）
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    for i in range(batch_size):
        strong_parameters = {
            'mix' : mix_masks[i], # [1,1,H,W]
            'color_jitter': np.random.uniform(0, 1),  # 随机颜色扰动强度
            'color_jitter_s': color_jitter_s, # 颜色扰动强度上限
            'color_jitter_p': color_jitter_p, # 颜色扰动概率
            'blur': np.random.uniform(0, 1) if blur else 0.0, # 随机模糊强度
            'mean': mean[0],
            'std': std[0]
        }
        # 混合图像和标签
        mixed_img_i, mixed_lbl_i = strong_transform(
            strong_parameters,
            data=torch.stack((src_img[i], tgt_img[i])), # [2,3,H,W]
            target=torch.stack((src_seg[i][0], pseudo_label[i])) # [2,H,W]
        ) # each: [1,3,H,W], [1,H,W]
        # 混合权重
        # 为源域 GT 创建全1权重
        gt_pixel_weight = torch.ones_like(pseudo_weight[i])
        _, mixed_weight_i = strong_transform(
            strong_parameters,
            target=torch.stack((gt_pixel_weight, pseudo_weight[i])) # [2,H,W]
        ) # each: [1,H,W]
        # 收集结果
        mixed_img_list.append(mixed_img_i)  # List of [B tensors of shape [1,3,H,W]]
        mixed_lbl_list.append(mixed_lbl_i)  # List of [B tensors of shape [1,H,W]]
        mixed_weight_list.append(mixed_weight_i)  # List of [B tensors of shape [1,H,W]]
    
    # 3. 拼接为 batch
    mixed_img = torch.cat(mixed_img_list, dim=0) # [B,3,H,W]
    mixed_lbl = torch.cat(mixed_lbl_list, dim=0) # [B,H,W]
    mixed_weight = torch.cat(mixed_weight_list, dim=0) # [B,H,W]
    
    return mixed_img, mixed_lbl, mixed_weight

