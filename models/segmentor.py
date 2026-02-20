# 学生模型

import torch.nn as nn
import torch

class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, num_classes=19, align_corners=False):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.num_classes = num_classes
        self.align_corners = align_corners
    
    # 训练前向传播，返回损失字典
    def forward_train(self, 
                     img, 
                     gt_semantic_seg, 
                     seg_weight=None, 
                     return_feat=False):
        losses = {}
        # 提取特征
        features = self.backbone(img)
        # 解码头前向传播
        seg_logits = self.decode_head(features)
        # 计算损失
        loss_dict = self._compute_loss(seg_logits, gt_semantic_seg, seg_weight)
        losses.update(loss_dict)
        if return_feat:
            losses['features'] = features
        return losses
    
    # 推理前向传播
    def encode_decode(self, img):
        # 提取多尺度特征
        features = self.backbone(img)
        # 解码头前向
        seg_logits = self.decode_head(features)  # (B, num_classes, H/4, W/4)
        # 3. 上采样到输入尺寸
        seg_logits = torch.nn.functional.interpolate(
            input=seg_logits,
            size=img.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=self.align_corners
        )
        return seg_logits
    
    # 计算损失
    def _compute_loss(self, seg_logits, gt_semantic_seg, seg_weight=None):
        loss_dict = {}
        # 上采样到标签尺寸
        seg_logits = torch.nn.functional.interpolate(
            input=seg_logits,
            size=gt_semantic_seg.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=self.align_corners
        )   
        # 去掉多余的维度
        gt_semantic_seg = gt_semantic_seg.squeeze(1).long()
        # 计算交叉熵损失
        if seg_weight is not None:
            loss_seg = torch.nn.functional.cross_entropy(
                seg_logits,  # (B, C, H, W)
                gt_semantic_seg,  # (B, H, W)
                ignore_index=255,
                reduction='none'  # 不进行自动归约
            )   # (B, H, W)
            # 应用权重加权平均
            loss_seg = (loss_seg * seg_weight).sum() / (seg_weight.sum() + 1e-8)
        else:
            loss_seg = torch.nn.functional.cross_entropy(
                seg_logits,
                gt_semantic_seg,
                ignore_index=255,
                reduction='mean'
            )
        loss_dict['loss_seg'] = loss_seg
        # 计算精度
        with torch.no_grad():
            seg_pred = seg_logits.argmax(dim=1)  # (B, H, W)
            valid_mask = (gt_semantic_seg != 255)
            if valid_mask.sum() > 0:
                acc_seg = (seg_pred == gt_semantic_seg)[valid_mask].float().mean()
            else:
                acc_seg = torch.tensor(0.0, device=seg_logits.device)
            loss_dict['acc_seg'] = acc_seg
        return loss_dict


# 根据配置构建分割模型
def build_segmentor(model_cfg):
    from models.backbones.mit_b5 import mit_b5
    from models.decode_heads.daformer_head import DAFormerHead

    # 1. 构建 backbone
    pretrained = model_cfg.get('pretrained', None)
    backbone = mit_b5(pretrained=pretrained)

    # 2. 构建 decode_head
    head_cfg = model_cfg['decode_head']
    decode_head = DAFormerHead(
        in_channels=head_cfg['in_channels'],
        in_index=head_cfg['in_index'],
        channels=head_cfg['channels'],
        dropout_ratio=head_cfg['dropout_ratio'],
        num_classes=head_cfg['num_classes'],
        norm_cfg=head_cfg['norm_cfg'],
        align_corners=head_cfg['align_corners'],
        decoder_params=head_cfg['decoder_params']
    )

    # 3. 组合为完整模型
    model = Segmentor(
        backbone=backbone,
        decode_head=decode_head,
        num_classes=head_cfg['num_classes'],
        align_corners=head_cfg['align_corners']
    )

    # 4. 初始化 backbone 权重（如果有预训练权重）
    if pretrained is not None:
        backbone.init_weights()
    
    return model