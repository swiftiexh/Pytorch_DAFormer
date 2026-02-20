# 组装模块（数据、模型、优化器）、控制训练流程（迭代、日志、保存）

import importlib.util
import sys
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import os
import glob
from utils.logger import JSONLogger
import json
from datetime import datetime

# 设置随机种子以确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 加载配置模块
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path) # 创建模块描述符
    cfg = importlib.util.module_from_spec(spec) # 创建模块对象
    sys.modules["config"] = cfg # 将模块添加到 sys.modules 中，使其可被导入
    spec.loader.exec_module(cfg) # 执行模块代码，加载配置
    return cfg

# 根据配置列表构建 pipeline
def build_pipeline(pipeline_cfg_list):
    from utils import transform as transforms
    pipeline = []
    for cfg in pipeline_cfg_list:
        cfg = cfg.copy()
        transform_type = cfg.pop('type')
        # 动态获取 transform 类
        transform_cls = getattr(transforms, transform_type)
        pipeline.append(transform_cls(**cfg))
    return pipeline

# 构建数据集
def build_dataset(data_cfg):
    from datasets.gta import GTADataset
    from datasets.cityscapes import CityscapesDataset
    from datasets.uda_dataset import UDADataset

    # 构建源域数据集
    source_cfg = data_cfg['train']['source']
    source_pipeline = build_pipeline(source_cfg['pipeline'])
    source_dataset = GTADataset(
        data_root=source_cfg['data_root'],
        img_dir=source_cfg['img_dir'],
        ann_dir=source_cfg['ann_dir'],
        pipeline=source_pipeline
    )

    # 构建目标域数据集
    target_cfg = data_cfg['train']['target']
    target_pipeline = build_pipeline(target_cfg['pipeline'])
    target_dataset = CityscapesDataset(
        data_root=target_cfg['data_root'],
        img_dir=target_cfg['img_dir'],
        ann_dir=target_cfg['ann_dir'],
        pipeline=target_pipeline
    )

    # 封装为 UDA 数据集
    uda_dataset = UDADataset(
        source=source_dataset,
        target=target_dataset,
        rare_class_sampling=data_cfg['train'].get('rare_class_sampling')
    )

    return uda_dataset

# 构建 DataLoader
def build_dataloader(dataset, data_cfg):
    return DataLoader(
        dataset,
        batch_size=data_cfg['samples_per_gpu'],
        num_workers=data_cfg['workers_per_gpu'],
        shuffle=True,
        pin_memory=True, # 加速数据传输到 GPU
        drop_last=True # 丢弃最后一个不完整的 batch
    )

# 构建模型
def build_model(model_cfg):
    from models.segmentor import build_segmentor
    model = build_segmentor(model_cfg)  
    return model


# 构建优化器
def build_optimizer(model, optim_cfg):
    import torch.optim as optim
    # 提取配置
    optim_type = optim_cfg.get('type', 'AdamW')
    base_lr = optim_cfg.get('lr', 6e-5)
    weight_decay = optim_cfg.get('weight_decay', 0.01)
    betas = optim_cfg.get('betas', (0.9, 0.999))
    # 参数分组配置
    paramwise_cfg = optim_cfg.get('paramwise_cfg', {})
    custom_keys = paramwise_cfg.get('custom_keys', {})
    # 参数分组：根据名称匹配规则
    params = []
    # 遍历所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 默认配置
        param_group = {
            'params': [param],
            'lr': base_lr,
            'weight_decay': weight_decay
        }
        # 检查是否匹配自定义规则
        for key, config in custom_keys.items():
            if key in name:
                # 应用学习率倍率
                if 'lr_mult' in config:
                    param_group['lr'] = base_lr * config['lr_mult']
                # 应用权重衰减倍率
                if 'decay_mult' in config:
                    param_group['weight_decay'] = weight_decay * config['decay_mult']
                # 找到第一个匹配的规则后跳出
                # 优先级：head > pos_block > norm
                break
        params.append(param_group)
    # 创建优化器
    optimizer_cls = getattr(optim, optim_type)
    optimizer = optimizer_cls(params, betas=betas)
    return optimizer

# 构建学习率调度器
def build_lr_scheduler(optimizer, lr_config):
    from utils.lr_scheduler import PolyLRWithWarmup
    return PolyLRWithWarmup(
            optimizer=optimizer,
            max_iters=lr_config['max_iters'],
            warmup_iters=lr_config.get('warmup_iters', 1500),
            warmup_ratio=lr_config.get('warmup_ratio', 1e-6),
            power=lr_config.get('power', 1.0),
            min_lr=lr_config.get('min_lr', 0.0)
        )

# 验证函数
def validate(model, val_loader, num_classes=19):
    model.eval()
    # 初始化混淆矩阵：用于计算 IoU
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    from tqdm import tqdm 
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'): 
            img = batch['img'].cuda()
            gt_seg = batch['gt_semantic_seg'].cuda()  # [B, 1, H, W]
            # 推理
            pred = model.encode_decode(img)  # [B, num_classes, H, W]
            pred = pred.argmax(dim=1)  # [B, H, W]
            # 展平
            gt_seg = gt_seg.squeeze(1).flatten()  # [B*H*W]
            pred = pred.flatten()  # [B*H*W]
            # 忽略 ignore_index=255 的像素
            valid_mask = (gt_seg != 255)
            gt_seg = gt_seg[valid_mask]
            pred = pred[valid_mask]
            # 更新混淆矩阵
            for t, p in zip(gt_seg, pred):
                confusion_matrix[t.long(), p.long()] += 1
    # 计算 IoU
    iou_per_class = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-10)
        iou_per_class.append(iou.item())
    mean_iou = sum(iou_per_class) / num_classes
    model.train()
    return {
        'mIoU': mean_iou,
        'IoU_per_class': iou_per_class
    }

# 构建验证集
def build_val_dataset(data_cfg):
    from datasets.cityscapes import CityscapesDataset
    
    val_cfg = data_cfg['val']
    val_pipeline = build_pipeline(val_cfg['pipeline'])
    val_dataset = CityscapesDataset(
        data_root=val_cfg['data_root'],
        img_dir=val_cfg['img_dir'],
        ann_dir=val_cfg['ann_dir'],
        pipeline=val_pipeline
    )
    return val_dataset

# 写入JSON日志
def write_log(log_path, log_dict):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_dict) + '\n')

# 保存 checkpoint
def save_checkpoint(model, optimizer, lr_scheduler, ema_model, iteration, checkpoint_dir, max_keep_ckpts=1):
    checkpoint_path = os.path.join(checkpoint_dir, f'iter_{iteration}.pth')
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'ema_model': ema_model.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    # 保留最新的 max_keep_ckpts 个
    if max_keep_ckpts > 0:
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'iter_*.pth')))
        if len(checkpoint_files) > max_keep_ckpts:
            for old_ckpt in checkpoint_files[:-max_keep_ckpts]:
                os.remove(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")


def main():
    # Cityscapes 19类别名称
    CITYSCAPES_CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    # 1. 加载配置
    cfg = load_config('configs/gta2cs_daformer.py')
    # 设置随机种子
    set_seed(cfg.seed)
    # 如果启用 cudnn benchmark，则设置
    if getattr(cfg, 'cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 2. 构建数据集和 DataLoader
    print("Building dataset...")
    train_dataset = build_dataset(cfg.data)
    train_loader = build_dataloader(train_dataset, cfg.data)
    print("Building validation dataset...")
    val_dataset = build_val_dataset(cfg.data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时 batch_size=1
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=False,
        pin_memory=True
    )
    # 3. 构建模型
    print("Building model...")
    model = build_model(cfg.model)
    model = model.cuda() # 将模型移到 GPU 上
    # 4. 构建优化器和学习率调度器
    optimizer = build_optimizer(model, cfg.optim)
    lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_schedule)
    # 5. 构建 Trainer（封装 UDA 逻辑）
    from trainer.train_daformer import DAFormerTrainer
    trainer = DAFormerTrainer(
        model=model,
        uda_cfg=cfg.uda, # UDA 相关配置，如损失权重、伪标签更新频率等
        optimizer=optimizer,
        lr_schedule=lr_scheduler, # 学习率调度配置，如 warmup、step decay 等
        runner_cfg=cfg.runner # 训练流程配置
    )
    # 6. 训练循环
    # 创建 checkpoint 和 log 目录
    checkpoint_dir = f'work_dirs/{cfg.name}'
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # 初始化日志记录器
    logger = JSONLogger(log_dir, cfg.name)
    # 记录配置信息
    logger.write_config({
        'max_iters': cfg.runner['max_iters'],
        'batch_size': cfg.data['samples_per_gpu'],
        'base_lr': cfg.optim['lr'],
        'weight_decay': cfg.optim.get('weight_decay', 0.01)
    })
    print("Start training...")
    from itertools import cycle
    train_loader_iter = cycle(train_loader) 
    for iteration in range(1, cfg.runner['max_iters'] + 1):
        # 获取一个 batch
        batch = next(train_loader_iter)
        # 将数据移到 GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        # 执行一步训练
        log_vars = trainer.train_step(batch, iteration)
        # 打印日志并记录到文件
        if iteration % cfg.log['interval'] == 0:
            log_str = f"Iter [{iteration}/{cfg.runner['max_iters']}]"
            for key, val in log_vars.items():
                # 提取 tensor 的标量值
                if isinstance(val, torch.Tensor):
                    val_scalar = val.item()
                else:
                    val_scalar = val
                log_str += f" {key}: {val_scalar:.4f}"
            print(log_str)
            # 使用logger记录
            logger.log_train(iteration, log_vars)
        # 验证
        if iteration % cfg.evaluation['interval'] == 0 and iteration > 0:
            print(f"\nEvaluating at iter {iteration}...")
            eval_results = validate(model, val_loader, num_classes=19)
            print(f"mIoU: {eval_results['mIoU']:.4f}")
            for i, iou in enumerate(eval_results['IoU_per_class']):
                print(f"{CITYSCAPES_CLASSES[i]:15s}: {iou:.4f}")
            # 使用logger记录验证结果
            logger.log_val(iteration, eval_results)
            print()  
        # 保存 checkpoint
        if iteration % cfg.checkpoint_config['interval'] == 0 and iteration > 0:
            print(f"\nSaving checkpoint at iter {iteration}...")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema_model=trainer.ema_model,
                iteration=iteration,
                checkpoint_dir=checkpoint_dir,
                max_keep_ckpts=cfg.checkpoint_config['max_keep_ckpts']
            )
            print() 

if __name__ == '__main__':
    main()