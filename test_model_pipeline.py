import torch
import sys
import os
import traceback

def print_section(title):
    """打印分隔线"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_success(msg):
    """打印成功信息"""
    print(f"✓ {msg}")

def print_error(msg):
    """打印错误信息"""
    print(f"✗ {msg}")

def print_info(msg, indent=1):
    """打印信息"""
    print("  " * indent + f"- {msg}")


# ============================================================================
# 测试 1: 配置文件加载
# ============================================================================
def test_config_loading():
    print_section("测试 1: 配置文件加载")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "configs/gta2cs_daformer.py")
        cfg = importlib.util.module_from_spec(spec)
        sys.modules["config"] = cfg
        spec.loader.exec_module(cfg)
        
        print_success("配置文件加载成功")
        print_info(f"Batch size: {cfg.data['samples_per_gpu']}")
        print_info(f"Crop size: {cfg.crop_size}")
        print_info(f"Num classes: {cfg.model['decode_head']['num_classes']}")
        print_info(f"Pretrained: {cfg.model.get('pretrained', 'None')}")
        print_info(f"Max iters: {cfg.runner['max_iters']}")
        
        return cfg, None
    except Exception as e:
        print_error(f"配置加载失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 2: Transform Pipeline
# ============================================================================
def test_transform_pipeline(cfg):
    print_section("测试 2: Transform Pipeline")
    
    try:
        from utils import transform as transforms
        
        # 测试构建 pipeline
        def build_pipeline(pipeline_cfg_list):
            pipeline = []
            for cfg_item in pipeline_cfg_list:
                cfg_item = cfg_item.copy()
                transform_type = cfg_item.pop('type')
                transform_cls = getattr(transforms, transform_type)
                pipeline.append(transform_cls(**cfg_item))
            return pipeline
        
        gta_pipeline = build_pipeline(cfg.gta_train_pipeline)
        cs_pipeline = build_pipeline(cfg.cityscapes_train_pipeline)
        
        print_success("Transform pipeline 构建成功")
        print_info(f"GTA pipeline steps: {len(gta_pipeline)}")
        print_info(f"Cityscapes pipeline steps: {len(cs_pipeline)}")
        
        return (gta_pipeline, cs_pipeline), None
    except Exception as e:
        print_error(f"Transform pipeline 构建失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 3: 数据集加载
# ============================================================================
def test_datasets(cfg, pipelines):
    print_section("测试 3: 数据集加载")
    
    gta_pipeline, cs_pipeline = pipelines
    
    try:
        from datasets.gta import GTADataset
        from datasets.cityscapes import CityscapesDataset
        
        # 测试 GTA 数据集
        gta_cfg = cfg.data['train']['source']
        gta_dataset = GTADataset(
            data_root=gta_cfg['data_root'],
            img_dir=gta_cfg['img_dir'],
            ann_dir=gta_cfg['ann_dir'],
            pipeline=gta_pipeline
        )
        print_success(f"GTA 数据集加载成功")
        print_info(f"样本数: {len(gta_dataset)}")
        
        # 测试 Cityscapes 数据集
        cs_cfg = cfg.data['train']['target']
        cs_dataset = CityscapesDataset(
            data_root=cs_cfg['data_root'],
            img_dir=cs_cfg['img_dir'],
            ann_dir=cs_cfg['ann_dir'],
            pipeline=cs_pipeline
        )
        print_success(f"Cityscapes 数据集加载成功")
        print_info(f"样本数: {len(cs_dataset)}")
        
        # 测试单个样本
        print("\n  测试数据格式...")
        gta_sample = gta_dataset[0]
        print_info(f"GTA sample keys: {list(gta_sample.keys())}")
        print_info(f"img shape: {gta_sample['img'].shape}", indent=2)
        print_info(f"gt_semantic_seg shape: {gta_sample['gt_semantic_seg'].shape}", indent=2)
        
        cs_sample = cs_dataset[0]
        print_info(f"Cityscapes sample keys: {list(cs_sample.keys())}")
        print_info(f"img shape: {cs_sample['img'].shape}", indent=2)
        print_info(f"gt_semantic_seg shape: {cs_sample['gt_semantic_seg'].shape}", indent=2)
        
        return (gta_dataset, cs_dataset), None
    except Exception as e:
        print_error(f"数据集加载失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 4: UDA Dataset
# ============================================================================
def test_uda_dataset(cfg, datasets):
    print_section("测试 4: UDA Dataset")
    
    gta_dataset, cs_dataset = datasets
    
    try:
        from datasets.uda_dataset import UDADataset
        
        uda_dataset = UDADataset(
            source=gta_dataset,
            target=cs_dataset,
            rare_class_sampling=cfg.data['train'].get('rare_class_sampling')
        )
        
        print_success("UDA 数据集构建成功")
        print_info(f"总样本数: {len(uda_dataset)}")
        
        # 测试样本
        sample = uda_dataset[0]
        print_info(f"Sample keys: {list(sample.keys())}")
        print_info(f"img (source): {sample['img'].shape}", indent=2)
        print_info(f"gt_semantic_seg: {sample['gt_semantic_seg'].shape}", indent=2)
        print_info(f"target_img: {sample['target_img'].shape}", indent=2)
        
        return uda_dataset, None
    except Exception as e:
        print_error(f"UDA 数据集构建失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 5: DataLoader
# ============================================================================
def test_dataloader(cfg, uda_dataset):
    print_section("测试 5: DataLoader")
    
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            uda_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=2,  # 减少 workers 避免问题
            shuffle=True,
            pin_memory=False,  # 先不用 pin_memory
            drop_last=True
        )
        
        print_success("DataLoader 创建成功")
        print_info(f"Batch size: {cfg.data['samples_per_gpu']}")
        
        # 获取一个 batch
        print("\n  加载一个 batch...")
        batch = next(iter(dataloader))
        
        print_success("Batch 加载成功")
        print_info(f"Batch keys: {list(batch.keys())}")
        print_info(f"img: {batch['img'].shape}", indent=2)
        print_info(f"gt_semantic_seg: {batch['gt_semantic_seg'].shape}", indent=2)
        print_info(f"target_img: {batch['target_img'].shape}", indent=2)
        
        return batch, None
    except Exception as e:
        print_error(f"DataLoader 测试失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 6: MiT-B5 Backbone
# ============================================================================
def test_backbone():
    print_section("测试 6: MiT-B5 Backbone")
    
    try:
        from models.backbones.mit_b5 import mit_b5
        
        backbone = mit_b5(pretrained=None)
        backbone.eval()
        
        print_success("Backbone 构建成功")
        
        # 测试前向传播
        print("\n  测试前向传播...")
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features = backbone(x)
        
        print_success("前向传播成功")
        print_info(f"输入: {x.shape}")
        print_info(f"输出特征数: {len(features)}")
        for i, feat in enumerate(features):
            print_info(f"Stage {i+1}: {feat.shape}", indent=2)
        
        return backbone, None
    except Exception as e:
        print_error(f"Backbone 测试失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 7: DAFormer Decoder Head
# ============================================================================
def test_decoder_head(cfg):
    print_section("测试 7: DAFormer Decoder Head")
    
    try:
        from models.decode_heads.daformer_head import DAFormerHead
        
        head_cfg = cfg.model['decode_head']
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
        decode_head.eval()
        
        print_success("Decoder Head 构建成功")
        
        # 测试前向传播
        print("\n  测试前向传播...")
        features = [
            torch.randn(2, 64, 128, 128),
            torch.randn(2, 128, 64, 64),
            torch.randn(2, 320, 32, 32),
            torch.randn(2, 512, 16, 16)
        ]
        
        with torch.no_grad():
            output = decode_head(features)
        
        print_success("前向传播成功")
        print_info(f"输入特征: {[f.shape for f in features]}")
        print_info(f"输出: {output.shape}")
        
        return decode_head, None
    except Exception as e:
        print_error(f"Decoder Head 测试失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 8: 完整 Segmentor
# ============================================================================
def test_segmentor(cfg):
    print_section("测试 8: 完整 Segmentor (加载预训练)")
    
    try:
        from models.segmentor import build_segmentor
        
        # 不加载预训练权重
        model_cfg = cfg.model.copy()
        
        model = build_segmentor(model_cfg)
        model.eval()
        
        print_success("Segmentor 构建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_info(f"总参数: {total_params:,}")
        print_info(f"可训练参数: {trainable_params:,}")
        
        # 测试 encode_decode (推理模式)
        print("\n  测试 encode_decode...")
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            output = model.encode_decode(x)
        print_success("encode_decode 成功")
        print_info(f"输入: {x.shape}", indent=2)
        print_info(f"输出: {output.shape}", indent=2)

        # 测试 forward_train (不带 return_feat)
        print("\n  测试 forward_train (return_feat=False)...")
        model.train()
        gt = torch.randint(0, 19, (2, 1, 512, 512))
        losses = model.forward_train(x, gt, seg_weight=None, return_feat=False)
        print_success("forward_train 成功")
        print_info(f"损失项: {list(losses.keys())}", indent=2)
        print_info(f"loss_seg: {losses['loss_seg'].item():.4f}", indent=2)
        print_info(f"acc_seg: {losses['acc_seg'].item():.4f}", indent=2)
        
        # 测试 forward_train (带 return_feat)
        print("\n  测试 forward_train (return_feat=True)...")
        losses = model.forward_train(x, gt, seg_weight=None, return_feat=True)
        print_success("forward_train with features 成功")
        print_info(f"损失项: {list(losses.keys())}", indent=2)
        print_info(f"特征数量: {len(losses['features'])}", indent=2)
        for i, feat in enumerate(losses['features']):
            print_info(f"Feature {i+1}: {feat.shape}", indent=3)
        
        # 测试带权重的损失
        print("\n  测试 forward_train (with seg_weight)...")
        seg_weight = torch.rand(2, 512, 512)
        losses = model.forward_train(x, gt, seg_weight=seg_weight, return_feat=False)
        print_success("weighted loss 成功")
        print_info(f"loss_seg: {losses['loss_seg'].item():.4f}", indent=2)
        
        return model, None
    except Exception as e:
        print_error(f"Segmentor 测试失败: {str(e)}")
        traceback.print_exc()
        return None, str(e)

# ============================================================================
# 测试 9: 完整数据流水线
# ============================================================================
def test_full_pipeline(cfg, model, batch):
    print_section("测试 9: 完整数据流水线 (真实数据 -> 模型)")
    
    try:
        model.eval()
        
        img = batch['img']
        gt = batch['gt_semantic_seg']
        
        print_success("使用真实 batch 数据")
        print_info(f"img: {img.shape}")
        print_info(f"gt: {gt.shape}")
        
        # 推理
        print("\n  测试推理...")
        with torch.no_grad():
            output = model.encode_decode(img)
        print_success("推理成功")
        print_info(f"输出: {output.shape}", indent=2)
        
        # 训练
        print("\n  测试训练前向传播...")
        model.train()
        losses = model.forward_train(img, gt, return_feat=True)
        print_success("训练前向传播成功")
        print_info(f"loss_seg: {losses['loss_seg'].item():.4f}", indent=2)
        print_info(f"acc_seg: {losses['acc_seg'].item():.4f}", indent=2)
        print_info(f"features: {len(losses['features'])} scales", indent=2)
        
        # 测试反向传播
        print("\n  测试反向传播...")
        loss = losses['loss_seg']
        loss.backward()
        print_success("反向传播成功")

        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters() if p.requires_grad)
        if has_grad:
            print_success("梯度计算正确")
        else:
            print_error("警告: 未检测到梯度")
        
        return True, None
    except Exception as e:
        print_error(f"完整流水线测试失败: {str(e)}")
        traceback.print_exc()
        return False, str(e)

# ============================================================================
# 测试 10: GPU 测试
# ============================================================================
def test_gpu(cfg, batch):
    print_section("测试 10: GPU 运行")
    
    if not torch.cuda.is_available():
        print("  跳过 (无可用 GPU)")
        return True, None
    
    try:
        from models.segmentor import build_segmentor
        
        # 构建模型
        model_cfg = cfg.model.copy()
        model = build_segmentor(model_cfg)
        
        print_success(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
        
        # 移到 GPU
        model = model.cuda()
        img = batch['img'].cuda()
        gt = batch['gt_semantic_seg'].cuda()
        
        print_success("模型和数据已移至 GPU")
        
        # 推理
        print("\n  GPU 推理...")
        model.eval()
        with torch.no_grad():
            output = model.encode_decode(img)
        print_success(f"GPU 推理成功: {output.shape}")

        # 训练
        print("\n  GPU 训练...")
        model.train()
        losses = model.forward_train(img, gt, return_feat=True)
        print_success("GPU 训练前向传播成功")
        print_info(f"loss_seg: {losses['loss_seg'].item():.4f}", indent=2)
        
        # 反向传播
        print("\n  GPU 反向传播...")
        loss = losses['loss_seg']
        loss.backward()
        print_success("GPU 反向传播成功")
        
        # 显存使用
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print_info(f"显存使用: {memory_allocated:.1f} MB (allocated)", indent=2)
        print_info(f"显存预留: {memory_reserved:.1f} MB (reserved)", indent=2)
        
        return True, None
    except Exception as e:
        print_error(f"GPU 测试失败: {str(e)}")
        traceback.print_exc()
        return False, str(e)



def main():
    print("\n" + "="*70)
    print("  DAFormer 模型实现验证脚本")
    print("  测试范围: 配置 -> 数据 -> Backbone -> Decoder -> Segmentor")
    print("="*70)
    
    errors = []
    
    # 测试 1: 配置
    cfg, err = test_config_loading()
    if err:
        errors.append(("配置加载", err))
        return 1

    # 测试 2: Transform
    pipelines, err = test_transform_pipeline(cfg)
    if err:
        errors.append(("Transform Pipeline", err))
        return 1

    # 测试 3: 数据集
    datasets, err = test_datasets(cfg, pipelines)
    if err:
        errors.append(("数据集加载", err))
        return 1

    # 测试 4: UDA Dataset
    uda_dataset, err = test_uda_dataset(cfg, datasets)
    if err:
        errors.append(("UDA Dataset", err))
        return 1

    # 测试 5: DataLoader
    batch, err = test_dataloader(cfg, uda_dataset)
    if err:
        errors.append(("DataLoader", err))
        return 1

    # 测试 6: Backbone
    backbone, err = test_backbone()
    if err:
        errors.append(("Backbone", err))
        # 继续测试，不返回
    
    # 测试 7: Decoder
    decoder, err = test_decoder_head(cfg)
    if err:
        errors.append(("Decoder Head", err))
        # 继续测试
    
    # 测试 8: Segmentor
    model, err = test_segmentor(cfg)
    if err:
        errors.append(("Segmentor", err))
        return 1

    # 测试 9: 完整流水线
    success, err = test_full_pipeline(cfg, model, batch)
    if err:
        errors.append(("完整流水线", err))
    
    # 测试 10: GPU
    success, err = test_gpu(cfg, batch)
    if err:
        errors.append(("GPU 测试", err))

if __name__ == '__main__':
    exit(main())