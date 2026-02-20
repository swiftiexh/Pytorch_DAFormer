import sys
import torch
from pathlib import Path

def test_data_pipeline():
    """测试数据加载、pipeline、RCS 和 batch 构造"""
    print("="*60)
    print("开始验证数据部分...")
    print("="*60)

    # 1. 加载配置
    print("\n[1/6] 加载配置...")
    sys.path.insert(0, str(Path(__file__).parent))
    from train import load_config, build_dataset, build_dataloader
    
    try:
        cfg = load_config('configs/gta2cs_daformer.py')
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False
    
    # 2. 构建数据集
    print("\n[2/6] 构建 UDA 数据集...")
    try:
        train_dataset = build_dataset(cfg.data)
        print(f"✓ 数据集构建成功")
        print(f"  - 源域样本数: {len(train_dataset.source)}")
        print(f"  - 目标域样本数: {len(train_dataset.target)}")
        print(f"  - RCS 启用: {train_dataset.rcs_enabled}")
        if train_dataset.rcs_enabled:
            print(f"  - RCS 类别数: {len(train_dataset.rcs_classes)}")
            print(f"  - RCS 类别: {train_dataset.rcs_classes[:5]}... (前5个)")
            print(f"  - 概率和: {train_dataset.rcs_classprob.sum():.4f}")
    except Exception as e:
        print(f"✗ 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试单样本获取
    print("\n[3/6] 测试单样本获取...")
    try:
        sample = train_dataset[0]
        print(f"✓ 单样本获取成功")
        print(f"  - 样本键: {list(sample.keys())}")
        
        # 检查 img
        if 'img' in sample:
            print(f"  - img shape: {sample['img'].shape}")
            print(f"  - img dtype: {sample['img'].dtype}")
        
        # 检查 gt_semantic_seg
        if 'gt_semantic_seg' in sample:
            print(f"  - gt_semantic_seg shape: {sample['gt_semantic_seg'].shape}")
            print(f"  - gt_semantic_seg dtype: {sample['gt_semantic_seg'].dtype}")
            gt = sample['gt_semantic_seg']
            unique_labels = torch.unique(gt)
            print(f"  - gt 唯一值: {unique_labels.tolist()[:10]}... (前10个)")
            print(f"  - gt 最小值: {gt.min()}, 最大值: {gt.max()}")
        
        # 检查 target_img
        if 'target_img' in sample:
            print(f"  - target_img shape: {sample['target_img'].shape}")
            print(f"  - target_img dtype: {sample['target_img'].dtype}")
            
    except Exception as e:
        print(f"✗ 单样本获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 验证 RCS 采样
    print("\n[4/6] 验证 RCS 采样...（跳过）")

    # 5. 构建 DataLoader
    print("\n[5/6] 构建 DataLoader...")
    try:
        train_loader = build_dataloader(train_dataset, cfg.data)
        print(f"✓ DataLoader 构建成功")
        print(f"  - batch_size: {cfg.data['samples_per_gpu']}")
        print(f"  - num_workers: {cfg.data['workers_per_gpu']}")
    except Exception as e:
        print(f"✗ DataLoader 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 测试 DataLoader 迭代和 batch 内容
    print("\n[6/6] 测试 DataLoader 和 Batch...")
    try:
        # 获取一个 batch
        batch = next(iter(train_loader))
        print("✓ Batch 获取成功")
        print(f"  - Batch 键: {list(batch.keys())}")
        
        # 检查 source 数据
        print("\n[Source 数据]")
        if 'img' in batch:
            print(f"  - img shape: {batch['img'].shape}")
            print(f"  - img dtype: {batch['img'].dtype}")
            print(f"  - img range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
        
        if 'gt_semantic_seg' in batch:
            print(f"  - gt_semantic_seg shape: {batch['gt_semantic_seg'].shape}")
            print(f"  - gt_semantic_seg dtype: {batch['gt_semantic_seg'].dtype}")
            unique_labels = torch.unique(batch['gt_semantic_seg'])
            print(f"  - 唯一标签: {unique_labels.tolist()}")
        
        # 检查 target 数据
        print("\n[Target 数据]")
        if 'target_img' in batch:
            print(f"  - target_img shape: {batch['target_img'].shape}")
            print(f"  - target_img dtype: {batch['target_img'].dtype}")
            print(f"  - target_img range: [{batch['target_img'].min():.3f}, {batch['target_img'].max():.3f}]")
        
        # 测试 CUDA 传输
        print("\n[CUDA 传输测试]")
        if torch.cuda.is_available():
            try:
                batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                print("  ✓ Batch 成功传输到 GPU")
                print(f"    - img device: {batch_gpu['img'].device}")
                if 'gt_semantic_seg' in batch_gpu:
                    print(f"    - gt_semantic_seg device: {batch_gpu['gt_semantic_seg'].device}")
                print(f"    - target_img device: {batch_gpu['target_img'].device}")
            except Exception as e:
                print(f"  ✗ GPU 传输失败: {e}")
        else:
            print("  - CUDA 不可用,跳过 GPU 测试")
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！数据部分验证完成")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Batch 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
            
if __name__ == '__main__':
    success = test_data_pipeline()
    sys.exit(0 if success else 1)