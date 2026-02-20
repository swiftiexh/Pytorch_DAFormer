# 日志记录工具

import json
import os
from datetime import datetime

class JSONLogger:
    def __init__(self, log_dir, experiment_name):
        os.makedirs(log_dir, exist_ok=True)
        self.train_log_path = os.path.join(log_dir, 'train.json')
        self.val_log_path = os.path.join(log_dir, 'val.json')
        self.experiment_name = experiment_name
        
    def write_config(self, config_dict):
        config_log = {
            'type': 'config',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': self.experiment_name,
            **config_dict
        }
        self._write_line(self.train_log_path, config_log)
    
    def log_train(self, iteration, log_vars):
        train_log = {
            'type': 'train',
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # 添加所有训练指标
        for key, val in log_vars.items():
            if hasattr(val, 'item'):  # torch.Tensor
                train_log[key] = float(val.item())
            else:
                train_log[key] = float(val)
        
        self._write_line(self.train_log_path, train_log)
    
    def log_val(self, iteration, eval_results):
        # Cityscapes 19类别名称
        CITYSCAPES_CLASSES = [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle'
        ]
        val_log = {
            'type': 'validation',
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mIoU': float(eval_results['mIoU'])
        }
        # 添加每类IoU
        for i, iou in enumerate(eval_results['IoU_per_class']):
            class_name = CITYSCAPES_CLASSES[i]
            val_log[class_name] = float(iou)
        
        self._write_line(self.val_log_path, val_log)
    
    def _write_line(self, file_path, log_dict):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_dict) + '\n')
