# Cityscapes 数据集类

from .base import BaseDataset
import os
import os.path as osp

class CityscapesDataset(BaseDataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    def __init__(self, data_root, img_dir, ann_dir, pipeline):
        super().__init__(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            pipeline=pipeline
        )

    # 递归扫描 leftImg8bit/train/ 子文件夹
    def load_annotations(self):
        img_infos = []
        for root, dirs, files in os.walk(self.img_dir):
            for filename in files:
                if filename.endswith(self.img_suffix):
                    rel_path = osp.relpath(osp.join(root, filename), self.img_dir)
                    ann_file = rel_path.replace(self.img_suffix, self.seg_map_suffix)
                    img_infos.append(dict(
                        filename=rel_path,
                        ann=dict(seg_map=ann_file)
                    ))
        return img_infos