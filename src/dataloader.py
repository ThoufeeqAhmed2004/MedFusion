import os
import glob
import logging
import cv2
import numpy as np
from torch.utils.data import Dataset

class KidneyDataset(Dataset):
    def __init__(self, root_dir, split='train', logger=None):
        """
        Args:
            root_dir (str): Path to data directory (e.g., 'data/KIDNEY CT').
            split (str): 'train', 'val', or 'test'.
            logger (logging.Logger): Logger instance.
        """
        self.root_dir = root_dir
        self.split = split
        self.logger = logger or logging.getLogger('MedFusion')
        
        # Adjust paths for the user's specific structure:
        # User has: data/KIDNEY CT/train/images/*.jpg
        # If root_dir includes 'train', we shouldn't append it again.
        
        if split in root_dir:
             # If root_dir is ".../train", just append images
             self.images_dir = os.path.join(root_dir, 'images')
             self.labels_dir = os.path.join(root_dir, 'labels')
        else:
             # Standard structure assumption
             self.images_dir = os.path.join(root_dir, split, 'images')
             self.labels_dir = os.path.join(root_dir, split, 'labels')

        # Find all JPG/PNG files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg'))) + \
                           sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
        
        # Labels are .txt (YOLO format), we might not load them for simple inference
        # self.label_files = ... 

        self.logger.info(f"Initialized {split} dataset from {root_dir}")
        self.logger.info(f"Checking {self.images_dir}")
        self.logger.info(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_name = os.path.basename(image_path)
        
        try:
            # Load Image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load Label
            label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            boxes = []
            class_ids = []
            
            height, width, _ = image.shape
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # Convert YOLO format (normalized) to [x1, y1, x2, y2] (pixel)
                            x1 = int((x_center - w / 2) * width)
                            y1 = int((y_center - h / 2) * height)
                            x2 = int((x_center + w / 2) * width)
                            y2 = int((y_center + h / 2) * height)
                            
                            boxes.append([x1, y1, x2, y2])
                            class_ids.append(cls_id)
            
            return {
                'image': image,
                'filename': image_name,
                'boxes': np.array(boxes, dtype=np.float32),  # SAM expects float32
                'class_ids': class_ids
            }
        except Exception as e:
            self.logger.error(f"Error loading file {image_path}: {e}")
            return None
