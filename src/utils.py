import logging
import os
import sys
import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def setup_logger(save_dir):
    """
    Sets up a logger that outputs to both console and a file.
    Args:
        save_dir (str): Directory to save the log file.
    Returns:
        logger (logging.Logger): Configured logger.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'run_{timestamp}.log')

    # Create logger
    logger = logging.getLogger('MedFusion')
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Logging started. Log file: {log_file}")
    return logger

def save_mask(image, masks, save_path, boxes=None, class_ids=None, class_names=None):
    """
    Overlays masks and bounding boxes on the image and saves the result.
    Args:
        image (np.ndarray): Original RGB image.
        masks (list): List of boolean masks from SAM.
        save_path (str): Path to save the output image.
        boxes (list): List of bounding boxes [x1, y1, x2, y2].
        class_ids (list): List of class IDs corresponding to boxes.
        class_names (list): List of class strings.
    """
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Overlay Mask
    mask_overlay = image_bgr.copy()
    if masks:
        for mask_item in masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            
            # Extract mask depending on format
            if isinstance(mask_item, dict):
                # SAM automatic mask generator returns a dict with 'segmentation' key
                mask = mask_item['segmentation']
            else:
                # SAM predictor returns raw boolean/float masks
                mask = mask_item
                
            mask_overlay[mask > 0] = color
            
    # Blend image and mask
    alpha = 0.5
    visualized_image = cv2.addWeighted(mask_overlay, alpha, image_bgr, 1 - alpha, 0)
    
    # Draw Bounding Boxes and Labels
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if class_ids and class_names:
                cls_id = int(class_ids[i])
                if 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]
                    cv2.putText(visualized_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(save_path, visualized_image)

def plot_metrics(csv_path, save_path):
    """
    Plots metrics from a CSV file.
    Args:
        csv_path (str): Path to the input CSV file.
        save_path (str): Path to save the plot image.
    """
    filename_indices = []
    num_masks = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                filename_indices.append(i)
                num_masks.append(int(row.get('num_masks', 0)))
        
        if not filename_indices:
            print("No data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(filename_indices, num_masks, marker='o', linestyle='-')
        plt.title('Number of Masks Detected per Image')
        plt.xlabel('Image Index')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")
