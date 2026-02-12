import os
import csv
import argparse
import numpy as np
import urllib.request
from datetime import datetime
from dataloader import MedicalDataset
from model import SAMWrapper
from utils import setup_logger, save_mask, plot_metrics
import torch

# Default Config
BASE_OUTPUT_DIR = "output"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DOWNLOAD_CHECKPOINT = True

# Device Fallback Logic
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def download_checkpoint(url, save_path):
    print(f"Downloading checkpoint from {url} to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="MedFusion Inference")
    parser.add_argument("--dataset", type=str, default="Kidney_Stone", 
                        choices=["Kidney_Stone", "Liver_Tumor"],
                        help="Name of the dataset folder in data/")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of images to process")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to a single image file for inference")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device to run on (cuda/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Structured Output Directory: output/dateset/dd-mm-yyyy/timestamp
    timestamp_date = datetime.now().strftime('%d-%m-%Y')
    timestamp_time = datetime.now().strftime('%H%M%S')
    
    # Organize output by dataset name
    output_dir = os.path.join(BASE_OUTPUT_DIR, args.dataset, timestamp_date, timestamp_time)
    images_output_dir = os.path.join(output_dir, 'images_output')
    
    os.makedirs(images_output_dir, exist_ok=True)
    
    # 2. Setup Logger
    logger = setup_logger(output_dir)
    
    # Resolve Data Directory
    # Assumes data is in project_root/data/{dataset_name}
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", args.dataset)
    
    logger.info("Configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Data Dir: {data_dir}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Model: {MODEL_TYPE}")

    # 3. Setup CSV Logging
    csv_path = os.path.join(output_dir, 'logs.csv')
    try:
        csv_file = open(csv_path, 'w', newline='')
        fieldnames = ['filename', 'num_masks', 'status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    except Exception as e:
        logger.error(f"Failed to setup CSV logging: {e}")
        return

    # Check/Download Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Checkpoint {CHECKPOINT_PATH} not found.")
        # Automatic download for vit_h
        if DOWNLOAD_CHECKPOINT and "vit_h" in CHECKPOINT_PATH:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            try:
                download_checkpoint(url, CHECKPOINT_PATH)
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                return
        else:
            logger.error("Please provide a valid path to the checkpoint.")
            return

    # Initialize Dataset or Single Image
    if args.image_path:
        if not os.path.exists(args.image_path):
            logger.error(f"Image not found: {args.image_path}")
            return
            
        import cv2
        try:
            image = cv2.imread(args.image_path)
            if image is None:
                raise ValueError("Failed to load image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(args.image_path)
            
            # Create a single-item dataset-like list
            dataset = [{
                'image': image, 
                'filename': filename, 
                'boxes': None, 
                'class_ids': None
            }]
            logger.info(f"Running inference on single image: {args.image_path}")
            
        except Exception as e:
             logger.error(f"Error loading image {args.image_path}: {e}")
             return
    else:
        dataset = MedicalDataset(root_dir=data_dir, split="train", logger=logger)
        
        if len(dataset) == 0:
            logger.error("No images found in dataset. Exiting.")
            return

    # Initialize Model
    try:
        model = SAMWrapper(
            checkpoint_path=CHECKPOINT_PATH, 
            model_type=MODEL_TYPE, 
            device=args.device, 
            logger=logger
        )
    except Exception as e:
        logger.error(f"Could not initialize model: {e}")
        return

    # Run Inference
    logger.info("Starting inference...")
    
    # Determine Class Names based on Dataset
    if args.dataset == "Kidney_Stone":
        # Check if data.yaml exists to confirm class names, or hardcode common ones
        CLASS_NAMES = ['stone'] 
    elif args.dataset == "Liver_Tumor":
        CLASS_NAMES = ['Liver']
    else:
        CLASS_NAMES = ['Object']
    
    # Determine number of images to process
    num_to_process = 1 if args.image_path else min(len(dataset), args.num_images)

    for i in range(num_to_process):
        data = dataset[i]
        if data is None:
            continue
            
        image = data['image']
        filename = data['filename']
        boxes = data.get('boxes')
        class_ids = data.get('class_ids')
        
        logger.info(f"Processing {filename}...")
        
        # Ensure 2D
        if len(image.shape) == 3 and image.shape[2] > 3:
             mid_slice_idx = image.shape[2] // 2
             img_slice = image[:, :, mid_slice_idx]
        else:
             img_slice = image
        
        try:
            # Hybrid Mode: "Segment Everything" + "Show Labels"
            
            # 1. Automatic Mask Generation (Segment Everything)
            masks = model.generate_masks(img_slice)
            num_masks = len(masks)
            logger.info(f"Generated {num_masks} auto-masks for {filename}")
            
            if boxes is not None:
                logger.info(f"Found {len(boxes)} GT boxes for visualization")
            
            # 2. Save Visualization (Auto Masks + GT Boxes overlaid)
            save_path = os.path.join(images_output_dir, f"{filename}_seg.png")
            save_mask(img_slice, masks, save_path, boxes=boxes, class_ids=class_ids, class_names=CLASS_NAMES)
            
            # Log to CSV
            writer.writerow({'filename': filename, 'num_masks': num_masks, 'status': 'success'})
            
        except Exception as e:
            logger.error(f"Inference failed for {filename}: {e}")
            writer.writerow({'filename': filename, 'num_masks': 0, 'status': f"failed: {e}"})

    # Close CSV
    csv_file.close()

    # Generate Graph
    graph_path = os.path.join(output_dir, 'graph.png')
    plot_metrics(csv_path, graph_path)
    logger.info(f"Metrics graph saved to {graph_path}")

    logger.info("Inference complete.")

if __name__ == "__main__":
    main()
