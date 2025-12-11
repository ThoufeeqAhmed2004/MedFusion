import os
import csv
import numpy as np
import urllib.request
from datetime import datetime
from dataloader import KidneyDataset
from model import SAMWrapper
from utils import setup_logger, save_mask, plot_metrics
import torch

class Config:
    """
    Configuration for MedFusion SAM Inference.
    Change the values here to adjust the run settings.
    """
    # Directory paths
    DATA_DIR = "/Users/sundar/Projects/MedFusion/data/KIDNEY_CT"
    BASE_OUTPUT_DIR = "output"
    
    # Model Config
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h" # 'vit_h', 'vit_l', 'vit_b'
    
    # Device Fallback Logic: CUDA -> CPU (MPS skipped due to float64 issues with SAM)
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    
    # Run Config
    NUM_IMAGES_TO_PROCESS = 5 # Number of images to run inference on
    DOWNLOAD_CHECKPOINT = True # Automatically download checkpoint if missing

def download_checkpoint(url, save_path):
    print(f"Downloading checkpoint from {url} to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")

def main():
    # 1. Setup Structured Output Directory: output/dd-mm-yyyy/timestamp
    timestamp_date = datetime.now().strftime('%d-%m-%Y')
    timestamp_time = datetime.now().strftime('%H%M%S')
    
    output_dir = os.path.join(Config.BASE_OUTPUT_DIR, timestamp_date, timestamp_time)
    images_output_dir = os.path.join(output_dir, 'images_output')
    
    os.makedirs(images_output_dir, exist_ok=True)
    
    # 2. Setup Logger
    logger = setup_logger(output_dir)
    
    logger.info("Configuration:")
    logger.info(f"  Data Dir: {Config.DATA_DIR}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Device: {Config.DEVICE}")
    logger.info(f"  Model: {Config.MODEL_TYPE}")

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
    if not os.path.exists(Config.CHECKPOINT_PATH):
        logger.info(f"Checkpoint {Config.CHECKPOINT_PATH} not found.")
        # Automatic download for vit_h
        if Config.DOWNLOAD_CHECKPOINT and "vit_h" in Config.CHECKPOINT_PATH:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            try:
                download_checkpoint(url, Config.CHECKPOINT_PATH)
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                return
        else:
            logger.error("Please provide a valid path to the checkpoint.")
            return

    # Initialize Dataset
    dataset = KidneyDataset(root_dir=Config.DATA_DIR, split="train", logger=logger)
    
    if len(dataset) == 0:
        logger.error("No images found in dataset. Exiting.")
        return

    # Initialize Model
    try:
        model = SAMWrapper(
            checkpoint_path=Config.CHECKPOINT_PATH, 
            model_type=Config.MODEL_TYPE, 
            device=Config.DEVICE, 
            logger=logger
        )
    except Exception as e:
        logger.error(f"Could not initialize model: {e}")
        return

    # Run Inference
    logger.info("Starting inference on TRAIN split...")
    CLASS_NAMES = ['Tas_Var']
    
    for i in range(min(len(dataset), Config.NUM_IMAGES_TO_PROCESS)):
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
