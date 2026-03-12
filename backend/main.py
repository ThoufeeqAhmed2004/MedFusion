import os
import cv2
import json
import uuid
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import sys
# Add parent directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SAMWrapper
from src.utils import setup_logger, save_mask

app = FastAPI(title="MedFusion API")

# Setup CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "sam_vit_h_4b8939.pth")
MODEL_TYPE = "vit_h"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "web")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the static directory so the frontend can access generated images
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Global variables for model and logger
sam_model = None
logger = None
executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam_model, logger
    logger = setup_logger(OUTPUT_DIR)
    logger.info("Starting up FastAPI application...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Please ensure it exists.")
    else:
        try:
            logger.info("Initializing SAM model...")
            sam_model = SAMWrapper(
                checkpoint_path=CHECKPOINT_PATH, 
                model_type=MODEL_TYPE, 
                logger=logger
            )
            logger.info("SAM model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    yield
    
    # Cleanup on shutdown
    executor.shutdown(wait=True)
    logger.info("Shutting down FastAPI application...")

# Assign lifespan
app.router.lifespan_context = lifespan

@app.get("/health")
def read_health():
    return {"status": "ok", "model_loaded": sam_model is not None, "checkpoint_path": CHECKPOINT_PATH}

@app.post("/api/segment")
async def segment_image(
    image: UploadFile = File(...),
    box: str = Form(None)
):
    if sam_model is None:
        return JSONResponse(status_code=500, content={"error": "Model failed to load on server."})

    try:
        # Read the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
             return JSONResponse(status_code=400, content={"error": "Invalid image file format."})
             
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ensure 2D (if the image has depth slice somehow in cv2 loaded)
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] > 3:
             mid_slice_idx = img_rgb.shape[2] // 2
             img_rgb = img_rgb[:, :, mid_slice_idx]
             
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{timestamp}_{session_id}"
        
        # Save original image for frontend display
        orig_img_path = os.path.join(OUTPUT_DIR, f"{base_filename}_original.png")
        cv2.imwrite(orig_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        boxes = None
        class_ids = None
        class_names = ["Object"]
        is_manual_box = False
        
        # Parse box prompt if provided
        class_name = "Kidney Stone"  # Default hardcoded for now, or infer from user selection
        if box and box.strip() != "null" and box.strip() != "":
            try:
                # Expecting format "[x1, y1, x2, y2]" or valid JSON array
                parsed_box = json.loads(box)
                if isinstance(parsed_box, list) and len(parsed_box) == 4:
                    boxes = np.array([parsed_box], dtype=np.float32)
                    class_ids = [0]
                    # The frontend could pass the target class, but since we know it's Kidney Stone
                    class_names = [class_name]
                    is_manual_box = True
                    logger.info(f"Box prompt received: {boxes}")
            except Exception as e:
                logger.warning(f"Failed to parse box: {box}. Error: {e}")

        # If not manually drawn, try to find ground truth labels
        if not is_manual_box and image.filename:
            base_name = os.path.splitext(image.filename)[0]
            data_dir = os.path.join(PROJECT_ROOT, "data")
            import pathlib
            for path in pathlib.Path(data_dir).rglob(f"{base_name}.txt"):
                if 'labels' in path.parts:
                    logger.info(f"Found GT label file: {path}")
                    try:
                        boxes_list = []
                        class_ids_list = []
                        height, width, _ = img_rgb.shape
                        with open(path, 'r') as f:
                            for line in f.readlines():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    
                                    x1 = int((x_center - w / 2) * width)
                                    y1 = int((y_center - h / 2) * height)
                                    x2 = int((x_center + w / 2) * width)
                                    y2 = int((y_center + h / 2) * height)
                                    
                                    boxes_list.append([x1, y1, x2, y2])
                                    class_ids_list.append(cls_id)
                        if boxes_list:
                            boxes = np.array(boxes_list, dtype=np.float32)
                            class_ids = class_ids_list
                            if "Kidney_Stone" in str(path):
                                class_names = ["stone"]
                            elif "Liver_Tumor" in str(path):
                                class_names = ["Tumor"]
                            logger.info(f"Loaded {len(boxes)} GT boxes from {path}")
                    except Exception as e:
                        logger.warning(f"Failed to parse GT label: {e}")
                    break

        # Run Segmentation
        logger.info(f"Processing image {image.filename}...")
        try:
            loop = asyncio.get_event_loop()
            if is_manual_box:
                # With Box Prompt (Run in executor to avoid blocking the event loop)
                masks, scores = await loop.run_in_executor(
                    executor, 
                    sam_model.predict_prompt, 
                    img_rgb, 
                    None, 
                    None, 
                    boxes[0]
                )
                logger.info(f"Generated predictions based on prompt. Scores: {scores}")
                # predict_prompt returns multiple masks with scores, we convert them nicely for save_mask
                # if mask is boolean array of shape (N, H, W)
                mask_list = []
                # Keep top scoring mask
                if len(masks) > 0:
                    best_idx = np.argmax(scores)
                    mask_list.append(masks[best_idx])
                
                result_masks = mask_list
            else:
                # Automatic Segmentation (Run in executor)
                masks = await loop.run_in_executor(
                    executor,
                    sam_model.generate_masks,
                    img_rgb
                )
                num_masks = len(masks)
                logger.info(f"Generated {num_masks} auto-masks.")
                result_masks = masks
                
                # For automatic segmentation, let's also pass the class names if we want them displayed 
                # (though usually auto segmentation masks don't have explicit class labels aligned to boxes)
                # But to satisfy the user's request, we'll ensure if bounding box is provided it labels it.

            # Save the segmented result
            seg_img_path = os.path.join(OUTPUT_DIR, f"{base_filename}_seg.png")
            save_mask(img_rgb, result_masks, seg_img_path, boxes=boxes, class_ids=class_ids, class_names=class_names)
            
            # Additional step image: just the box (if box is provided)
            box_img_path = None
            if boxes is not None:
                box_img_path = os.path.join(OUTPUT_DIR, f"{base_filename}_box.png")
                # Just draw the box on the original
                cv2.imwrite(box_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                # Re-load to use save_mask style or just draw manually
                box_img = cv2.imread(box_img_path)
                x1, y1, x2, y2 = map(int, boxes[0])
                cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(box_img_path, box_img)

            # Return the URLs accessible via /static endpoint
            response_data = {
                "status": "success",
                "original_url": f"/static/{base_filename}_original.png",
                "segmented_url": f"/static/{base_filename}_seg.png",
            }
            if box_img_path:
                response_data["box_url"] = f"/static/{base_filename}_box.png"
                
            return response_data
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process image."})

# Mount the frontend directory at the root path, serving index.html automatically for "/"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
