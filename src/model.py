import torch
import numpy as np
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMWrapper:
    def __init__(self, checkpoint_path, model_type="vit_h", device=None, logger=None):
        """
        Wrapper for Segment Anything Model.
        Args:
            checkpoint_path (str): Path to .pth checkpoint.
            model_type (str): 'vit_h', 'vit_l', or 'vit_b'.
            device (str): 'cuda' or 'cpu'.
        """
        self.logger = logger or logging.getLogger('MedFusion')
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Loading SAM model ({model_type}) to {self.device}...")
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.logger.info("SAM model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}")
            raise

    def generate_masks(self, image):
        """
        Automatic mask generation for the entire image.
        Args:
            image (np.ndarray): Image array (H, W, 3) 0-255 uint8.
        Returns:
            list: List of dicts containing mask data.
        """
        # Ensure image is in 0-255 uint8 format for SAM
        if image.dtype != np.uint8:
             if image.max() <= 1.0:
                 image = (image * 255).astype(np.uint8)
             else:
                 image = image.astype(np.uint8)
        
        # If grayscale (H, W), convert to RGB (H, W, 3)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
            
        self.logger.info(f"Generating masks for image shape {image.shape}")
        
        # SAM accepts numpy uint8 images directly for generate(), 
        # but internally it turns them into tensors. 
        # Only if we use predictor.set_image do we need to be careful with embeddings?
        # Actually generate() uses the predictor internally. 
        # The error "Cannot convert a MPS Tensor to float64" usually happens when a tensor 
        # is created from a float64 numpy array on MPS.
        # But 'image' is uint8. 
        # It's possible SAM does some internal operation that defaults to float64.
        # We can try to monkeypatch or just rely on 'image' being uint8.
        # Let's verify 'image' is strictly uint8.
        
        masks = self.mask_generator.generate(image)
        return masks

    def predict_prompt(self, image, point_coords=None, point_labels=None, box=None):
        """
        Predict masks using prompts (points or boxes).
        """
        # Ensure image is suitable
        if image.dtype != np.uint8:
             if image.max() <= 1.0:
                 image = (image * 255).astype(np.uint8)
             else:
                 image = image.astype(np.uint8)
                 
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

        self.predictor.set_image(image)
        
        # If points are provided, ensure they are float32
        if point_coords is not None:
             point_coords = np.array(point_coords, dtype=np.float32)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        return masks, scores
