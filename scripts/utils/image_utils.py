import numpy as np
import torch
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def tensor2pil(image):
    """Convert a PyTorch tensor to a PIL Image"""
    try:
        # Move to CPU if on GPU
        if image.device != 'cpu':
            image = image.cpu()
        # Convert to numpy array
        return Image.fromarray(np.clip(255. * image.numpy().squeeze(), 0, 255).astype(np.uint8))
    except Exception as e:
        logger.error(f"Error converting tensor to PIL: {str(e)}")
        return Image.new('RGB', (image.shape[-2], image.shape[-1]), (0, 0, 0))

def pil2tensor(image):
    """Convert a PIL Image to a PyTorch tensor"""
    try:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error converting PIL to tensor: {str(e)}")
        return torch.zeros((1, 3, image.size[1], image.size[0]))

def debug_tensor_info(tensor, name="Tensor"):
    """Utility function to debug tensor information"""
    try:
        logger.info(f"{name} shape: {tensor.shape}")
        logger.info(f"{name} dtype: {tensor.dtype}")
        logger.info(f"{name} device: {tensor.device}")
        logger.info(f"{name} min: {tensor.min()}")
        logger.info(f"{name} max: {tensor.max()}")
    except Exception as e:
        logger.error(f"Error debugging tensor info: {str(e)}")

def parse_aspect_ratio(aspect_ratio_input: str):
    """Enhanced aspect ratio parsing with better error handling"""
    if not aspect_ratio_input:
        return None
    
    try:
        if ':' in aspect_ratio_input:
            w, h = map(float, aspect_ratio_input.split(':'))
            if h == 0:
                logger.warning("Invalid aspect ratio: height cannot be zero")
                return None
            return w / h
        
        try:
            return float(aspect_ratio_input)
        except ValueError:
            pass

        standard_ratios = {
            '4:3': 4/3,
            '16:9': 16/9,
            '21:9': 21/9,
            '1:1': 1,
            'square': 1,
            'portrait': 3/4,
            'landscape': 4/3
        }
        
        return standard_ratios.get(aspect_ratio_input.lower())
    
    except Exception as e:
        logger.error(f"Error parsing aspect ratio: {str(e)}")
        return None