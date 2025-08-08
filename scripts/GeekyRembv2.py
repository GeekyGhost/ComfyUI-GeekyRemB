import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw
from rembg import remove, new_session
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Callable, Any
import math
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
import gc
import warnings
from functools import lru_cache, wraps
import copy
import json
import tempfile
import os
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# === ENUMERATIONS ===
class RemovalMethod(Enum):
    """Background removal methods"""
    RMBG_V2 = "rmbg_v2_birefnet"
    RMBG_V1_4 = "rmbg_v1_4_briaai"
    BIREFNET = "birefnet_zhengpeng7"
    CHROMA_KEY = "chroma_key_advanced"
    HYBRID = "hybrid_ai_chroma"

class BlendMode(Enum):
    """Professional blend modes"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    LINEAR_BURN = "linear_burn"
    LINEAR_DODGE = "linear_dodge"

class PositionMode(Enum):
    """Position modes for precise placement"""
    ABSOLUTE_PIXELS = "absolute_pixels"
    RELATIVE_PERCENT = "relative_percent"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    CUSTOM_ANCHOR = "custom_anchor"

class AnimationType(Enum):
    """Animation types for dynamic effects"""
    NONE = "none"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    SLIDE_DIAGONAL = "slide_diagonal"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    SCALE_PULSE = "scale_pulse"
    ROTATE_CW = "rotate_clockwise"
    ROTATE_CCW = "rotate_counter_clockwise"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADE_PULSE = "fade_pulse"
    BOUNCE_IN = "bounce_in"
    BOUNCE_OUT = "bounce_out"
    ELASTIC_IN = "elastic_in"
    ELASTIC_OUT = "elastic_out"
    ORBIT_CIRCULAR = "orbit_circular"
    SPIRAL_IN = "spiral_in"
    SHAKE_SUBTLE = "shake_subtle"
    WOBBLE = "wobble"

class EasingFunction(Enum):
    """Easing functions for smooth animation"""
    LINEAR = "linear"
    EASE_IN = "ease_in_quad"
    EASE_OUT = "ease_out_quad"
    EASE_IN_OUT = "ease_in_out_quad"
    ELASTIC_OUT = "elastic_out"
    BOUNCE_OUT = "bounce_out"
    BACK_OUT = "back_out"
    SINE_IN_OUT = "sine_in_out"

# === CONFIGURATION CLASSES ===
@dataclass
class BackgroundRemovalConfig:
    """Advanced background removal configuration"""
    method: RemovalMethod = RemovalMethod.RMBG_V2
    model_precision: str = "fp16"
    batch_size: int = 1
    use_gpu_acceleration: bool = True
    chroma_color: str = "green"
    custom_chroma_rgb: Tuple[int, int, int] = (0, 255, 0)
    tolerance: float = 0.15
    softness: float = 0.8
    spill_suppression: float = 0.9
    edge_feathering: float = 3.0
    mask_blur: float = 1.5
    mask_dilation: int = 2
    mask_erosion: int = 1
    remove_small_objects: bool = True
    small_object_threshold: int = 500
    processing_resolution: int = 1024
    maintain_aspect_ratio: bool = True
    anti_aliasing: bool = True

@dataclass
class PositionConfig:
    """Comprehensive positioning configuration"""
    mode: PositionMode = PositionMode.CENTER
    x_offset: float = 0.0
    y_offset: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False

@dataclass
class BlendingConfig:
    """Professional blending configuration"""
    blend_mode: BlendMode = BlendMode.NORMAL
    opacity: float = 1.0
    preserve_luminosity: bool = False
    knock_out: bool = False
    drop_shadow: bool = False
    shadow_offset_x: float = 5.0
    shadow_offset_y: float = 5.0
    shadow_blur: float = 10.0
    shadow_opacity: float = 0.5
    shadow_color: Tuple[int, int, int] = (0, 0, 0)

@dataclass
class AnimationConfig:
    """Advanced animation configuration"""
    animation_type: AnimationType = AnimationType.NONE
    duration: float = 2.0
    easing: EasingFunction = EasingFunction.EASE_IN_OUT
    amplitude: float = 100.0
    frequency: float = 1.0
    start_delay: float = 0.0
    loop_count: int = 1
    ping_pong: bool = False
    custom_curve: Optional[List[Tuple[float, float]]] = None

# === UTILITY FUNCTIONS ===
def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image with comprehensive error handling"""
    try:
        if tensor is None:
            logger.error("Received None tensor")
            return Image.new('RGB', (512, 512), (128, 128, 128))
        
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"Expected tensor, got {type(tensor)}")
            return Image.new('RGB', (512, 512), (128, 128, 128))
        
        if tensor.numel() == 0:
            logger.error("Received empty tensor")
            return Image.new('RGB', (512, 512), (128, 128, 128))
        
        if tensor.device != 'cpu':
            tensor = tensor.cpu()
        
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        if len(tensor.shape) == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            if tensor.shape[2] == 1:
                tensor = tensor.repeat(1, 1, 3)
        
        numpy_array = tensor.numpy()
        if np.any(np.isnan(numpy_array)) or np.any(np.isinf(numpy_array)):
            logger.warning("Tensor contains NaN or Inf values, cleaning...")
            numpy_array = np.nan_to_num(numpy_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        numpy_array = np.clip(numpy_array * 255.0, 0, 255).astype(np.uint8)
        
        if len(numpy_array.shape) == 2:
            numpy_array = np.stack([numpy_array] * 3, axis=-1)
        
        if numpy_array.shape[0] == 0 or numpy_array.shape[1] == 0:
            logger.error(f"Invalid array dimensions: {numpy_array.shape}")
            return Image.new('RGB', (512, 512), (128, 128, 128))
        
        return Image.fromarray(numpy_array)
    
    except Exception as e:
        logger.error(f"Error converting tensor to PIL: {e}")
        return Image.new('RGB', (512, 512), (128, 128, 128))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor with error handling"""
    try:
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        numpy_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(numpy_array).unsqueeze(0)
    
    except Exception as e:
        logger.error(f"Error converting PIL to tensor: {e}")
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

# === EASING FUNCTIONS ===
def ease_linear(t: float) -> float:
    return t

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return t * (2 - t)

def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

def ease_elastic_out(t: float) -> float:
    if t == 0 or t == 1:
        return t
    return 2 ** (-10 * t) * math.sin((t - 0.1) * 5 * math.pi) + 1

def ease_bounce_out(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        return n1 * (t - 1.5 / d1) ** 2 + 0.75
    elif t < 2.5 / d1:
        return n1 * (t - 2.25 / d1) ** 2 + 0.9375
    else:
        return n1 * (t - 2.625 / d1) ** 2 + 0.984375

def ease_back_out(t: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def ease_sine_in_out(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2

EASING_FUNCTIONS = {
    EasingFunction.LINEAR: ease_linear,
    EasingFunction.EASE_IN: ease_in_quad,
    EasingFunction.EASE_OUT: ease_out_quad,
    EasingFunction.EASE_IN_OUT: ease_in_out_quad,
    EasingFunction.ELASTIC_OUT: ease_elastic_out,
    EasingFunction.BOUNCE_OUT: ease_bounce_out,
    EasingFunction.BACK_OUT: ease_back_out,
    EasingFunction.SINE_IN_OUT: ease_sine_in_out,
}

# === CORE PROCESSORS ===
class AdvancedBackgroundRemover:
    """State-of-the-art background removal with latest models"""
    
    def __init__(self):
        self.sessions = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session_lock = RLock()
        self.birefnet = None
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def _load_birefnet(self, config: BackgroundRemovalConfig):
        """Load BiRefNet model with proper device handling"""
        try:
            if self.birefnet is None:
                torch.set_float32_matmul_precision("high")
                self.birefnet = AutoModelForImageSegmentation.from_pretrained(
                    "cocktailpeanut/rm", trust_remote_code=True
                )
                self.birefnet = self.birefnet.to(self.device)
                logger.info(f"Loaded BiRefNet model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BiRefNet: {e}")
            raise RuntimeError(f"BiRefNet loading failed: {e}")

    def _prepare_image_for_processing(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Tuple[int, int]]:
        """Prepare image for processing with aspect ratio preservation"""
        try:
            original_size = image.size
            if config.maintain_aspect_ratio:
                aspect_ratio = original_size[0] / original_size[1]
                if aspect_ratio > 1:
                    new_width = config.processing_resolution
                    new_height = int(config.processing_resolution / aspect_ratio)
                else:
                    new_height = config.processing_resolution
                    new_width = int(config.processing_resolution * aspect_ratio)
            else:
                new_width, new_height = config.processing_resolution, config.processing_resolution
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image.convert("RGB"), original_size
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            return image.convert("RGB"), image.size

    def _process_mask(self, mask: Image.Image, config: BackgroundRemovalConfig) -> Image.Image:
        """Apply advanced mask processing"""
        try:
            mask_array = np.array(mask)
            if config.remove_small_objects:
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < config.small_object_threshold:
                        cv2.drawContours(mask_array, [contour], -1, 0, -1)
            
            kernel = np.ones((config.mask_dilation, config.mask_dilation), np.uint8)
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            if config.mask_erosion > 0:
                erode_kernel = np.ones((config.mask_erosion, config.mask_erosion), np.uint8)
                mask_array = cv2.erode(mask_array, erode_kernel, iterations=1)
            
            if config.mask_blur > 0:
                mask_array = cv2.GaussianBlur(mask_array, (0, 0), config.mask_blur)
            
            return Image.fromarray(mask_array)
        except Exception as e:
            logger.error(f"Error processing mask: {e}")
            return mask

    def remove_background(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        """Remove background using specified method"""
        if config.method == RemovalMethod.RMBG_V2 or config.method == RemovalMethod.BIREFNET:
            return self._remove_birefnet(image, config)
        elif config.method == RemovalMethod.RMBG_V1_4:
            return self._remove_rmbg_v1_4(image, config)
        elif config.method == RemovalMethod.CHROMA_KEY:
            return self._remove_chroma_key(image, config)
        elif config.method == RemovalMethod.HYBRID:
            return self._remove_hybrid(image, config)
        else:
            raise ValueError(f"Unknown removal method: {config.method}")

    def _remove_birefnet(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        """Remove background using BiRefNet model"""
        try:
            self._load_birefnet(config)
            processed_image, original_size = self._prepare_image_for_processing(image, config)
            
            # Validate image dimensions
            if processed_image.size[0] == 0 or processed_image.size[1] == 0:
                raise ValueError("Processed image has invalid dimensions")
            
            # Transform image for BiRefNet
            input_images = self.transform_image(processed_image).unsqueeze(0).to(self.device)
            
            # Resize input to match model expectations
            resize_transform = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR)
            input_images = resize_transform(input_images)
            
            # Process with BiRefNet
            with torch.no_grad():
                preds = self.birefnet(input_images)[-1].sigmoid().cpu()
            
            # Convert prediction to mask
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # Validate mask dimensions
            if mask.size != original_size:
                logger.error(f"Mask size {mask.size} does not match original size {original_size}")
                raise ValueError("images do not match")
            
            # Apply mask processing
            mask = self._process_mask(mask, config)
            
            # Apply mask to original image
            result = image.convert("RGBA")
            result.putalpha(mask)
            
            # Clean up memory
            torch.cuda.empty_cache() if self.device == "cuda" else None
            
            return result, mask
        
        except Exception as e:
            logger.error(f"BiRefNet processing failed: {e}")
            return image.convert("RGBA"), Image.new('L', image.size, 255)

    def _remove_rmbg_v1_4(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        """Remove background using RMBG v1.4"""
        with self.session_lock:
            model_name = "briaai/RMBG-1.4"
            if model_name not in self.sessions:
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.use_gpu_acceleration and self.device == "cuda" else ['CPUExecutionProvider']
                    session = new_session(model_name, providers=providers)
                    self.sessions[model_name] = session
                    logger.info(f"Loaded RMBG v1.4 model")
                except Exception as e:
                    logger.warning(f"Failed to load RMBG v1.4, falling back to u2net: {e}")
                    session = new_session("u2net", providers=providers)
                    self.sessions[model_name] = session
            
            session = self.sessions[model_name]
        
        original_size = image.size
        processed_image = self._prepare_image_for_processing(image, config)[0]
        
        result = remove(processed_image, session=session)
        
        if processed_image.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        if result.mode == 'RGBA':
            mask = result.split()[3]
        else:
            result = result.convert('RGBA')
            mask = Image.new('L', result.size, 255)
        
        mask = self._process_mask(mask, config)
        
        result_array = np.array(result)
        mask_array = np.array(mask)
        result_array[:, :, 3] = mask_array
        result = Image.fromarray(result_array, 'RGBA')
        
        return result, mask

    def _remove_chroma_key(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        """Advanced chroma key removal with sophisticated algorithms"""
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV).astype(np.float32)
        
        if config.chroma_color == "green":
            mask = self._create_green_mask(hsv, lab, yuv, config)
        elif config.chroma_color == "blue":
            mask = self._create_blue_mask(hsv, lab, yuv, config)
        elif config.chroma_color == "red":
            mask = self._create_red_mask(hsv, lab, yuv, config)
        else:
            mask = self._create_custom_mask(hsv, lab, yuv, config)
        
        mask = self._process_mask(Image.fromarray((mask * 255).astype(np.uint8)), config)
        result = image.convert('RGBA')
        result.putalpha(mask)
        
        return result, mask

    def _remove_hybrid(self, image: Image.Image, config: BackgroundRemovalConfig) -> Tuple[Image.Image, Image.Image]:
        """Hybrid AI + Chroma key removal"""
        ai_result, ai_mask = self._remove_birefnet(image, config)
        chroma_result, chroma_mask = self._remove_chroma_key(image, config)
        
        ai_mask_array = np.array(ai_mask).astype(np.float32) / 255.0
        chroma_mask_array = np.array(chroma_mask).astype(np.float32) / 255.0
        combined_mask = (ai_mask_array * 0.7 + chroma_mask_array * 0.3)
        combined_mask = Image.fromarray((combined_mask * 255).astype(np.uint8))
        
        result = image.convert('RGBA')
        result.putalpha(combined_mask)
        
        return result, combined_mask

    def _create_green_mask(self, hsv, lab, yuv, config):
        """Create mask for green screen"""
        hue = hsv[:, :, 0]
        green_hue_range = (60, 90)
        base_mask = (hue >= green_hue_range[0]) & (hue <= green_hue_range[1])
        base_mask = base_mask.astype(np.float32)
        
        if config.spill_suppression > 0:
            base_mask = self._apply_spill_suppression(base_mask, hsv, config)
        
        return cv2.GaussianBlur(base_mask, (0, 0), config.softness)

    def _create_blue_mask(self, hsv, lab, yuv, config):
        """Create mask for blue screen"""
        hue = hsv[:, :, 0]
        blue_hue_range = (210, 270)
        base_mask = (hue >= blue_hue_range[0]) & (hue <= blue_hue_range[1])
        base_mask = base_mask.astype(np.float32)
        
        if config.spill_suppression > 0:
            base_mask = self._apply_spill_suppression(base_mask, hsv, config)
        
        return cv2.GaussianBlur(base_mask, (0, 0), config.softness)

    def _create_red_mask(self, hsv, lab, yuv, config):
        """Create mask for red screen"""
        hue = hsv[:, :, 0]
        red_hue_range = (0, 30)
        base_mask = (hue >= red_hue_range[0]) & (hue <= red_hue_range[1])
        base_mask = base_mask.astype(np.float32)
        
        if config.spill_suppression > 0:
            base_mask = self._apply_spill_suppression(base_mask, hsv, config)
        
        return cv2.GaussianBlur(base_mask, (0, 0), config.softness)

    def _create_custom_mask(self, hsv, lab, yuv, config):
        """Create mask for custom color"""
        rgb = np.array(config.custom_chroma_rgb).astype(np.float32) / 255.0
        hsv_color = cv2.cvtColor(rgb[np.newaxis, np.newaxis, :], cv2.COLOR_RGB2HSV)[0, 0]
        hue = hsv[:, :, 0]
        base_mask = np.abs(hue - hsv_color[0]) < (config.tolerance * 180)
        base_mask = base_mask.astype(np.float32)
        
        if config.spill_suppression > 0:
            base_mask = self._apply_spill_suppression(base_mask, hsv, config)
        
        return cv2.GaussianBlur(base_mask, (0, 0), config.softness)

    def _apply_spill_suppression(self, mask, hsv, config):
        """Apply color spill suppression"""
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        spill_mask = (saturation > config.tolerance * 255) & (value > config.tolerance * 255)
        return mask * (1 - config.spill_suppression * spill_mask.astype(np.float32))

class ProfessionalCompositor:
    """Handle professional compositing operations"""
    
    def composite_layers(self, bg: Image.Image, fg: Image.Image, position: PositionConfig, blend: BlendingConfig) -> Image.Image:
        """Composite foreground and background with advanced positioning and blending"""
        try:
            if bg.mode != 'RGBA':
                bg = bg.convert('RGBA')
            if fg.mode != 'RGBA':
                fg = fg.convert('RGBA')
            
            result = bg.copy()
            fg_size = fg.size
            
            if position.flip_horizontal:
                fg = fg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if position.flip_vertical:
                fg = fg.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            
            new_width = int(fg_size[0] * position.scale_x)
            new_height = int(fg_size[1] * position.scale_y)
            if new_width > 0 and new_height > 0:
                fg = fg.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if position.rotation != 0:
                fg = fg.rotate(position.rotation, expand=True, resample=Image.Resampling.BICUBIC)
            
            if position.mode == PositionMode.CENTER:
                x = (bg.size[0] - fg.size[0]) // 2 + int(position.x_offset)
                y = (bg.size[1] - fg.size[1]) // 2 + int(position.y_offset)
            elif position.mode == PositionMode.ABSOLUTE_PIXELS:
                x = int(position.x_offset)
                y = int(position.y_offset)
            elif position.mode == PositionMode.RELATIVE_PERCENT:
                x = int(bg.size[0] * position.x_offset / 100)
                y = int(bg.size[1] * position.y_offset / 100)
            else:
                positions = {
                    PositionMode.TOP_LEFT: (0, 0),
                    PositionMode.TOP_CENTER: ((bg.size[0] - fg.size[0]) // 2, 0),
                    PositionMode.TOP_RIGHT: (bg.size[0] - fg.size[0], 0),
                    PositionMode.MIDDLE_LEFT: (0, (bg.size[1] - fg.size[1]) // 2),
                    PositionMode.MIDDLE_RIGHT: (bg.size[0] - fg.size[0], (bg.size[1] - fg.size[1]) // 2),
                    PositionMode.BOTTOM_LEFT: (0, bg.size[1] - fg.size[1]),
                    PositionMode.BOTTOM_CENTER: ((bg.size[0] - fg.size[0]) // 2, bg.size[1] - fg.size[1]),
                    PositionMode.BOTTOM_RIGHT: (bg.size[0] - fg.size[0], bg.size[1] - fg.size[1]),
                    PositionMode.CUSTOM_ANCHOR: (
                        int(bg.size[0] * position.anchor_x - fg.size[0] * position.anchor_x + position.x_offset),
                        int(bg.size[1] * position.anchor_y - fg.size[1] * position.anchor_y + position.y_offset)
                    )
                }
                x, y = positions.get(position.mode, (0, 0))
            
            if blend.blend_mode == BlendMode.NORMAL:
                result.paste(fg, (x, y), fg)
            else:
                fg_array = np.array(fg).astype(np.float32) / 255.0
                bg_array = np.array(bg).astype(np.float32) / 255.0
                result_array = self._apply_advanced_blending(fg_array, bg_array, blend)
                result = Image.fromarray((result_array * 255).astype(np.uint8), 'RGBA')
            
            if blend.drop_shadow:
                result = self._apply_drop_shadow(result, fg, x, y, blend)
            
            return result
        
        except Exception as e:
            logger.error(f"Compositing error: {e}")
            return bg

    def _apply_advanced_blending(self, fg: np.ndarray, bg: np.ndarray, blend: BlendingConfig) -> np.ndarray:
        """Apply advanced blending modes"""
        result = bg.copy()
        if blend.blend_mode == BlendMode.MULTIPLY:
            result[:, :, :3] = bg[:, :, :3] * fg[:, :, :3]
        elif blend.blend_mode == BlendMode.SCREEN:
            result[:, :, :3] = 1 - (1 - bg[:, :, :3]) * (1 - fg[:, :, :3])
        elif blend.blend_mode == BlendMode.OVERLAY:
            mask = bg[:, :, :3] < 0.5
            result[:, :, :3][mask] = 2 * bg[:, :, :3][mask] * fg[:, :, :3][mask]
            result[:, :, :3][~mask] = 1 - 2 * (1 - bg[:, :, :3][~mask]) * (1 - fg[:, :, :3][~mask])
        result[:, :, 3] = bg[:, :, 3] * (1 - blend.opacity) + fg[:, :, 3] * blend.opacity
        return np.clip(result, 0, 1)

    def _apply_drop_shadow(self, image: Image.Image, fg: Image.Image, x: int, y: int, blend: BlendingConfig) -> Image.Image:
        """Apply drop shadow effect"""
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_layer = Image.new('RGBA', fg.size, blend.shadow_color + (int(255 * blend.shadow_opacity),))
        shadow.paste(shadow_layer, (x + int(blend.shadow_offset_x), y + int(blend.shadow_offset_y)), shadow_layer)
        shadow = shadow.filter(ImageFilter.GaussianBlur(blend.shadow_blur))
        result = Image.alpha_composite(shadow, image)
        return result

class AdvancedAnimator:
    """Handle advanced animation calculations"""
    
    def animate_layer(self, position_config: PositionConfig, animation_config: AnimationConfig, 
                    current_time: float, total_duration: float) -> PositionConfig:
        """Apply animation transformations to position config"""
        try:
            if animation_config.animation_type == AnimationType.NONE or total_duration <= 0:
                return position_config
            
            result_config = copy.deepcopy(position_config)
            anim_type = animation_config.animation_type
            t = max(0, min(1, (current_time - animation_config.start_delay) / animation_config.duration))
            
            if animation_config.loop_count != 1:
                if animation_config.ping_pong:
                    t = abs((t % 2) - 1) if int(t) % 2 == 1 else t % 1
                else:
                    t = t % 1
            
            eased_t = EASING_FUNCTIONS[animation_config.easing](t)
            amplitude = animation_config.amplitude
            frequency = animation_config.frequency
            
            original_scale_x = result_config.scale_x
            original_scale_y = result_config.scale_y
            
            if anim_type in [AnimationType.SLIDE_LEFT, AnimationType.SLIDE_RIGHT, AnimationType.SLIDE_DIAGONAL]:
                direction = -1 if anim_type == AnimationType.SLIDE_LEFT else 1
                result_config.x_offset += eased_t * amplitude * direction
                if anim_type == AnimationType.SLIDE_DIAGONAL:
                    result_config.y_offset += eased_t * amplitude * direction
            
            elif anim_type in [AnimationType.SLIDE_UP, AnimationType.SLIDE_DOWN]:
                direction = -1 if anim_type == AnimationType.SLIDE_UP else 1
                result_config.y_offset += eased_t * amplitude * direction
            
            elif anim_type in [AnimationType.SCALE_IN, AnimationType.SCALE_OUT, AnimationType.SCALE_PULSE]:
                scale_factor = eased_t if anim_type == AnimationType.SCALE_IN else (1 - eased_t)
                if anim_type == AnimationType.SCALE_PULSE:
                    scale_factor = 0.5 + 0.5 * math.sin(eased_t * frequency * 2 * math.pi)
                result_config.scale_x *= (1 + scale_factor * amplitude / 100)
                result_config.scale_y *= (1 + scale_factor * amplitude / 100)
            
            elif anim_type in [AnimationType.ROTATE_CW, AnimationType.ROTATE_CCW]:
                direction = 1 if anim_type == AnimationType.ROTATE_CW else -1
                result_config.rotation += eased_t * amplitude * direction
            
            elif anim_type in [AnimationType.BOUNCE_IN, AnimationType.BOUNCE_OUT]:
                bounce = eased_t if anim_type == AnimationType.BOUNCE_IN else (1 - eased_t)
                result_config.y_offset += bounce * amplitude
                result_config.scale_x *= (1 + bounce * amplitude / 1000)
                result_config.scale_y *= (1 + bounce * amplitude / 1000)
            
            elif anim_type in [AnimationType.ELASTIC_IN, AnimationType.ELASTIC_OUT]:
                elastic = eased_t if anim_type == AnimationType.ELASTIC_IN else (1 - eased_t)
                result_config.scale_x *= (1 + elastic * amplitude / 100)
                result_config.scale_y *= (1 + elastic * amplitude / 100)
            
            elif anim_type == AnimationType.ORBIT_CIRCULAR:
                angle = t * frequency * 2 * math.pi
                result_config.x_offset += amplitude * math.cos(angle)
                result_config.y_offset += amplitude * math.sin(angle)
            
            elif anim_type == AnimationType.SPIRAL_IN:
                angle = t * frequency * 6 * math.pi
                radius = (1 - t) * amplitude
                result_config.x_offset += radius * math.cos(angle)
                result_config.y_offset += radius * math.sin(angle)
            
            elif anim_type == AnimationType.SHAKE_SUBTLE:
                shake_x = math.sin(t * frequency * 20 * math.pi) * amplitude * 0.1
                shake_y = math.cos(t * frequency * 23 * math.pi) * amplitude * 0.1
                result_config.x_offset += shake_x
                result_config.y_offset += shake_y
            
            elif anim_type == AnimationType.WOBBLE:
                wobble_angle = t * frequency * 4 * math.pi
                wobble_amount = math.sin(t * frequency * 2 * math.pi) * amplitude * 0.2
                result_config.x_offset += wobble_amount * math.cos(wobble_angle)
                result_config.y_offset += wobble_amount * math.sin(wobble_angle)
            
            result_config.scale_x = max(0.001, result_config.scale_x) if result_config.scale_x != original_scale_x else result_config.scale_x
            result_config.scale_y = max(0.001, result_config.scale_y) if result_config.scale_y != original_scale_y else result_config.scale_y
            
            return result_config
        
        except Exception as e:
            logger.error(f"Animation error: {e}")
            return position_config

# === MAIN NODE CLASS ===
class GeekyRemBv4:
    """
    GeekyRemB v4.0 - Ultimate Background Removal and Video Layering Node with Latest AI Models
    """
    
    def __init__(self):
        self.background_remover = AdvancedBackgroundRemover()
        self.compositor = ProfessionalCompositor()
        self.animator = AdvancedAnimator()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "skip_background_removal": ("BOOLEAN", {"default": False}),
                "removal_method": ([method.value for method in RemovalMethod], {"default": RemovalMethod.RMBG_V2.value}),
                "chroma_color": (["green", "blue", "red"], {"default": "green"}),
                "chroma_tolerance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_softness": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "spill_suppression": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "position_mode": ([mode.value for mode in PositionMode], {"default": PositionMode.CENTER.value}),
                "x_position": ("FLOAT", {"default": 0.0, "min": -4000.0, "max": 4000.0, "step": 1.0}),
                "y_position": ("FLOAT", {"default": 0.0, "min": -4000.0, "max": 4000.0, "step": 1.0}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "blend_mode": ([mode.value for mode in BlendMode], {"default": BlendMode.NORMAL.value}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "animation_type": ([anim.value for anim in AnimationType], {"default": AnimationType.NONE.value}),
                "animation_duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "animation_amplitude": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 10.0}),
                "easing_function": ([ease.value for ease in EasingFunction], {"default": EasingFunction.EASE_IN_OUT.value}),
            },
            "optional": {
                "anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "mask_blur": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "edge_feathering": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "remove_small_objects": ("BOOLEAN", {"default": True}),
                "drop_shadow": ("BOOLEAN", {"default": False}),
                "shadow_offset_x": ("FLOAT", {"default": 5.0, "min": -50.0, "max": 50.0, "step": 1.0}),
                "shadow_offset_y": ("FLOAT", {"default": 5.0, "min": -50.0, "max": 50.0, "step": 1.0}),
                "shadow_blur": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "shadow_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "animation_frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "animation_loops": ("INT", {"default": 1, "min": -1, "max": 20, "step": 1}),
                "ping_pong": ("BOOLEAN", {"default": False}),
                "start_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 1}),
                "processing_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "use_gpu_acceleration": ("BOOLEAN", {"default": True}),
                "batch_processing": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("composite_result", "foreground_mask", "foreground_removed_bg")
    FUNCTION = "process"
    CATEGORY = "image/video/background_removal"
    
    def process(self, foreground, background, skip_background_removal=False, removal_method="rmbg_v2_birefnet", chroma_color="green",
               chroma_tolerance=0.15, chroma_softness=0.8, spill_suppression=0.9,
               position_mode="center", x_position=0.0, y_position=0.0, scale_x=1.0, scale_y=1.0, rotation=0.0,
               blend_mode="normal", opacity=1.0, animation_type="none", animation_duration=2.0,
               animation_amplitude=100.0, easing_function="ease_in_out",
               anchor_x=0.5, anchor_y=0.5, flip_horizontal=False, flip_vertical=False,
               mask_blur=1.5, edge_feathering=3.0, remove_small_objects=True,
               drop_shadow=False, shadow_offset_x=5.0, shadow_offset_y=5.0, shadow_blur=10.0, shadow_opacity=0.5,
               animation_frequency=1.0, animation_loops=1, ping_pong=False, start_delay=0.0,
               video_fps=24.0, total_frames=0, processing_resolution=1024, use_gpu_acceleration=True, batch_processing=True):
        
        try:
            fg_batch_size = foreground.shape[0]
            bg_batch_size = background.shape[0]
            
            if total_frames == 0:
                total_frames = max(fg_batch_size, bg_batch_size)
            
            removal_config = BackgroundRemovalConfig(
                method=RemovalMethod(removal_method),
                chroma_color=chroma_color,
                tolerance=chroma_tolerance,
                softness=chroma_softness,
                spill_suppression=spill_suppression,
                edge_feathering=edge_feathering,
                mask_blur=mask_blur,
                remove_small_objects=remove_small_objects,
                processing_resolution=processing_resolution,
                use_gpu_acceleration=use_gpu_acceleration
            )
            
            position_config = PositionConfig(
                mode=PositionMode(position_mode),
                x_offset=x_position,
                y_offset=y_position,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=rotation,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical
            )
            
            blend_config = BlendingConfig(
                blend_mode=BlendMode(blend_mode),
                opacity=opacity,
                drop_shadow=drop_shadow,
                shadow_offset_x=shadow_offset_x,
                shadow_offset_y=shadow_offset_y,
                shadow_blur=shadow_blur,
                shadow_opacity=shadow_opacity
            )
            
            animation_config = AnimationConfig(
                animation_type=AnimationType(animation_type),
                duration=animation_duration,
                easing=EasingFunction(easing_function),
                amplitude=animation_amplitude,
                frequency=animation_frequency,
                start_delay=start_delay,
                loop_count=animation_loops,
                ping_pong=ping_pong
            )
            
            composite_frames = []
            mask_frames = []
            removed_frames = []
            
            frame_duration = 1.0 / video_fps if video_fps > 0 else 1.0 / 24.0
            total_duration = total_frames * frame_duration
            
            for frame_idx in range(total_frames):
                fg_idx = frame_idx % fg_batch_size
                bg_idx = frame_idx % bg_batch_size
                
                fg_pil = tensor2pil(foreground[fg_idx])
                bg_pil = tensor2pil(background[bg_idx])
                
                if skip_background_removal:
                    if fg_pil.mode != 'RGBA':
                        fg_pil = fg_pil.convert('RGBA')
                    fg_removed = fg_pil
                    fg_mask = fg_pil.split()[3] if fg_pil.mode == 'RGBA' else Image.new('L', fg_pil.size, 255)
                    logger.info(f"Skipped background removal for frame {frame_idx + 1}")
                else:
                    fg_removed, fg_mask = self.background_remover.remove_background(fg_pil, removal_config)
                
                current_time = frame_idx * frame_duration
                
                animated_position = self.animator.animate_layer(
                    position_config, animation_config, current_time, total_duration
                )
                
                animated_blend = copy.deepcopy(blend_config)
                if animation_type in ["fade_in", "fade_out", "fade_pulse"]:
                    t = (current_time - start_delay) / animation_duration if animation_duration > 0 else 0
                    t = max(0, min(1, t))
                    eased_t = EASING_FUNCTIONS[EasingFunction(easing_function)](t)
                    
                    if animation_type == "fade_in":
                        animated_blend.opacity *= eased_t
                    elif animation_type == "fade_out":
                        animated_blend.opacity *= (1 - eased_t)
                    elif animation_type == "fade_pulse":
                        pulse = 0.5 + 0.5 * math.sin(eased_t * animation_frequency * 2 * math.pi)
                        animated_blend.opacity *= pulse
                
                composite = self.compositor.composite_layers(bg_pil, fg_removed, animated_position, animated_blend)
                
                composite_frames.append(pil2tensor(composite))
                mask_frames.append(pil2tensor(fg_mask.convert('RGB')))
                removed_frames.append(pil2tensor(fg_removed))
                
                if frame_idx % 10 == 0:
                    logger.info(f"Processed frame {frame_idx + 1}/{total_frames}")
            
            composite_result = torch.cat(composite_frames, dim=0)
            mask_result = torch.cat(mask_frames, dim=0)
            removed_result = torch.cat(removed_frames, dim=0)
            
            logger.info(f"Successfully processed {total_frames} frames with {removal_method if not skip_background_removal else 'no background removal'} and {animation_type} animation")
            
            return (composite_result, mask_result, removed_result)
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return (background, torch.zeros_like(background), foreground)

# === NODE REGISTRATION ===
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemBv4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB v4.0 - Ultimate AI Background Removal & Video Layering"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
