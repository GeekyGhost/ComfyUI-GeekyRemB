import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from rembg import remove, new_session
from enum import Enum, auto
import math
from tqdm import tqdm
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
import logging
import warnings
from threading import Lock
from multiprocessing import cpu_count
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        print(f"Error converting tensor to PIL: {str(e)}")
        return Image.new('RGB', (image.shape[-2], image.shape[-1]), (0, 0, 0))

def pil2tensor(image):
    """Convert a PIL Image to a PyTorch tensor"""
    try:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    except Exception as e:
        print(f"Error converting PIL to tensor: {str(e)}")
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

class AnimationType(Enum):
    NONE = "none"
    BOUNCE = "bounce"
    TRAVEL_LEFT = "travel_left"
    TRAVEL_RIGHT = "travel_right"
    ROTATE = "rotate"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SCALE_BOUNCE = "scale_bounce"  # New animation type
    SPIRAL = "spiral"  # New animation type

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters"""
    enable_background_removal: bool = True
    removal_method: str = "rembg"
    model: str = "u2net"
    chroma_key_color: str = "green"
    chroma_key_tolerance: float = 0.1
    mask_expansion: int = 0
    edge_detection: bool = False
    edge_thickness: int = 1
    mask_blur: int = 0
    threshold: float = 0.5
    invert_generated_mask: bool = False
    remove_small_regions: bool = False
    small_region_size: int = 100
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10

class EnhancedBlendMode:
    """Enhanced blend mode operations with optimized processing"""
    
    @staticmethod
    def _ensure_rgba(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype) * 255
                return np.concatenate([img, alpha], axis=-1)
            return img
        return np.stack([img] * 4, axis=-1)

    @staticmethod
    def _apply_blend(target: np.ndarray, blend: np.ndarray, operation, opacity: float = 1.0) -> np.ndarray:
        target = EnhancedBlendMode._ensure_rgba(target).astype(np.float32)
        blend = EnhancedBlendMode._ensure_rgba(blend).astype(np.float32)
        
        target = target / 255.0
        blend = blend / 255.0
        
        target_rgb = target[..., :3]
        blend_rgb = blend[..., :3]
        target_a = target[..., 3:4]
        blend_a = blend[..., 3:4]
        
        result_rgb = operation(target_rgb, blend_rgb)
        result_a = target_a + blend_a * (1 - target_a) * opacity
        
        result = np.concatenate([
            result_rgb * opacity + target_rgb * (1 - opacity),
            result_a
        ], axis=-1)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    @classmethod
    def get_blend_modes(cls) -> Dict:
        return {
            "normal": cls.normal,
            "multiply": cls.multiply,
            "screen": cls.screen,
            "overlay": cls.overlay,
            "soft_light": cls.soft_light,
            "hard_light": cls.hard_light,
            "difference": cls.difference,
            "exclusion": cls.exclusion,
            "color_dodge": cls.color_dodge,
            "color_burn": cls.color_burn,
            "linear_light": cls.linear_light,  # New blend mode
            "pin_light": cls.pin_light,  # New blend mode
        }

    @staticmethod
    def normal(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: b, opacity)

    @staticmethod
    def multiply(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: t * b, opacity)

    @staticmethod
    def screen(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: 1 - (1 - t) * (1 - b), opacity)

    @staticmethod
    def overlay(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def overlay_op(t, b):
            return np.where(t > 0.5, 1 - 2 * (1 - t) * (1 - b), 2 * t * b)
        return EnhancedBlendMode._apply_blend(target, blend, overlay_op, opacity)

    @staticmethod
    def soft_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def soft_light_op(t, b):
            return np.where(b > 0.5,
                          t + (2 * b - 1) * (t - t * t),
                          t - (1 - 2 * b) * t * (1 - t))
        return EnhancedBlendMode._apply_blend(target, blend, soft_light_op, opacity)

    @staticmethod
    def hard_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def hard_light_op(t, b):
            return np.where(b > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return EnhancedBlendMode._apply_blend(target, blend, hard_light_op, opacity)

    @staticmethod
    def difference(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: np.abs(t - b), opacity)

    @staticmethod
    def exclusion(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: t + b - 2 * t * b, opacity)

    @staticmethod
    def color_dodge(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def color_dodge_op(t, b):
            return np.where(b >= 1, 1, np.minimum(1, t / (1 - b + 1e-6)))
        return EnhancedBlendMode._apply_blend(target, blend, color_dodge_op, opacity)

    @staticmethod
    def color_burn(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def color_burn_op(t, b):
            return np.where(b <= 0, 0, np.maximum(0, 1 - (1 - t) / (b + 1e-6)))
        return EnhancedBlendMode._apply_blend(target, blend, color_burn_op, opacity)

    @staticmethod
    def linear_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def linear_light_op(t, b):
            return np.clip(2 * b + t - 1, 0, 1)
        return EnhancedBlendMode._apply_blend(target, blend, linear_light_op, opacity)

    @staticmethod
    def pin_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def pin_light_op(t, b):
            return np.where(b > 0.5,
                          np.maximum(t, 2 * (b - 0.5)),
                          np.minimum(t, 2 * b))
        return EnhancedBlendMode._apply_blend(target, blend, pin_light_op, opacity)

class EnhancedMaskProcessor:
    """Enhanced mask processing with advanced refinement techniques"""
    
    @staticmethod
    def refine_mask(mask: Image.Image, config: ProcessingConfig) -> Image.Image:
        mask_np = np.array(mask)
        
        # Enhanced thresholding
        if config.threshold > 0:
            mask_np = cv2.threshold(
                mask_np, 
                int(config.threshold * 255), 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        
        # Enhanced edge detection
        if config.edge_detection:
            edges = cv2.Canny(mask_np, 100, 200)
            kernel = np.ones((config.edge_thickness, config.edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            mask_np = cv2.addWeighted(mask_np, 1, edges, 0.5, 0)
        
        # Enhanced morphological operations
        if config.mask_expansion != 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (abs(config.mask_expansion), abs(config.mask_expansion))
            )
            if config.mask_expansion > 0:
                mask_np = cv2.dilate(mask_np, kernel)
            else:
                mask_np = cv2.erode(mask_np, kernel)
        
        # Enhanced blur
        if config.mask_blur > 0:
            mask_np = cv2.GaussianBlur(
                mask_np,
                (config.mask_blur * 2 + 1, config.mask_blur * 2 + 1),
                0
            )
        
        # Enhanced small region removal
        if config.remove_small_regions:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                mask_np, 
                connectivity=8
            )
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            
            min_size = config.small_region_size
            img2 = np.zeros(output.shape)
            
            for i in range(nb_components):
                if sizes[i] >= min_size:
                    img2[output == i + 1] = 255
            
            mask_np = img2.astype(np.uint8)
        
        mask = Image.fromarray(mask_np)
        
        if config.invert_generated_mask:
            mask = ImageOps.invert(mask)
        
        return mask

class EnhancedAnimator:
    """Enhanced animation processing with additional effects"""
    
    @staticmethod
    def animate_element(
        element: Image.Image,
        animation_type: str,
        animation_speed: float,
        frame_number: int,
        total_frames: int,
        x_start: int,
        y_start: int,
        canvas_width: int,
        canvas_height: int,
        scale: float,
        rotation: float
    ) -> Tuple[Image.Image, int, int]:
        progress = frame_number / total_frames
        orig_width, orig_height = element.size
        
        if element.mode != 'RGBA':
            element = element.convert('RGBA')
        
        new_size = (int(orig_width * scale), int(orig_height * scale))
        element = element.resize(new_size, Image.LANCZOS)
        
        rotated = Image.new('RGBA', element.size, (0, 0, 0, 0))
        rotated.paste(element, (0, 0), element)
        
        if rotation != 0:
            rotated = rotated.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=True,
                center=(rotated.width // 2, rotated.height // 2)
            )
        
        x, y = x_start, y_start
        
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            y += y_offset
        
        elif animation_type == AnimationType.SCALE_BOUNCE.value:
            scale_factor = 1 + math.sin(progress * 2 * math.pi) * animation_speed * 0.2
            new_size = (int(rotated.width * scale_factor), int(rotated.height * scale_factor))
            rotated = rotated.resize(new_size, Image.LANCZOS)
            x -= (new_size[0] - rotated.width) // 2
            y -= (new_size[1] - rotated.height) // 2
        
        elif animation_type == AnimationType.SPIRAL.value:
            radius = 50 * animation_speed
            angle = progress * 4 * math.pi
            x += int(radius * math.cos(angle))
            y += int(radius * math.sin(angle))
            rotated = rotated.rotate(
                angle * 180 / math.pi,
                resample=Image.BICUBIC,
                expand=True,
                center=(rotated.width // 2, rotated.height // 2)
            )
        
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(canvas_width - (canvas_width + rotated.width) * progress)
        
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(-rotated.width + (canvas_width + rotated.width) * progress)
        
        elif animation_type == AnimationType.ROTATE.value:
            angle = progress * 360 * animation_speed
            rotated = rotated.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True,
                center=(rotated.width // 2, rotated.height // 2)
            )
        
        elif animation_type == AnimationType.FADE_IN.value:
            opacity = int(progress * 255)
            r, g, b, a = rotated.split()
            a = a.point(lambda i: i * opacity // 255)
            rotated = Image.merge('RGBA', (r, g, b, a))
        
        elif animation_type == AnimationType.FADE_OUT.value:
            opacity = int((1 - progress) * 255)
            r, g, b, a = rotated.split()
            a = a.point(lambda i: i * opacity // 255)
            rotated = Image.merge('RGBA', (r, g, b, a))
        
        elif animation_type == AnimationType.ZOOM_IN.value:
            zoom_scale = 1 + progress * animation_speed
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            rotated = rotated.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the zoomed image
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            rotated = rotated.crop((left, top, right, bottom))
        
        elif animation_type == AnimationType.ZOOM_OUT.value:
            zoom_scale = 1 + (1 - progress) * animation_speed
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            rotated = rotated.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the zoomed image
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            rotated = rotated.crop((left, top, right, bottom))

        return rotated, x, y

class EnhancedGeekyRemB:
    def __init__(self):
        self.session = None
        self.session_lock = Lock()
        self.use_gpu = torch.cuda.is_available()
        self.config = ProcessingConfig()
        self.blend_modes = EnhancedBlendMode.get_blend_modes()
        self.mask_processor = EnhancedMaskProcessor()
        self.animator = EnhancedAnimator()
        
        # Enhanced thread pool configuration with proper error handling
        try:
            cpu_cores = cpu_count()
        except:
            cpu_cores = os.cpu_count() or 4  # Fallback if cpu_count fails
            
        self.max_workers = min(cpu_cores, 8)  # Limit to reasonable number
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Enhanced cache configuration
        self.frame_cache = LRUCache(maxsize=100)

    def cleanup(self):
        """Enhanced cleanup resources with better error handling"""
        try:
            if hasattr(self, 'session') and self.session is not None:
                self.session = None
            
            if hasattr(self, 'executor'):
                try:
                    self.executor.shutdown(wait=False)
                except:
                    pass
                self.executor = None
            
            if hasattr(self, 'frame_cache'):
                try:
                    self.frame_cache.clear()
                except:
                    pass
                self.frame_cache = None
            
            # Force garbage collection
            gc.collect()
            if self.use_gpu:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Enhanced destructor with better error handling"""
        try:
            self.cleanup()
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_format": (["RGBA", "RGB"],),
                "foreground": ("IMAGE",),
                "enable_background_removal": ("BOOLEAN", {"default": True}),
                "removal_method": (["rembg", "chroma_key"],),
                "model": ([
                    "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
                    "silueta", "isnet-general-use", "isnet-anime"
                ],),
                "chroma_key_color": (["green", "blue", "red"],),
                "chroma_key_tolerance": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_generated_mask": ("BOOLEAN", {"default": False}),
                "remove_small_regions": ("BOOLEAN", {"default": False}),
                "small_region_size": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "animation_type": ([anim.value for anim in AnimationType],),
                "animation_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "animation_frames": ("INT", {"default": 30, "min": 1, "max": 3000, "step": 1}),
                "x_position": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "blend_mode": (list(EnhancedBlendMode.get_blend_modes().keys()),),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "aspect_ratio": ("STRING", {
                    "default": "", 
                    "placeholder": "e.g., 16:9, 4:3, 1:1, portrait, landscape"
                })
            },
            "optional": {
                "background": ("IMAGE",),
                "additional_mask": ("MASK",),
                "invert_additional_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "image/processing"

    def initialize_model(self, model: str) -> None:
        """Thread-safe model initialization"""
        with self.session_lock:
            if self.session is None or self.session.model_name != model:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                try:
                    self.session = new_session(model, providers=providers)
                except Exception as e:
                    logger.error(f"Failed to initialize model: {str(e)}")
                    raise RuntimeError(f"Model initialization failed: {str(e)}")

    def remove_background_rembg(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Enhanced background removal using rembg"""
        try:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            result = remove(
                image,
                session=self.session,
                alpha_matting=self.config.alpha_matting,
                alpha_matting_foreground_threshold=self.config.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=self.config.alpha_matting_background_threshold
            )
            
            return result, result.split()[3]
        except Exception as e:
            logger.error(f"Background removal failed: {str(e)}")
            raise RuntimeError(f"Background removal failed: {str(e)}")

    def remove_background_chroma(self, image: Image.Image) -> Image.Image:
        """Enhanced chroma key background removal"""
        try:
            img_np = np.array(image)
            if img_np.shape[2] == 4:
                img_np = img_np[:,:,:3]
            
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Enhanced color ranges with better tolerance handling
            if self.config.chroma_key_color == "green":
                lower = np.array([60 - 30*self.config.chroma_key_tolerance, 100, 100])
                upper = np.array([60 + 30*self.config.chroma_key_tolerance, 255, 255])
            elif self.config.chroma_key_color == "blue":
                lower = np.array([120 - 30*self.config.chroma_key_tolerance, 100, 100])
                upper = np.array([120 + 30*self.config.chroma_key_tolerance, 255, 255])
            else:  # red
                lower = np.array([0, 100, 100])
                upper = np.array([30*self.config.chroma_key_tolerance, 255, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = 255 - mask
            
            # Enhanced mask cleanup
            mask = cv2.medianBlur(mask, 3)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            return Image.fromarray(mask)
        except Exception as e:
            logger.error(f"Chroma key removal failed: {str(e)}")
            raise RuntimeError(f"Chroma key removal failed: {str(e)}")

    def process_frame(self, frame: Image.Image, background_frame: Optional[Image.Image], 
                     frame_number: int, total_frames: int) -> Tuple[Image.Image, Image.Image]:
        """Enhanced frame processing with improved error handling"""
        try:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            # Handle background removal
            if self.config.enable_background_removal:
                if self.config.removal_method == "rembg":
                    frame_with_alpha, mask = self.remove_background_rembg(frame)
                    frame = frame_with_alpha
                else:
                    mask = self.remove_background_chroma(frame)
                    frame = Image.composite(
                        frame, 
                        Image.new('RGBA', frame.size, (0, 0, 0, 0)), 
                        mask
                    )

                mask = self.mask_processor.refine_mask(mask, self.config)

            else:
                mask = Image.new('L', frame.size, 255)

            # Apply animation
            animated_frame, x, y = self.animator.animate_element(
                frame,
                self.animation_type,
                self.animation_speed,
                frame_number,
                total_frames,
                self.x_position,
                self.y_position,
                background_frame.width if background_frame else frame.width,
                background_frame.height if background_frame else frame.height,
                self.scale,
                self.rotation
            )

            # Handle background blending
            if background_frame is not None:
                bg_width, bg_height = background_frame.size
                
                if self.blend_mode != "normal":
                    canvas = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
                    paste_x = max(0, min(int(x), bg_width - animated_frame.width))
                    paste_y = max(0, min(int(y), bg_height - animated_frame.height))
                    canvas.paste(animated_frame, (paste_x, paste_y), animated_frame)
                    
                    bg_array = np.array(background_frame)
                    canvas_array = np.array(canvas)
                    
                    result_array = self.blend_modes[self.blend_mode](
                        bg_array, 
                        canvas_array, 
                        self.opacity
                    )
                    result = Image.fromarray(result_array)
                else:
                    result = background_frame.copy().convert('RGBA')
                    result.alpha_composite(animated_frame, (int(x), int(y)))
            else:
                result = animated_frame

            return result, mask

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return frame, Image.new('L', frame.size, 255)

    def process_image(self, output_format, foreground, enable_background_removal, removal_method, model,
                     chroma_key_color, chroma_key_tolerance, mask_expansion, edge_detection,
                     edge_thickness, mask_blur, threshold, invert_generated_mask,
                     remove_small_regions, small_region_size, alpha_matting,
                     alpha_matting_foreground_threshold, alpha_matting_background_threshold,
                     animation_type, animation_speed, animation_frames, x_position, y_position,
                     scale, rotation, blend_mode, opacity, aspect_ratio, background=None,
                     additional_mask=None, invert_additional_mask=False):
        try:
            # Store animation and processing parameters
            self.animation_type = animation_type
            self.animation_speed = animation_speed
            self.x_position = x_position
            self.y_position = y_position
            self.scale = scale
            self.rotation = rotation
            self.blend_mode = blend_mode
            self.opacity = opacity
            
            # Update config
            self.config.enable_background_removal = enable_background_removal
            self.config.removal_method = removal_method
            self.config.chroma_key_color = chroma_key_color
            self.config.chroma_key_tolerance = chroma_key_tolerance
            self.config.mask_expansion = mask_expansion
            self.config.edge_detection = edge_detection
            self.config.edge_thickness = edge_thickness
            self.config.mask_blur = mask_blur
            self.config.threshold = threshold
            self.config.invert_generated_mask = invert_generated_mask
            self.config.remove_small_regions = remove_small_regions
            self.config.small_region_size = small_region_size
            self.config.alpha_matting = alpha_matting
            self.config.alpha_matting_foreground_threshold = alpha_matting_foreground_threshold
            self.config.alpha_matting_background_threshold = alpha_matting_background_threshold

            debug_tensor_info(foreground, "Input foreground")
            
            if enable_background_removal and removal_method == "rembg":
                self.initialize_model(model)

            # Convert inputs to PIL images
            fg_frames = [tensor2pil(foreground[i]) for i in range(foreground.shape[0])]
            bg_frames = [tensor2pil(background[i]) for i in range(background.shape[0])] if background is not None else None

            if bg_frames and len(bg_frames) < animation_frames:
                bg_frames = bg_frames * (animation_frames // len(bg_frames) + 1)
                bg_frames = bg_frames[:animation_frames]

            # Parse and apply aspect ratio if specified
            aspect_ratio_value = self.parse_aspect_ratio(aspect_ratio)
            if aspect_ratio_value is not None:
                for i in range(len(fg_frames)):
                    new_width = int(fg_frames[i].width * scale)
                    new_height = int(new_width / aspect_ratio_value)
                    fg_frames[i] = fg_frames[i].resize((new_width, new_height), Image.LANCZOS)

            animated_frames = []
            masks = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for frame in range(animation_frames):
                    fg_index = frame % len(fg_frames)
                    bg_frame = bg_frames[frame % len(bg_frames)] if bg_frames else None
                    
                    future = executor.submit(
                        self.process_frame,
                        fg_frames[fg_index],
                        bg_frame,
                        frame,
                        animation_frames
                    )
                    futures.append(future)
                
                # Process results as they complete
                for future in tqdm(futures, desc="Processing frames"):
                    try:
                        result_frame, mask = future.result()
                        
                        # Handle additional mask if provided
                        if additional_mask is not None:
                            additional_mask_pil = tensor2pil(
                                additional_mask[len(animated_frames) % len(additional_mask)]
                            )
                            if invert_additional_mask:
                                additional_mask_pil = ImageOps.invert(additional_mask_pil)
                            mask = Image.fromarray(
                                np.minimum(np.array(mask), np.array(additional_mask_pil))
                            )
                        
                        # Cache results
                        frame_key = f"frame_{len(animated_frames)}"
                        self.frame_cache[frame_key] = (result_frame, mask)
                        
                        animated_frames.append(pil2tensor(result_frame))
                        masks.append(pil2tensor(mask.convert('L')))
                        
                    except Exception as e:
                        logger.error(f"Error processing frame {len(animated_frames)}: {str(e)}")
                        if animated_frames:
                            animated_frames.append(animated_frames[-1])
                            masks.append(masks[-1])
                        else:
                            blank_frame = Image.new('RGBA', fg_frames[0].size, (0, 0, 0, 0))
                            blank_mask = Image.new('L', fg_frames[0].size, 0)
                            animated_frames.append(pil2tensor(blank_frame))
                            masks.append(pil2tensor(blank_mask))

            # Convert output format if needed
            if output_format == "RGB":
                for i in range(len(animated_frames)):
                    frame = tensor2pil(animated_frames[i])
                    frame = frame.convert('RGB')
                    animated_frames[i] = pil2tensor(frame)

            # Cleanup and return results
            try:
                result = torch.cat(animated_frames, dim=0)
                result_masks = torch.cat(masks, dim=0)
                debug_tensor_info(result, "Output result")
                debug_tensor_info(result_masks, "Output masks")
                return (result, result_masks)
            except Exception as e:
                logger.error(f"Error concatenating results: {str(e)}")
                return (foreground, torch.zeros_like(foreground[:, :1, :, :]))
        
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return (foreground, torch.zeros_like(foreground[:, :1, :, :]))

    def parse_aspect_ratio(self, aspect_ratio_input: str) -> Optional[float]:
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

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session is not None:
                self.session = None
            
            self.executor.shutdown(wait=True)
            self.frame_cache.clear()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.cleanup()

# Helper class for frame caching
class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, maxsize: int = 100):
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
        self.lock = Lock()
    
    def __getitem__(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end to mark as recently used
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            raise KeyError(key)
    
    def __setitem__(self, key, value):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used item
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": EnhancedGeekyRemB
}

# Display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB"
}
