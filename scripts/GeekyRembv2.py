import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from rembg import remove, new_session
from enum import Enum, auto
from dataclasses import dataclass
import math
from tqdm import tqdm
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Union, Dict, Callable
from threading import Lock
from multiprocessing import cpu_count
import os
import gc
import logging
import warnings

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

# Easing Functions
def linear(t):
    return t

def ease_in_quad(t):
    return t * t

def ease_out_quad(t):
    return t * (2 - t)

def ease_in_out_quad(t):
    return 2*t*t if t < 0.5 else -1 + (4 - 2*t)*t

def ease_in_cubic(t):
    return t ** 3

def ease_out_cubic(t):
    return (t - 1) ** 3 + 1

def ease_in_out_cubic(t):
    return 4*t*t*t if t < 0.5 else (t-1)*(2*t-2)*(2*t-2)+1

# Add more easing functions as needed

EASING_FUNCTIONS = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    # Add more mappings
}

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
    SCALE_BOUNCE = "scale_bounce"  # Existing animation type
    SPIRAL = "spiral"  # Existing animation type
    SHAKE = "shake"  # New animation type
    SLIDE_UP = "slide_up"  # New animation type
    SLIDE_DOWN = "slide_down"  # New animation type
    FLIP_HORIZONTAL = "flip_horizontal"  # New animation type
    FLIP_VERTICAL = "flip_vertical"  # New animation type
    WAVE = "wave"  # New animation type
    PULSE = "pulse"  # New animation type
    SWING = "swing"  # New animation type
    SPIN = "spin"  # Additional new animation type
    FLASH = "flash"  # Additional new animation type

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
    # New parameters
    easing_function: str = "linear"  # Default easing
    repeats: int = 1  # Number of repeats
    reverse: bool = False  # Whether to reverse after each repeat
    delay: float = 0.0  # Delay before animation starts (in seconds or frames)
    animation_duration: float = 1.0  # Duration of one animation cycle
    steps: int = 1  # Number of steps in animation
    phase_shift: float = 0.0  # Phase shift for staggered animations

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
    def get_blend_modes(cls) -> Dict[str, Callable]:
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
            # Add more blend modes as needed
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
            _, mask_np = cv2.threshold(
                mask_np, 
                int(config.threshold * 255), 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        # Enhanced edge detection
        if config.edge_detection:
            edges = cv2.Canny(mask_np, 100, 200)
            kernel = np.ones((config.edge_thickness, config.edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            mask_np = cv2.addWeighted(mask_np, 1, edges, 0.5, 0)
        
        # Enhanced morphological operations
        if config.mask_expansion != 0:
            kernel_size = max(1, abs(config.mask_expansion))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (kernel_size, kernel_size)
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
            nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
                mask_np, 
                connectivity=8
            )
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            
            min_size = config.small_region_size
            img2 = np.zeros(output.shape, dtype=np.uint8)
            
            for i in range(nb_components):
                if sizes[i] >= min_size:
                    img2[output == i + 1] = 255
            
            mask_np = img2
        
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
        rotation: float,
        easing_func: Callable[[float], float],
        repeat: int,
        reverse: bool,
        delay: float,
        steps: int = 1,
        phase_shift: float = 0.0
    ) -> Tuple[Image.Image, int, int]:
        # Adjust frame_number based on delay
        adjusted_frame = frame_number - int(delay * total_frames)
        if adjusted_frame < 0:
            return element, x_start, y_start  # No animation yet
        
        # Handle repeats
        cycle_length = total_frames / repeat
        current_cycle = int(adjusted_frame / cycle_length)
        frame_in_cycle = adjusted_frame % cycle_length
        progress = frame_in_cycle / cycle_length  # Normalized progress within the cycle
        
        # Apply easing function
        progress = easing_func(progress)
        
        # Handle reverse
        if reverse and current_cycle % 2 == 1:
            progress = 1 - progress
        
        orig_width, orig_height = element.size
        
        if element.mode != 'RGBA':
            element = element.convert('RGBA')
        
        # Calculate the bounding box of visible pixels to determine rotation center
        bbox = element.getbbox()
        if bbox:
            cropped = element.crop(bbox)
            center_x = cropped.width // 2
            center_y = cropped.height // 2
        else:
            cropped = element
            center_x, center_y = element.width // 2, element.height // 2
        
        # Apply scaling
        new_size = (int(orig_width * scale), int(orig_height * scale))
        element = element.resize(new_size, Image.LANCZOS)
        
        # Apply rotation around the center of visible pixels
        if rotation != 0:
            # Calculate bounding box for the rotated image
            rotated_image = element.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=True
            )
            rotated_width, rotated_height = rotated_image.size
            new_canvas = Image.new("RGBA", (rotated_width, rotated_height), (0, 0, 0, 0))
            offset_x = (rotated_width - element.width) // 2
            offset_y = (rotated_height - element.height) // 2
            new_canvas.paste(element, (offset_x, offset_y))
            element = new_canvas.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=False
            )
        
        x, y = x_start, y_start
        
        # Apply steps and phase_shift for staggered animations
        if steps > 1:
            step_progress = progress * steps
            current_step = int(step_progress)
            progress = step_progress - current_step
            x += int(phase_shift * current_step)
            y += int(phase_shift * current_step)
            progress = min(progress, 1.0)
        
        # Existing animation types with adjusted progress
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            y += y_offset
        
        elif animation_type == AnimationType.SCALE_BOUNCE.value:
            scale_factor = 1 + math.sin(progress * 2 * math.pi) * animation_speed * 0.2
            scaled_size = (int(element.width * scale_factor), int(element.height * scale_factor))
            element = element.resize(scaled_size, Image.LANCZOS)
            x -= (scaled_size[0] - new_size[0]) // 2
            y -= (scaled_size[1] - new_size[1]) // 2
        
        elif animation_type == AnimationType.SPIRAL.value:
            radius = 50 * animation_speed
            angle = progress * 4 * math.pi
            x += int(radius * math.cos(angle))
            y += int(radius * math.sin(angle))
            element = element.rotate(
                angle * 180 / math.pi,
                resample=Image.BICUBIC,
                expand=True,
                center=(center_x, center_y)
            )
        
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(canvas_width - (canvas_width + element.width) * progress)
        
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(-element.width + (canvas_width + element.width) * progress)
        
        elif animation_type == AnimationType.ROTATE.value:
            spin_speed = 360 * animation_speed  # degrees per cycle
            angle = progress * spin_speed
            element = element.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True,
                center=(center_x, center_y)
            )
        
        elif animation_type == AnimationType.FADE_IN.value:
            opacity = int(progress * 255)
            r, g, b, a = element.split()
            a = a.point(lambda i: i * opacity // 255)
            element = Image.merge('RGBA', (r, g, b, a))
        
        elif animation_type == AnimationType.FADE_OUT.value:
            opacity = int((1 - progress) * 255)
            r, g, b, a = element.split()
            a = a.point(lambda i: i * opacity // 255)
            element = Image.merge('RGBA', (r, g, b, a))
        
        elif animation_type == AnimationType.ZOOM_IN.value:
            zoom_scale = 1 + progress * animation_speed
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            element = element.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the zoomed image
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            element = element.crop((left, top, right, bottom))
        
        elif animation_type == AnimationType.ZOOM_OUT.value:
            zoom_scale = 1 + (1 - progress) * animation_speed
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            element = element.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the zoomed image
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            element = element.crop((left, top, right, bottom))
        
        # New animation types with adjusted progress
        elif animation_type == AnimationType.SHAKE.value:
            shake_amplitude = 10 * animation_speed
            x_offset = int(math.sin(progress * 10 * math.pi) * shake_amplitude)
            y_offset = int(math.cos(progress * 10 * math.pi) * shake_amplitude)
            x += x_offset
            y += y_offset
        
        elif animation_type == AnimationType.SLIDE_UP.value:
            y = int(y_start - (y_start + element.height) * progress)
        
        elif animation_type == AnimationType.SLIDE_DOWN.value:
            y = int(-element.height + (canvas_height + element.height) * progress)
        
        elif animation_type == AnimationType.FLIP_HORIZONTAL.value:
            if progress > 0.5:
                element = element.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif animation_type == AnimationType.FLIP_VERTICAL.value:
            if progress > 0.5:
                element = element.transpose(Image.FLIP_TOP_BOTTOM)
        
        elif animation_type == AnimationType.WAVE.value:
            wave_amplitude = 20 * animation_speed
            wave_frequency = 2
            y_offset = int(math.sin(progress * wave_frequency * 2 * math.pi) * wave_amplitude)
            y += y_offset
        
        elif animation_type == AnimationType.PULSE.value:
            pulse_scale = 1 + 0.3 * math.sin(progress * 4 * math.pi) * animation_speed
            new_size = (int(orig_width * scale * pulse_scale), int(orig_height * scale * pulse_scale))
            element = element.resize(new_size, Image.LANCZOS)
            x -= (new_size[0] - orig_width * scale) // 2
            y -= (new_size[1] - orig_height * scale) // 2
        
        elif animation_type == AnimationType.SWING.value:
            swing_amplitude = 15 * animation_speed
            swing_angle = math.sin(progress * 4 * math.pi) * swing_amplitude
            element = element.rotate(
                swing_angle,
                resample=Image.BICUBIC,
                expand=True,
                center=(center_x, center_y)
            )
        
        elif animation_type == AnimationType.SPIN.value:
            spin_speed = 360 * animation_speed  # degrees per cycle
            angle = progress * spin_speed
            element = element.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True,
                center=(center_x, center_y)
            )
        
        elif animation_type == AnimationType.FLASH.value:
            brightness = 1 + 0.5 * math.sin(progress * 2 * math.pi) * animation_speed
            enhancer = ImageEnhance.Brightness(element)
            element = enhancer.enhance(brightness)
        
        # Add more animation types as needed
        
        return element, x, y

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

    def cleanup_resources(self):
        """Enhanced cleanup resources with better error handling"""
        try:
            if self.session is not None:
                self.session = None
            
            if self.executor is not None:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as e:
                    logger.error(f"Error shutting down executor: {str(e)}")
                self.executor = None
            
            if self.frame_cache is not None:
                try:
                    self.frame_cache.clear()
                except Exception as e:
                    logger.error(f"Error clearing cache: {str(e)}")
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
            self.cleanup_resources()
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
                "animation_duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "reverse": ("BOOLEAN", {"default": False}),
                "easing_function": (list(EASING_FUNCTIONS.keys()),),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
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
                }),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "phase_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
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
                    logger.info(f"Model '{model}' initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize model '{model}': {str(e)}")
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
                lower = np.array([60 - int(30 * self.config.chroma_key_tolerance), 100, 100])
                upper = np.array([60 + int(30 * self.config.chroma_key_tolerance), 255, 255])
            elif self.config.chroma_key_color == "blue":
                lower = np.array([120 - int(30 * self.config.chroma_key_tolerance), 100, 100])
                upper = np.array([120 + int(30 * self.config.chroma_key_tolerance), 255, 255])
            else:  # red
                lower = np.array([0, 100, 100])
                upper = np.array([30, 255, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = 255 - mask  # Invert mask to get foreground
            
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

            # Apply animation with new parameters
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
                self.rotation,
                EASING_FUNCTIONS.get(self.config.easing_function, linear),
                self.config.repeats,
                self.config.reverse,
                self.config.delay,
                steps=self.config.steps,
                phase_shift=self.config.phase_shift
            )

            # Validate blend mode
            if self.blend_mode not in self.blend_modes:
                logger.warning(f"Unsupported blend mode '{self.blend_mode}'. Falling back to 'normal'.")
                blend_mode_func = self.blend_modes["normal"]
            else:
                blend_mode_func = self.blend_modes[self.blend_mode]

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
                    
                    # Ensure both arrays are in the same color space
                    if bg_array.shape[2] != canvas_array.shape[2]:
                        bg_array = EnhancedBlendMode._ensure_rgba(bg_array)
                    
                    result_array = blend_mode_func(
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
                     animation_type, animation_speed, animation_duration, repeats, reverse,
                     easing_function, delay, animation_frames, x_position, y_position,
                     scale, rotation, blend_mode, opacity, aspect_ratio, steps=1,
                     phase_shift=0.0, background=None,
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
            # New config parameters
            self.config.easing_function = easing_function
            self.config.repeats = repeats
            self.config.reverse = reverse
            self.config.delay = delay
            self.config.animation_duration = animation_duration
            self.config.steps = steps
            self.config.phase_shift = phase_shift

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
        self.cleanup_resources()

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
