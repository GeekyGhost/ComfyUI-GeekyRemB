import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from rembg import remove, new_session
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import gc
import logging

# Import utilities
from .utils.image_utils import tensor2pil, pil2tensor, debug_tensor_info, parse_aspect_ratio
from .utils.cache import LRUCache
from .utils.mask_processing import EnhancedMaskProcessor, MaskProcessingConfig, remove_background_chroma, ChromaKeyConfig
from .utils.blend_modes import EnhancedBlendMode
from .utils.animation import EnhancedAnimator, AnimationType, EASING_FUNCTIONS
from .utils.light_shadow_util import LightShadowProcessor, LightingEffectConfig, ShadowEffectConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BackgroundRemovalConfig:
    """Configuration for background removal processing"""
    enable_background_removal: bool = True
    removal_method: str = "rembg"  # "rembg" or "chroma_key"
    model: str = "u2net"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    
    # For chroma key
    chroma_key_color: str = "green"
    chroma_key_tolerance: float = 0.1
    spill_reduction: float = 0.0
    edge_refinement: bool = False
    
    # For mask processing
    mask_expansion: int = 0
    edge_detection: bool = False
    edge_thickness: int = 1
    edge_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    mask_blur: int = 5
    threshold: float = 0.5
    invert_generated_mask: bool = False
    remove_small_regions: bool = False
    small_region_size: int = 100

class GeekyRemB:
    """Core background removal node with advanced mask processing"""
    
    def __init__(self):
        self.session = None
        self.session_lock = Lock()
        self.use_gpu = torch.cuda.is_available()
        self.config = BackgroundRemovalConfig()
        self.mask_processor = EnhancedMaskProcessor()
        self.light_config = LightingEffectConfig(enable_lighting=False)  # Disabled by default
        self.shadow_config = ShadowEffectConfig(enable_shadow=False)    # Disabled by default
        
        # Thread pool for parallel processing
        cpu_cores = os.cpu_count() or 4
        self.max_workers = min(cpu_cores, 8)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Cache for processed frames
        self.frame_cache = LRUCache(maxsize=100)

    def cleanup_resources(self):
        """Clean up resources"""
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
        """Ensure proper cleanup on deletion"""
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
                "spill_reduction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_refinement": ("BOOLEAN", {"default": False}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "mask_blur": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_generated_mask": ("BOOLEAN", {"default": False}),
                "remove_small_regions": ("BOOLEAN", {"default": False}),
                "small_region_size": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "aspect_ratio": ("STRING", {
                    "default": "", 
                    "placeholder": "e.g., 16:9, 4:3, 1:1, portrait, landscape"
                }),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "frames": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "x_position": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
            },
            "optional": {
                "background": ("IMAGE",),
                "additional_mask": ("MASK",),
                "invert_additional_mask": ("BOOLEAN", {"default": False}),
                # Update auxiliary node inputs
                "animator": ("ANIMATOR",),  # Connection for GeekyRemB_Animator
                "lightshadow": ("LIGHTSHADOW",),  # Connection for GeekyRemB_LightShadow
                "keyframe": ("KEYFRAME",),  # Direct connection for individual keyframes - replacing keyframe_animator
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "image/processing"
    DISPLAY_NAME = "Geeky RemB"

    def initialize_model(self, model: str) -> None:
        """Thread-safe model initialization"""
        with self.session_lock:
            if self.session is None or getattr(self.session, 'model_name', None) != model:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                try:
                    self.session = new_session(model, providers=providers)
                    logger.info(f"Model '{model}' initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize model '{model}': {str(e)}")
                    raise RuntimeError(f"Model initialization failed: {str(e)}")

    def remove_background_rembg(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Enhanced background removal using rembg with alpha matting"""
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

    def process_frame(self, frame: Image.Image, background_frame: Optional[Image.Image], x_pos: int = 0, y_pos: int = 0) -> Tuple[Image.Image, Image.Image]:
        """Process a single frame for background removal"""
        try:
            logger.info(f"Processing frame with background: {background_frame is not None}")
            if background_frame:
                logger.info(f"Background frame size: {background_frame.size}")
            logger.info(f"Foreground frame size: {frame.size}")
            
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Ensure RGBA mode
            frame = frame.convert('RGBA')

            if self.config.enable_background_removal:
                if self.config.removal_method == "rembg":
                    frame_with_alpha, mask = self.remove_background_rembg(frame)
                    frame = frame_with_alpha
                else:
                    # Create chroma key config
                    chroma_config = ChromaKeyConfig(
                        chroma_key_color=self.config.chroma_key_color,
                        chroma_key_tolerance=self.config.chroma_key_tolerance,
                        spill_reduction=self.config.spill_reduction,
                        edge_refinement=self.config.edge_refinement
                    )
                    
                    # Get mask using chroma key
                    mask = remove_background_chroma(frame, chroma_config)
                    
                    # Create mask processing config
                    mask_config = MaskProcessingConfig(
                        mask_expansion=self.config.mask_expansion,
                        edge_detection=self.config.edge_detection,
                        edge_thickness=self.config.edge_thickness,
                        edge_color=self.config.edge_color,
                        mask_blur=self.config.mask_blur,
                        threshold=self.config.threshold,
                        invert_generated_mask=self.config.invert_generated_mask,
                        remove_small_regions=self.config.remove_small_regions,
                        small_region_size=self.config.small_region_size,
                        edge_refinement=self.config.edge_refinement
                    )
                    
                    # Refine mask
                    refined_mask = self.mask_processor.refine_mask(mask, mask_config, frame)
                    
                    # Create new frame with refined alpha
                    frame_array = np.array(frame)
                    mask_array = np.array(refined_mask)
                    
                    # Ensure mask is in correct shape for alpha channel
                    if len(mask_array.shape) == 2:
                        mask_array = mask_array[:, :, None]

                    # Apply mask to RGB channels
                    frame_array = frame_array * (mask_array / 255.0)
                    frame_array[:, :, 3] = mask_array[:, :, 0]  # Set alpha channel
                    
                    frame = Image.fromarray(frame_array.astype(np.uint8), 'RGBA')
                    mask = refined_mask
            else:
                mask = Image.new('L', frame.size, 255)

            # Apply light and shadow effects if enabled
            # Apply lighting effect if enabled
            if self.light_config.enable_lighting:
                frame = LightShadowProcessor.apply_lighting_effect(frame, self.light_config)
            
            # Create shadow if enabled
            if self.shadow_config.enable_shadow:
                # Calculate shadow offset
                shadow_offset = (
                    self.shadow_config.shadow_direction_x,
                    self.shadow_config.shadow_direction_y
                )
                
                # Create shadow from the frame
                shadow = LightShadowProcessor.create_shadow(frame, self.shadow_config)
                
                if shadow:
                    # Create new canvas with background first if available
                    if background_frame is not None:
                        canvas = background_frame.convert('RGBA')
                    else:
                        canvas = Image.new('RGBA', frame.size, (0, 0, 0, 0))
                    
                    # Paste shadow without offset (shadow already contains the offset)
                    canvas.paste(shadow, (0, 0), shadow)
                    
                    # Paste original frame on top with position offset
                    canvas.paste(frame, (x_pos, y_pos), frame)
                    frame = canvas
            
            # Handle background composition if provided and shadow is not enabled
            elif background_frame is not None:
                # Make sure the background frame is ready for composition
                bg = background_frame.convert('RGBA')
                
                # Create a new canvas with the appropriate size
                if bg.size != frame.size:
                    # If sizes differ, create a canvas with background size
                    result = Image.new('RGBA', bg.size, (0, 0, 0, 0))
                    # Place the background first
                    result.paste(bg, (0, 0))
                    
                    # Calculate center position if not provided
                    if x_pos == 0 and y_pos == 0:
                        x_pos = (bg.width - frame.width) // 2
                        y_pos = (bg.height - frame.height) // 2
                        
                    # Place the foreground on top
                    result.paste(frame, (x_pos, y_pos), frame)
                else:
                    # If sizes match, create a canvas with that size
                    result = Image.new('RGBA', frame.size, (0, 0, 0, 0))
                    result.paste(bg, (0, 0))
                    result.paste(frame, (x_pos, y_pos), frame)
                
                frame = result

            return frame, mask

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return frame, Image.new('L', frame.size, 255)

    def _process_foreground_only(self, frame: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Process a single frame for background removal and lighting only, no background compositing"""
        try:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Ensure RGBA mode
            frame = frame.convert('RGBA')

            # Handle background removal
            if self.config.enable_background_removal:
                if self.config.removal_method == "rembg":
                    frame_with_alpha, mask = self.remove_background_rembg(frame)
                    frame = frame_with_alpha
                else:
                    # Create chroma key config
                    chroma_config = ChromaKeyConfig(
                        chroma_key_color=self.config.chroma_key_color,
                        chroma_key_tolerance=self.config.chroma_key_tolerance,
                        spill_reduction=self.config.spill_reduction,
                        edge_refinement=self.config.edge_refinement
                    )
                    
                    # Get mask using chroma key
                    mask = remove_background_chroma(frame, chroma_config)
                    
                    # Create mask processing config
                    mask_config = MaskProcessingConfig(
                        mask_expansion=self.config.mask_expansion,
                        edge_detection=self.config.edge_detection,
                        edge_thickness=self.config.edge_thickness,
                        edge_color=self.config.edge_color,
                        mask_blur=self.config.mask_blur,
                        threshold=self.config.threshold,
                        invert_generated_mask=self.config.invert_generated_mask,
                        remove_small_regions=self.config.remove_small_regions,
                        small_region_size=self.config.small_region_size,
                        edge_refinement=self.config.edge_refinement
                    )
                    
                    # Refine mask
                    refined_mask = self.mask_processor.refine_mask(mask, mask_config, frame)
                    
                    # Create new frame with refined alpha
                    frame_array = np.array(frame)
                    mask_array = np.array(refined_mask)
                    
                    # Ensure mask is in correct shape for alpha channel
                    if len(mask_array.shape) == 2:
                        mask_array = mask_array[:, :, None]

                    # Apply mask to RGB channels
                    frame_array = frame_array * (mask_array / 255.0)
                    frame_array[:, :, 3] = mask_array[:, :, 0]  # Set alpha channel
                    
                    frame = Image.fromarray(frame_array.astype(np.uint8), 'RGBA')
                    mask = refined_mask
            else:
                mask = Image.new('L', frame.size, 255)

            # Apply lighting effect if enabled (but not shadow)
            if self.light_config.enable_lighting:
                frame = LightShadowProcessor.apply_lighting_effect(frame, self.light_config)
            
            # Return processed foreground only (no shadow or background composition)
            return frame, mask

        except Exception as e:
            logger.error(f"Foreground processing failed: {str(e)}")
            return frame, Image.new('L', frame.size, 255)

    def process_image(self, output_format, foreground, enable_background_removal, removal_method, model,
                     chroma_key_color, chroma_key_tolerance, spill_reduction, edge_refinement,
                     mask_expansion, edge_detection, edge_thickness, mask_blur, threshold, 
                     invert_generated_mask, remove_small_regions, small_region_size, alpha_matting,
                     alpha_matting_foreground_threshold, alpha_matting_background_threshold, 
                     aspect_ratio, scale, frames, x_position, y_position, background=None, additional_mask=None, 
                     invert_additional_mask=False, animator=None, lightshadow=None, keyframe=None):
        try:
            # Debug logging for input parameters
            logger.info(f"Foreground batch size: {foreground.shape[0]}")
            if background is not None:
                logger.info(f"Background batch size: {background.shape[0]}")
            else:
                logger.info("No background provided")
                
            # Update configuration
            self.config.enable_background_removal = enable_background_removal
            self.config.removal_method = removal_method
            self.config.model = model
            self.config.chroma_key_color = chroma_key_color
            self.config.chroma_key_tolerance = chroma_key_tolerance
            self.config.spill_reduction = spill_reduction
            self.config.edge_refinement = edge_refinement
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
        
            # Initialize AI model if needed
            if enable_background_removal and removal_method == "rembg":
                self.initialize_model(model)
                
            # Check if lightshadow node is connected and initialize configs
            if lightshadow is not None:
                light_shadow_params = lightshadow.get("effect_params", {})
                
                # Update light config
                if "light_config" in light_shadow_params:
                    self.light_config = LightingEffectConfig(
                        enable_lighting=light_shadow_params["light_config"]["enable_lighting"],
                        light_intensity=light_shadow_params["light_config"]["light_intensity"],
                        light_direction_x=light_shadow_params["light_config"]["light_direction_x"],
                        light_direction_y=light_shadow_params["light_config"]["light_direction_y"],
                        light_radius=light_shadow_params["light_config"]["light_radius"],
                        light_falloff=light_shadow_params["light_config"]["light_falloff"],
                        light_color=light_shadow_params["light_config"]["light_color"],
                        light_from_behind=light_shadow_params["light_config"].get("light_from_behind", False),
                        
                        # Advanced lighting parameters
                        enable_normal_mapping=light_shadow_params["light_config"].get("enable_normal_mapping", False),
                        enable_specular=light_shadow_params["light_config"].get("enable_specular", False),
                        specular_intensity=light_shadow_params["light_config"].get("specular_intensity", 0.3),
                        specular_shininess=light_shadow_params["light_config"].get("specular_shininess", 32),
                        ambient_light=light_shadow_params["light_config"].get("ambient_light", 0.2),
                        light_source_height=light_shadow_params["light_config"].get("light_source_height", 200),
                        kelvin_temperature=light_shadow_params["light_config"].get("kelvin_temperature", 0)
                    )
                
                # Update shadow config
                if "shadow_config" in light_shadow_params:
                    self.shadow_config = ShadowEffectConfig(
                        enable_shadow=light_shadow_params["shadow_config"]["enable_shadow"],
                        shadow_opacity=light_shadow_params["shadow_config"]["shadow_opacity"],
                        shadow_blur=light_shadow_params["shadow_config"]["shadow_blur"],
                        shadow_direction_x=light_shadow_params["shadow_config"]["shadow_direction_x"],
                        shadow_direction_y=light_shadow_params["shadow_config"]["shadow_direction_y"],
                        shadow_color=light_shadow_params["shadow_config"]["shadow_color"],
                        shadow_expansion=light_shadow_params["shadow_config"]["shadow_expansion"],
                        
                        # Advanced shadow parameters
                        perspective_shadow=light_shadow_params["shadow_config"].get("perspective_shadow", False),
                        light_source_height=light_shadow_params["shadow_config"].get("light_source_height", 200),
                        distance_fade=light_shadow_params["shadow_config"].get("distance_fade", False),
                        fade_distance=light_shadow_params["shadow_config"].get("fade_distance", 100),
                        soft_edges=light_shadow_params["shadow_config"].get("soft_edges", True)
                    )
            else:
                # Disable light and shadow effects if node is not connected
                self.light_config.enable_lighting = False
                self.shadow_config.enable_shadow = False

            # Convert inputs to PIL images
            fg_frames = [tensor2pil(foreground[i]) for i in range(foreground.shape[0])]
            bg_frames = [tensor2pil(background[i]) for i in range(background.shape[0])] if background is not None else None

            # After converting to PIL images - debug logging
            logger.info(f"Number of foreground frames: {len(fg_frames)}")
            if bg_frames:
                logger.info(f"Number of background frames: {len(bg_frames)}")
                logger.info(f"First background frame size: {bg_frames[0].size}")
            logger.info(f"First foreground frame size: {fg_frames[0].size}")

            # Parse and apply aspect ratio if specified
            aspect_ratio_value = parse_aspect_ratio(aspect_ratio)
            if aspect_ratio_value is not None:
                for i in range(len(fg_frames)):
                    new_width = int(fg_frames[i].width * scale)
                    new_height = int(new_width / aspect_ratio_value)
                    fg_frames[i] = fg_frames[i].resize((new_width, new_height), Image.LANCZOS)

            # Check if animation is enabled through animator
            animation_enabled = animator is not None
            
            # If direct keyframe is provided but no animator, log warning
            if keyframe is not None and not animation_enabled:
                logger.warning("Keyframe provided but no animator is connected. Keyframe will be ignored.")
                
            # If animation is enabled, create multiple frames
            if animation_enabled:
                # Extract animation parameters
                animation_params = animator.get("animation_params", {})
                
                # Check if this is keyframe animation
                animation_type_value = animation_params.get("animation_type", "none")
                is_keyframe_animation = animation_type_value == "keyframe"
                
                logger.info(f"Animation type: {animation_type_value}, Is keyframe animation: {is_keyframe_animation}")
                
                if is_keyframe_animation:
                    # Extract keyframe specific parameters
                    keyframes = animation_params.get("keyframes", [])
                    fps = animation_params.get("fps", 30)
                    animation_frames = animation_params.get("total_frames", frames)  # Use total_frames directly from keyframe params
                    easing_function = animation_params.get("default_easing", "linear")
                    
                    # Log keyframe details
                    logger.info(f"Using keyframe animation with {len(keyframes)} keyframes for {animation_frames} total frames")
                    for i, kf in enumerate(keyframes):
                        kf_frame = kf.get("frame", 0)
                        kf_pos = kf.get("position", (0, 0))
                        kf_scale = kf.get("scale", 1.0)
                        kf_rot = kf.get("rotation", 0.0)
                        logger.info(f"Keyframe {i}: frame={kf_frame}, position={kf_pos}, scale={kf_scale}, rotation={kf_rot}")
                    
                    # Get easing function
                    easing_func = EASING_FUNCTIONS.get(easing_function, lambda t: t)
                    
                    # Set placeholder values that will be overridden by keyframe interpolation
                    animation_speed = 1.0
                    repeats = 1
                    reverse = False
                    delay = 0.0
                    x_pos = animation_params.get("base_position", (x_position, y_position))[0]
                    y_pos = animation_params.get("base_position", (x_position, y_position))[1]
                    animation_scale = animation_params.get("base_scale", scale)
                    rotation = animation_params.get("base_rotation", 0.0)
                    steps = 1
                    phase_shift = 0.0
                    
                    # Setup keyframe parameters
                    keyframe_parameters = animation_params
                    
                else:
                    # Standard animation parameters
                    animation_speed = animation_params.get("animation_speed", 1.0)
                    frame_count = animation_params.get("frame_count", frames)  # Get frame_count directly
                    animation_frames = frame_count  # Use frame_count as animation_frames
                    repeats = animation_params.get("repeats", 1)
                    reverse = animation_params.get("reverse", False)
                    easing_function = animation_params.get("easing_function", "linear")
                    delay = animation_params.get("delay", 0.0)
                    x_pos = animation_params.get("x_position", x_position)
                    y_pos = animation_params.get("y_position", y_position)
                    animation_scale = animation_params.get("scale", 1.0)
                    rotation = animation_params.get("rotation", 0.0)
                    steps = animation_params.get("steps", 1)
                    phase_shift = animation_params.get("phase_shift", 0.0)
                    
                    # Get easing function
                    easing_func = EASING_FUNCTIONS.get(easing_function, lambda t: t)
                    
                    # No keyframe parameters for standard animation
                    keyframe_parameters = None
                
                # Process frames with animation
                processed_frames = []
                masks = []
                
                # Handle background frames for animation
                if bg_frames:
                    if len(bg_frames) == 1:
                        # If only one background frame, replicate it for all frames
                        bg_frames = [bg_frames[0]] * animation_frames
                    elif len(bg_frames) < animation_frames:
                        # If not enough background frames, extend by repeating
                        bg_frames = bg_frames * (animation_frames // len(bg_frames) + 1)
                        bg_frames = bg_frames[:animation_frames]
                
                # Process each frame with background removal and lighting effects, but NOT shadow or background compositing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for i in range(len(fg_frames)):
                        future = executor.submit(
                            # Modified process_frame call that skips background compositing
                            lambda frame: self._process_foreground_only(frame),
                            fg_frames[i]
                        )
                        futures.append(future)
                
                    # Process results as they complete
                    for i, future in enumerate(tqdm(futures, desc="Processing foregrounds")):
                        try:
                            result_frame, mask = future.result()

                # Handle additional mask if provided
                            if additional_mask is not None:
                                additional_mask_pil = tensor2pil(
                                    additional_mask[i % len(additional_mask)]
                                )
                                if invert_additional_mask:
                                    additional_mask_pil = ImageOps.invert(additional_mask_pil)
                                mask = Image.fromarray(
                                    np.minimum(np.array(mask), np.array(additional_mask_pil))
                                )
                        
                            # Cache results
                            frame_key = f"frame_{i}"
                            self.frame_cache[frame_key] = (result_frame, mask)
                        
                            processed_frames.append(result_frame)
                            masks.append(mask.convert('L'))
                        
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            if processed_frames:
                                processed_frames.append(processed_frames[-1])
                                masks.append(masks[-1])
                            else:
                                blank_frame = Image.new('RGBA', fg_frames[0].size, (0, 0, 0, 0))
                                blank_mask = Image.new('L', fg_frames[0].size, 0)
                                processed_frames.append(blank_frame)
                                masks.append(blank_mask)
                
                # Create animated sequence
                animated_frames = []
                animated_masks = []
                
                # Get dimensions for output canvas
                example_frame = processed_frames[0]
                frame_width, frame_height = example_frame.size
                
                # If background provided, use its dimensions for canvas
                if bg_frames:
                    bg_width, bg_height = bg_frames[0].size
                    canvas_width, canvas_height = bg_width, bg_height
                else:
                    # Create canvas with enough space for animations
                    padding = int(100 * animation_speed)
                    canvas_width = frame_width + padding * 2
                    canvas_height = frame_height + padding * 2
                
                # Pre-compute shadows if shadow is enabled
                shadow_frames = []
                if self.shadow_config.enable_shadow:
                    for frame in processed_frames:
                        shadow = LightShadowProcessor.create_shadow(frame, self.shadow_config)
                        shadow_frames.append(shadow)
                
                # Process each frame index
                for frame_idx in range(animation_frames):
                    # Calculate which source frame to use (loop if needed)
                    source_idx = frame_idx % len(processed_frames)
                    frame = processed_frames[source_idx]
                    current_mask = masks[source_idx]
                    
                    # Get background for this frame if available
                    bg = bg_frames[frame_idx % len(bg_frames)] if bg_frames else None
                    
                    # Create initial canvas with background
                    if bg:
                        canvas = bg.convert('RGBA')
                    else:
                        canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
                    
                    # Debug log for keyframe animation
                    if is_keyframe_animation:
                        logger.info(f"Processing keyframe animation frame {frame_idx}/{animation_frames}")
                    
                    # Apply animation to frame
                    animated_frame, x, y = EnhancedAnimator.animate_element(
                        frame,
                        animation_type_value,
                        animation_speed,
                        frame_idx,
                        animation_frames,
                        x_pos,
                        y_pos,
                        canvas_width,
                        canvas_height,
                        animation_scale,
                        rotation,
                        easing_func,
                        repeats,
                        reverse,
                        delay,
                        steps,
                        phase_shift,
                        keyframe_parameters  # Pass keyframe parameters
                    )
                    
                    # Log position for debugging
                    if is_keyframe_animation and frame_idx % 10 == 0:
                        logger.info(f"Frame {frame_idx}: Position = ({x}, {y})")
                    
                    # Apply shadow if enabled
                    if self.shadow_config.enable_shadow and shadow_frames[source_idx]:
                        # Animate the shadow with same parameters
                        animated_shadow, shadow_x, shadow_y = EnhancedAnimator.animate_element(
                            shadow_frames[source_idx],
                            animation_type_value,
                            animation_speed,
                            frame_idx,
                            animation_frames,
                            x_pos + self.shadow_config.shadow_direction_x,
                            y_pos + self.shadow_config.shadow_direction_y,
                            canvas_width,
                            canvas_height,
                            animation_scale,
                            rotation,
                            easing_func,
                            repeats,
                            reverse,
                            delay,
                            steps,
                            phase_shift,
                            keyframe_parameters  # Pass keyframe parameters
                        )
                        
                        # Paste shadow onto canvas
                        canvas.paste(animated_shadow, (shadow_x, shadow_y), animated_shadow)
                    
                    # Position the animated frame on canvas
                    canvas.paste(animated_frame, (x, y), animated_frame)
                    
                    # Handle mask similarly
                    mask_canvas = Image.new('L', (canvas_width, canvas_height), 0)
                    animated_mask, _, _ = EnhancedAnimator.animate_element(
                        current_mask,
                        animation_type_value,
                        animation_speed,
                        frame_idx,
                        animation_frames,
                        x_pos,
                        y_pos,
                        canvas_width,
                        canvas_height,
                        animation_scale,
                        rotation,
                        easing_func,
                        repeats,
                        reverse,
                        delay,
                        steps,
                        phase_shift,
                        keyframe_parameters  # Pass keyframe parameters
                    )
                    mask_canvas.paste(animated_mask, (x, y))
                    
                    # Convert to desired output format
                    if output_format == "RGB":
                        canvas = canvas.convert('RGB')
                    
                    # Add to result lists
                    animated_frames.append(pil2tensor(canvas))
                    animated_masks.append(pil2tensor(mask_canvas))
                
                # Combine all frames into a single tensor
                result = torch.cat(animated_frames, dim=0)
                result_masks = torch.cat(animated_masks, dim=0)
                
            else:
                # No animation, just process frames normally
                processed_frames = []
                masks = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for i in range(len(fg_frames)):
                        # Modified background handling logic
                        if bg_frames:
                            # If we have only one background frame but multiple foreground frames,
                            # reuse the same background for all foreground frames
                            if len(bg_frames) == 1 and len(fg_frames) > 1:
                                bg_frame = bg_frames[0]
                            else:
                                # Otherwise use the corresponding frame if available
                                bg_frame = bg_frames[i] if i < len(bg_frames) else None
                        else:
                            bg_frame = None
                        
                        future = executor.submit(
                            self.process_frame,
                            fg_frames[i],
                            bg_frame,
                            x_position,  # Use x_position parameter
                            y_position   # Use y_position parameter
                        )
                        futures.append(future)
                
                    # Process results as they complete
                    for i, future in enumerate(tqdm(futures, desc="Processing frames")):
                        try:
                            result_frame, mask = future.result()
                        
                            # Handle additional mask if provided
                            if additional_mask is not None:
                                additional_mask_pil = tensor2pil(
                                    additional_mask[i % len(additional_mask)]
                                )
                                if invert_additional_mask:
                                    additional_mask_pil = ImageOps.invert(additional_mask_pil)
                                mask = Image.fromarray(
                                    np.minimum(np.array(mask), np.array(additional_mask_pil))
                                )
                        
                            # Cache results
                            frame_key = f"frame_{i}"
                            self.frame_cache[frame_key] = (result_frame, mask)
                        
                            processed_frames.append(pil2tensor(result_frame))
                            masks.append(pil2tensor(mask.convert('L')))
                        
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            if processed_frames:
                                processed_frames.append(processed_frames[-1])
                                masks.append(masks[-1])
                            else:
                                blank_frame = Image.new('RGBA', fg_frames[0].size, (0, 0, 0, 0))
                                blank_mask = Image.new('L', fg_frames[0].size, 0)
                                processed_frames.append(pil2tensor(blank_frame))
                                masks.append(pil2tensor(blank_mask))

                # Convert output format if needed
                if output_format == "RGB":
                    for i in range(len(processed_frames)):
                        frame = tensor2pil(processed_frames[i])
                        frame = frame.convert('RGB')
                        processed_frames[i] = pil2tensor(frame)

                # Combine initial results
                result = torch.cat(processed_frames, dim=0)
                result_masks = torch.cat(masks, dim=0)
                
            return (result, result_masks)

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return (foreground, torch.zeros_like(foreground[:, :1, :, :]))
