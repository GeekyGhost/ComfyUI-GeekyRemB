from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging
import os
import gc

# Import utilities
from .utils.cache import LRUCache
from .utils.animation import AnimationType, EASING_FUNCTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeekyRemB_Animator:
    """Animation module for GeekyRemB with keyframe support"""
    
    def __init__(self):
        self.use_gpu = False  # No need for GPU in parameter provider
        
        # Thread pool for parallel processing
        cpu_cores = os.cpu_count() or 4
        self.max_workers = min(cpu_cores, 8)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Cache for processed frames
        self.frame_cache = LRUCache(maxsize=100)

    def cleanup_resources(self):
        """Clean up resources"""
        try:
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
                "animation_type": ([anim.value for anim in AnimationType],),
                "animation_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "animation_duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "reverse": ("BOOLEAN", {"default": False}),
                "easing_function": (list(EASING_FUNCTIONS.keys()),),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "use_keyframes": ("BOOLEAN", {"default": False}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "x_position": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "phase_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Keyframe inputs
                "keyframe1": ("KEYFRAME",),
                "keyframe2": ("KEYFRAME",),
                "keyframe3": ("KEYFRAME",),
                "keyframe4": ("KEYFRAME",),
                "keyframe5": ("KEYFRAME",),
            }
        }

    RETURN_TYPES = ("ANIMATOR",)
    FUNCTION = "provide_animator"
    CATEGORY = "image/animation"
    DISPLAY_NAME = "Geeky RemB Animator"

    def provide_animator(self, output_format, animation_type, animation_speed, animation_duration,
                       repeats, reverse, easing_function, fps, use_keyframes, delay, x_position, y_position,
                       scale, rotation, steps, phase_shift,
                       keyframe1=None, keyframe2=None, keyframe3=None, keyframe4=None, keyframe5=None):
        """Provides animation parameters to the main GeekyRemB node"""
        
        # Check if keyframes are provided and should be used
        if use_keyframes:
            keyframes = []
            # Add provided keyframes to the list, ignore None values
            if keyframe1 is not None:
                keyframes.append(keyframe1)
                logger.info(f"Added keyframe1: {keyframe1}")
            if keyframe2 is not None:
                keyframes.append(keyframe2)
                logger.info(f"Added keyframe2: {keyframe2}")
            if keyframe3 is not None:
                keyframes.append(keyframe3)
                logger.info(f"Added keyframe3: {keyframe3}")
            if keyframe4 is not None:
                keyframes.append(keyframe4)
                logger.info(f"Added keyframe4: {keyframe4}")
            if keyframe5 is not None:
                keyframes.append(keyframe5)
                logger.info(f"Added keyframe5: {keyframe5}")
            
            if keyframes:
                # Sort keyframes by frame number
                keyframes.sort(key=lambda k: k.get("frame", 0))
                
                # Create keyframe animation parameters with explicit animation_type=keyframe
                keyframe_params = {
                    "animation_type": "keyframe",  # Explicitly set as keyframe
                    "fps": fps,
                    "keyframes": keyframes,
                    "default_easing": easing_function,
                    "duration": animation_duration,
                    "total_frames": int(animation_duration * fps),
                    # Add base values for interpolation reference
                    "base_position": (x_position, y_position),
                    "base_scale": scale,
                    "base_rotation": rotation
                }
                
                # Log keyframe details for debugging
                for i, kf in enumerate(keyframes):
                    frame = kf.get("frame", 0)
                    pos = kf.get("position", (0, 0))
                    scl = kf.get("scale", 1.0)
                    rot = kf.get("rotation", 0.0)
                    logger.info(f"Keyframe {i}: frame={frame}, position={pos}, scale={scl}, rotation={rot}")
                
                logger.info(f"Using keyframe animation with {len(keyframes)} keyframes for {keyframe_params['total_frames']} total frames")
                return ({"animation_params": keyframe_params},)
        
        # If no keyframes or use_keyframes is False, use standard animation
        animation_params = {
            "output_format": output_format,
            "animation_type": animation_type,
            "animation_speed": animation_speed,
            "animation_duration": animation_duration,
            "repeats": repeats,
            "reverse": reverse,
            "easing_function": easing_function,
            "fps": fps,
            "delay": delay,
            "x_position": x_position,
            "y_position": y_position,
            "scale": scale,
            "rotation": rotation,
            "steps": steps,
            "phase_shift": phase_shift
        }
        
        return ({"animation_params": animation_params},)