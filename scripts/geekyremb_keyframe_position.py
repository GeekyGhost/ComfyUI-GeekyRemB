import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeekyRemB_KeyframePosition:
    """Position keyframe node with 2D slider for visual positioning of animated elements"""
    
    def __init__(self):
        self.frame = 0
        self.canvas_width = 512
        self.canvas_height = 512
        self.position = (256, 256)  # Default to center
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "frame_number": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "x_position": ("INT", {"default": 256, "min": -2048, "max": 2048, "step": 1}),
                "y_position": ("INT", {"default": 256, "min": -2048, "max": 2048, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "easing": (list(["linear", "ease_in_quad", "ease_out_quad", "ease_in_out_quad", 
                                 "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
                                 "ease_in_sine", "ease_out_sine", "ease_in_out_sine",
                                 "ease_in_expo", "ease_out_expo", "ease_in_out_expo",
                                 "ease_in_bounce", "ease_out_bounce", "ease_in_out_bounce"]),),
            }
        }

    RETURN_TYPES = ("KEYFRAME",)
    FUNCTION = "provide_keyframe"
    CATEGORY = "image/animation"
    DISPLAY_NAME = "Geeky RemB Keyframe Position"

    def provide_keyframe(self, width, height, frame_number, x_position, y_position, 
                         scale=1.0, rotation=0.0, opacity=1.0, easing="linear"):
        """Generate keyframe data for a specific position and frame"""
        
        # Store canvas settings
        self.canvas_width = width
        self.canvas_height = height
        self.frame = frame_number
        self.position = (x_position, y_position)
        
        # Create keyframe data
        keyframe = {
            "frame": frame_number,
            "position": (x_position, y_position),
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
            "easing": easing
        }
        
        logger.info(f"Created position keyframe at frame {frame_number}, position ({x_position}, {y_position})")
        return (keyframe,)