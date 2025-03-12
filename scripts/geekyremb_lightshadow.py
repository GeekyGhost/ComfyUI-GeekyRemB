from threading import Lock
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeekyRemB_LightShadow:
    """Light and shadow effect provider for GeekyRemB"""
    
    def __init__(self):
        self.lock = Lock()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Lighting effects - Basic
                "enable_lighting": ("BOOLEAN", {"default": True}),
                "light_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "light_direction_x": ("INT", {"default": 0, "min": -200, "max": 200, "step": 5}),
                "light_direction_y": ("INT", {"default": -100, "min": -200, "max": 200, "step": 5}),
                "light_radius": ("INT", {"default": 150, "min": 10, "max": 500, "step": 10}),
                "light_falloff": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "light_from_behind": ("BOOLEAN", {"default": False}),
                
                # Light Color Control
                "use_kelvin_temperature": ("BOOLEAN", {"default": False}),
                "kelvin_temperature": ("INT", {"default": 6500, "min": 2000, "max": 10000, "step": 100}),
                "light_color_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "light_color_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "light_color_b": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                
                # Advanced Lighting
                "enable_normal_mapping": ("BOOLEAN", {"default": False}),
                "enable_specular": ("BOOLEAN", {"default": False}),
                "specular_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "specular_shininess": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "ambient_light": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "light_source_height": ("INT", {"default": 200, "min": 50, "max": 500, "step": 10}),
                
                # Shadow effects - Basic
                "enable_shadow": ("BOOLEAN", {"default": True}),
                "shadow_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "shadow_blur": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "shadow_direction_x": ("INT", {"default": 5, "min": -50, "max": 50, "step": 1}),
                "shadow_direction_y": ("INT", {"default": 5, "min": -50, "max": 50, "step": 1}),
                "shadow_expansion": ("INT", {"default": 0, "min": -10, "max": 20, "step": 1}),
                "shadow_color_r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "shadow_color_g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "shadow_color_b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                
                # Advanced Shadow
                "perspective_shadow": ("BOOLEAN", {"default": False}),
                "distance_fade": ("BOOLEAN", {"default": False}),
                "fade_distance": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "soft_edges": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIGHTSHADOW",)
    FUNCTION = "provide_lightshadow"
    CATEGORY = "image/effects"
    DISPLAY_NAME = "Geeky RemB Light & Shadow"

    def provide_lightshadow(
        self,
        # Lighting effects - Basic 
        enable_lighting, light_intensity, light_direction_x, light_direction_y,
        light_radius, light_falloff, light_from_behind,
        # Light Color Control
        use_kelvin_temperature, kelvin_temperature, light_color_r, light_color_g, light_color_b,
        # Advanced Lighting
        enable_normal_mapping, enable_specular, specular_intensity, specular_shininess,
        ambient_light, light_source_height,
        # Shadow effects - Basic
        enable_shadow, shadow_opacity, shadow_blur, shadow_direction_x, shadow_direction_y,
        shadow_expansion, shadow_color_r, shadow_color_g, shadow_color_b,
        # Advanced Shadow
        perspective_shadow, distance_fade, fade_distance, soft_edges
    ):
        """Provides light and shadow parameters to the main GeekyRemB node"""
        
        # Get light color (either from RGB or Kelvin temperature)
        if use_kelvin_temperature:
            from .utils.light_shadow_util import LightShadowProcessor
            light_color = LightShadowProcessor.kelvin_to_rgb(kelvin_temperature)
        else:
            light_color = (light_color_r, light_color_g, light_color_b)
        
        # Create config dictionary to pass to the main node
        effect_params = {
            "light_config": {
                # Basic lighting
                "enable_lighting": enable_lighting,
                "light_intensity": light_intensity,
                "light_direction_x": light_direction_x,
                "light_direction_y": light_direction_y,
                "light_radius": light_radius,
                "light_falloff": light_falloff,
                "light_from_behind": light_from_behind,
                "light_color": light_color,
                
                # Advanced lighting
                "enable_normal_mapping": enable_normal_mapping,
                "enable_specular": enable_specular,
                "specular_intensity": specular_intensity,
                "specular_shininess": specular_shininess,
                "ambient_light": ambient_light,
                "light_source_height": light_source_height,
                "kelvin_temperature": kelvin_temperature if use_kelvin_temperature else 0
            },
            "shadow_config": {
                # Basic shadow
                "enable_shadow": enable_shadow,
                "shadow_opacity": shadow_opacity,
                "shadow_blur": shadow_blur,
                "shadow_direction_x": shadow_direction_x,
                "shadow_direction_y": shadow_direction_y,
                "shadow_color": (shadow_color_r, shadow_color_g, shadow_color_b),
                "shadow_expansion": shadow_expansion,
                
                # Advanced shadow
                "perspective_shadow": perspective_shadow,
                "light_source_height": light_source_height,
                "distance_fade": distance_fade,
                "fade_distance": fade_distance,
                "soft_edges": soft_edges
            }
        }
        
        return ({"effect_params": effect_params},)