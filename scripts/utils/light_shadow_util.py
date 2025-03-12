import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LightingEffectConfig:
    """Configuration for lighting effects"""
    enable_lighting: bool = True
    light_intensity: float = 0.5
    light_direction_x: int = 0
    light_direction_y: int = -100
    light_radius: int = 150
    light_falloff: float = 1.0
    light_color: Tuple[int, int, int] = (255, 255, 220)  # Slightly warm white
    light_from_behind: bool = False  # Toggle between front and behind lighting
    
    # Enhanced lighting parameters
    enable_normal_mapping: bool = False  # Use normal mapping for 3D effect
    enable_specular: bool = False  # Add specular highlights
    specular_intensity: float = 0.3  # Strength of specular highlights
    specular_shininess: int = 32  # Shininess factor (higher = sharper highlights)
    ambient_light: float = 0.2  # Ambient light factor
    light_source_height: int = 200  # Height of light source for 3D effect
    kelvin_temperature: int = 6500  # Color temperature in Kelvin

@dataclass
class ShadowEffectConfig:
    """Configuration for shadow effects"""
    enable_shadow: bool = True
    shadow_opacity: float = 0.5
    shadow_blur: int = 10
    shadow_direction_x: int = 5
    shadow_direction_y: int = 5
    shadow_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    shadow_expansion: int = 0
    
    # Enhanced shadow parameters
    perspective_shadow: bool = False  # Use perspective projection for shadow
    light_source_height: int = 200  # Height of light source (affects shadow perspective)
    distance_fade: bool = False  # Fade shadow with distance
    fade_distance: int = 100  # Distance at which shadow begins to fade
    soft_edges: bool = True  # Use soft edges for shadow

class LightShadowProcessor:
    """Utility class for applying light and shadow effects to images"""
    
    @staticmethod
    def generate_simple_normal_map(alpha_channel):
        """Generate a simple normal map from the alpha channel edges"""
        try:
            # Convert to float and normalize
            alpha_float = alpha_channel.astype(np.float32) / 255.0
            
            # Apply Sobel filter to get gradients
            from scipy.ndimage import sobel
            dx = sobel(alpha_float, axis=1)  # Horizontal gradient
            dy = sobel(alpha_float, axis=0)  # Vertical gradient
            dz = np.ones_like(dx) * 0.1  # Small z-component for flatness
            
            # Normalize to get unit vectors
            # Add small epsilon to avoid division by zero
            norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            normal_map = np.stack([dx/norm, dy/norm, dz/norm], axis=-1)
            
            return normal_map
        except Exception as e:
            logger.error(f"Error generating normal map: {str(e)}")
            # Return a flat normal map in case of error
            height, width = alpha_channel.shape[:2]
            return np.zeros((height, width, 3), dtype=np.float32)

    @staticmethod
    def kelvin_to_rgb(kelvin):
        """Convert color temperature in Kelvin to RGB values"""
        try:
            # Clamp temperature to valid range
            temperature = max(1000, min(40000, kelvin)) / 100.0
            
            # Calculate red
            if temperature <= 66:
                red = 255
            else:
                red = temperature - 60
                red = 329.698727446 * (red ** -0.1332047592)
                red = max(0, min(255, red))
            
            # Calculate green
            if temperature <= 66:
                green = temperature
                green = 99.4708025861 * math.log(green) - 161.1195681661
            else:
                green = temperature - 60
                green = 288.1221695283 * (green ** -0.0755148492)
            green = max(0, min(255, green))
            
            # Calculate blue
            if temperature >= 66:
                blue = 255
            elif temperature <= 19:
                blue = 0
            else:
                blue = temperature - 10
                blue = 138.5177312231 * math.log(blue) - 305.0447927307
                blue = max(0, min(255, blue))
            
            return (int(red), int(green), int(blue))
        except Exception as e:
            logger.error(f"Error converting Kelvin to RGB: {str(e)}")
            return (255, 255, 255)  # Return white in case of error
    
    @staticmethod
    def apply_lighting_effect(image: Image.Image, config: LightingEffectConfig) -> Image.Image:
        """Apply realistic directional lighting effect to an image with normal mapping and specular"""
        try:
            if not config.enable_lighting or config.light_intensity <= 0:
                return image
            
            # Ensure we're working with RGBA
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Get image data
            img_array = np.array(image).astype(np.float32)
            alpha_channel = img_array[:, :, 3]
            
            # Skip if no visible pixels
            if not np.any(alpha_channel > 0):
                return image
            
            # Convert Kelvin temperature to RGB if enabled
            if hasattr(config, 'kelvin_temperature') and config.kelvin_temperature > 0:
                light_color = LightShadowProcessor.kelvin_to_rgb(config.kelvin_temperature)
            else:
                light_color = config.light_color
            
            # Calculate normalized light direction vector
            light_dir_x = config.light_direction_x
            light_dir_y = config.light_direction_y
            light_z = config.light_source_height if hasattr(config, 'light_source_height') else 200
            light_dir_length = math.sqrt(light_dir_x**2 + light_dir_y**2 + light_z**2)
            
            # Default to top light if direction is zero
            if light_dir_length < 1:
                light_dir_x, light_dir_y, light_z = 0, -1, 1
            else:
                light_dir_x /= light_dir_length
                light_dir_y /= light_dir_length
                light_z /= light_dir_length
            
            # Create light direction vector
            light_dir = np.array([light_dir_x, light_dir_y, light_z])
            
            # Create lighting array (same shape as image)
            height, width = img_array.shape[:2]
            
            # Generate normal map if normal mapping is enabled
            if hasattr(config, 'enable_normal_mapping') and config.enable_normal_mapping:
                normal_map = LightShadowProcessor.generate_simple_normal_map(alpha_channel)
                
                # Calculate diffuse lighting using normal map
                lighting = np.zeros((height, width), dtype=np.float32)
                
                # View direction (assuming viewer is looking straight at the image)
                view_dir = np.array([0, 0, 1])
                
                # Calculate halfway vector for specular (Blinn-Phong)
                if hasattr(config, 'enable_specular') and config.enable_specular:
                    halfway = light_dir + view_dir
                    halfway = halfway / np.linalg.norm(halfway)
                
                # Apply lighting based on normal map
                for y in range(height):
                    for x in range(width):
                        if alpha_channel[y, x] > 0:
                            # Get normal at this pixel
                            normal = normal_map[y, x]
                            
                            # Calculate diffuse lighting (Lambert's law)
                            if config.light_from_behind:
                                # Backlighting effect
                                dot = max(0, -np.dot(normal, light_dir))
                            else:
                                # Front lighting
                                dot = max(0, np.dot(normal, light_dir))
                            
                            lighting[y, x] = dot
                            
                            # Add specular highlight if enabled
                            if hasattr(config, 'enable_specular') and config.enable_specular:
                                spec_dot = max(0, np.dot(normal, halfway))
                                shininess = config.specular_shininess if hasattr(config, 'specular_shininess') else 32
                                spec = spec_dot ** shininess
                                spec_intensity = config.specular_intensity if hasattr(config, 'specular_intensity') else 0.3
                                lighting[y, x] += spec * spec_intensity
                
                # Add ambient light term if enabled
                if hasattr(config, 'ambient_light') and config.ambient_light > 0:
                    ambient = config.ambient_light
                    lighting += ambient
            else:
                # Use the simpler lighting model if normal mapping is disabled
                y_coords, x_coords = np.mgrid[:height, :width]
                x_norm = (x_coords - width/2) / (width/2)
                y_norm = (y_coords - height/2) / (height/2)
                
                if config.light_from_behind:
                    # Light from behind creates a rim light effect
                    rim_strength = np.abs(x_norm * light_dir_x + y_norm * light_dir_y)
                    lighting = np.clip(rim_strength, 0, 1) ** (1/config.light_falloff)
                else:
                    # Front lighting - brightest where surface normal faces light
                    dot_product = -(x_norm * light_dir_x + y_norm * light_dir_y)
                    lighting = np.clip(dot_product, 0, 1) ** (1/config.light_falloff)
                
                # Add ambient light term if enabled
                if hasattr(config, 'ambient_light') and config.ambient_light > 0:
                    lighting += config.ambient_light
            
            # Scale by light intensity
            lighting *= config.light_intensity
            
            # Apply falloff based on distance from center for more natural look
            center_x, center_y = width // 2, height // 2
            x_dist = (x_coords - center_x) / config.light_radius
            y_dist = (y_coords - center_y) / config.light_radius
            distance = np.sqrt(x_dist**2 + y_dist**2)
            falloff_mask = np.clip(1 - distance, 0, 1) ** 0.5
            
            # Combine directional lighting with falloff
            lighting *= falloff_mask
            
            # Ensure lighting is within valid range
            lighting = np.clip(lighting, 0, 1)
            
            # Only apply lighting to non-transparent pixels
            lighting *= (alpha_channel > 0)
            
            # Create RGB lighting array
            light_r = np.ones_like(lighting) * (light_color[0] / 255.0)
            light_g = np.ones_like(lighting) * (light_color[1] / 255.0)
            light_b = np.ones_like(lighting) * (light_color[2] / 255.0)
            
            # Apply lighting using screen blend mode
            # Screen formula: Result = 1 - (1 - Light) * (1 - Base)
            result_r = 1 - (1 - (light_r * lighting))[:, :, np.newaxis] * (1 - img_array[:, :, 0:1] / 255.0)
            result_g = 1 - (1 - (light_g * lighting))[:, :, np.newaxis] * (1 - img_array[:, :, 1:2] / 255.0)
            result_b = 1 - (1 - (light_b * lighting))[:, :, np.newaxis] * (1 - img_array[:, :, 2:3] / 255.0)
            
            # Combine channels and convert back to uint8
            result = np.concatenate([
                (result_r * 255).astype(np.uint8),
                (result_g * 255).astype(np.uint8),
                (result_b * 255).astype(np.uint8),
                img_array[:, :, 3:4].astype(np.uint8)
            ], axis=2)
            
            return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"Error applying lighting effect: {str(e)}")
            return image

    @staticmethod
    def create_shadow(image: Image.Image, config: ShadowEffectConfig) -> Optional[Image.Image]:
        """Create a shadow for the given image with perspective and distance-based fade"""
        try:
            if not config.enable_shadow or config.shadow_opacity <= 0:
                return None
                
            # Get alpha channel to use as shadow base
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
                
            _, _, _, alpha = image.split()
            alpha_np = np.array(alpha)
            
            # Create a mask for the shadow with proper dimensions
            shadow_mask = np.zeros_like(alpha_np, dtype=np.float32)
            
            # Calculate shadow parameters
            offset_x = config.shadow_direction_x
            offset_y = config.shadow_direction_y
            
            # Use perspective shadow if enabled
            if hasattr(config, 'perspective_shadow') and config.perspective_shadow:
                # Calculate center of image (for perspective calculations)
                height, width = alpha_np.shape
                center_x, center_y = width // 2, height // 2
                
                # Get light source height
                light_height = config.light_source_height if hasattr(config, 'light_source_height') else 200
                
                # Process each pixel for perspective shadow
                for y in range(height):
                    for x in range(width):
                        if alpha_np[y, x] > 0:
                            # Calculate depth factor (simulating z-depth)
                            depth_factor = 1.0 - ((y - center_y) / height) * 0.5
                            
                            # Calculate perspective shadow position
                            shadow_x = int(x + offset_x * light_height / (light_height * depth_factor))
                            shadow_y = int(y + offset_y * light_height / (light_height * depth_factor))
                            
                            # Check if shadow position is within bounds
                            if 0 <= shadow_x < width and 0 <= shadow_y < height:
                                # Copy alpha value to shadow mask
                                shadow_mask[shadow_y, shadow_x] = alpha_np[y, x]
            else:
                # Standard shadow without perspective
                # Calculate bounds to ensure shadow stays within image
                height, width = alpha_np.shape
                min_x = max(0, -offset_x)
                min_y = max(0, -offset_y)
                max_x = min(width, width - offset_x)
                max_y = min(height, height - offset_y)
                
                # Copy the alpha channel to the appropriate position in the shadow mask
                for y in range(min_y, max_y):
                    for x in range(min_x, max_x):
                        src_x = x
                        src_y = y
                        dst_x = x + offset_x
                        dst_y = y + offset_y
                        
                        # Only copy if both source and destination are in bounds
                        if (0 <= src_x < width and 0 <= src_y < height and
                            0 <= dst_x < width and 0 <= dst_y < height):
                            shadow_mask[dst_y, dst_x] = alpha_np[src_y, src_x]
            
            # Apply distance-based fade if enabled
            if hasattr(config, 'distance_fade') and config.distance_fade:
                # Convert shadow mask to binary for distance transform
                from scipy.ndimage import distance_transform_edt
                
                # Find non-zero pixels in the alpha channel
                object_mask = (alpha_np > 0).astype(np.uint8)
                
                # Calculate distance from the object
                distance = distance_transform_edt(1 - object_mask)
                
                # Get fade distance
                fade_dist = config.fade_distance if hasattr(config, 'fade_distance') else 100
                
                # Apply fade based on distance
                fade_factor = np.clip(1 - distance / fade_dist, 0, 1)
                shadow_mask = shadow_mask * fade_factor
            
            # Convert to PIL Image
            shadow_mask_pil = Image.fromarray(shadow_mask.astype(np.uint8), 'L')
            
            # Adjust shadow size if expansion is non-zero
            if config.shadow_expansion != 0:
                if config.shadow_expansion > 0:
                    shadow_mask_pil = shadow_mask_pil.filter(ImageFilter.MaxFilter(config.shadow_expansion))
                else:
                    shadow_mask_pil = shadow_mask_pil.filter(ImageFilter.MinFilter(abs(config.shadow_expansion)))
            
            # Create shadow from alpha
            shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
            shadow_pixels = shadow.load()
            shadow_mask_pixels = shadow_mask_pil.load()
            
            height, width = image.size[1], image.size[0]
            for y in range(height):
                for x in range(width):
                    alpha_val = shadow_mask_pixels[x, y]
                    if alpha_val > 0:
                        shadow_opacity = int(alpha_val * config.shadow_opacity)
                        shadow_pixels[x, y] = (*config.shadow_color, shadow_opacity)
            
            # Apply shadow blur
            if config.shadow_blur > 0:
                # Apply soft edges if enabled
                if hasattr(config, 'soft_edges') and config.soft_edges:
                    # More refined blur with iteration for softer edges
                    temp_shadow = shadow
                    for _ in range(2):  # Multiple passes for softer result
                        temp_shadow = temp_shadow.filter(ImageFilter.GaussianBlur(config.shadow_blur / 2))
                    shadow = temp_shadow
                else:
                    # Standard blur
                    shadow = shadow.filter(ImageFilter.GaussianBlur(config.shadow_blur))
                
            return shadow
            
        except Exception as e:
            logger.error(f"Error creating shadow: {str(e)}")
            return None