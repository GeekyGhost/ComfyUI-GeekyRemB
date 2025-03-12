import math
import numpy as np
from enum import Enum
from PIL import Image, ImageOps, ImageEnhance
from typing import Tuple, Callable, Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AnimationType(Enum):
    """Animation type options for GeekyRemB animations"""
    NONE = "none"
    BOUNCE = "bounce"
    TRAVEL_LEFT = "travel_left"
    TRAVEL_RIGHT = "travel_right"
    ROTATE = "rotate"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SCALE_BOUNCE = "scale_bounce"
    SPIRAL = "spiral"
    SHAKE = "shake"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    WAVE = "wave"
    PULSE = "pulse"
    SWING = "swing"
    SPIN = "spin"
    FLASH = "flash"
    KEYFRAME = "keyframe"  # Animation type for keyframe-based animation

# Easing Functions
def linear(t):
    """Linear easing function"""
    return t

def ease_in_quad(t):
    """Quadratic ease in"""
    return t * t

def ease_out_quad(t):
    """Quadratic ease out"""
    return t * (2 - t)

def ease_in_out_quad(t):
    """Quadratic ease in and out"""
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

def ease_in_cubic(t):
    """Cubic ease in"""
    return t ** 3

def ease_out_cubic(t):
    """Cubic ease out"""
    return (t - 1) ** 3 + 1

def ease_in_out_cubic(t):
    """Cubic ease in and out"""
    return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1

def ease_in_quart(t):
    """Quartic ease in"""
    return t ** 4

def ease_out_quart(t):
    """Quartic ease out"""
    return 1 - (t - 1) ** 4

def ease_in_out_quart(t):
    """Quartic ease in and out"""
    return 8 * t ** 4 if t < 0.5 else 1 - 8 * (t - 1) ** 4

def ease_in_sine(t):
    """Sinusoidal ease in"""
    return 1 - math.cos(t * math.pi / 2)

def ease_out_sine(t):
    """Sinusoidal ease out"""
    return math.sin(t * math.pi / 2)

def ease_in_out_sine(t):
    """Sinusoidal ease in and out"""
    return -(math.cos(math.pi * t) - 1) / 2

def ease_in_expo(t):
    """Exponential ease in"""
    return 0 if t == 0 else 2 ** (10 * (t - 1))

def ease_out_expo(t):
    """Exponential ease out"""
    return 1 if t == 1 else 1 - 2 ** (-10 * t)

def ease_in_out_expo(t):
    """Exponential ease in and out"""
    if t == 0 or t == 1:
        return t
    t *= 2
    if t < 1:
        return 0.5 * 2 ** (10 * (t - 1))
    t -= 1
    return 0.5 * (2 - 2 ** (-10 * t))

def ease_in_elastic(t):
    """Elastic ease in"""
    if t == 0 or t == 1:
        return t
    return -2 ** (10 * (t - 1)) * math.sin((t - 1.1) * 5 * math.pi)

def ease_out_elastic(t):
    """Elastic ease out"""
    if t == 0 or t == 1:
        return t
    return 1 + 2 ** (-10 * t) * math.sin((t - 0.1) * 5 * math.pi)

def bounce(t):
    """Bounce easing function"""
    if t < 4/11:
        return (121 * t * t) / 16
    elif t < 8/11:
        return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17/5.0
    elif t < 9/10:
        return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061/1805.0
    else:
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268/25.0

def ease_out_bounce(t):
    """Bounce ease out"""
    return bounce(t)

def ease_in_bounce(t):
    """Bounce ease in"""
    return 1 - bounce(1 - t)

def ease_in_out_bounce(t):
    """Bounce ease in and out"""
    if t < 0.5:
        return 0.5 * (1 - bounce(1 - 2 * t))
    else:
        return 0.5 * bounce(2 * t - 1) + 0.5

# Dictionary of all available easing functions
EASING_FUNCTIONS = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_quart": ease_in_quart,
    "ease_out_quart": ease_out_quart,
    "ease_in_out_quart": ease_in_out_quart,
    "ease_in_sine": ease_in_sine,
    "ease_out_sine": ease_out_sine,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_in_expo": ease_in_expo,
    "ease_out_expo": ease_out_expo,
    "ease_in_out_expo": ease_in_out_expo,
    "ease_in_elastic": ease_in_elastic,
    "ease_out_elastic": ease_out_elastic,
    "ease_in_bounce": ease_in_bounce,
    "ease_out_bounce": ease_out_bounce,
    "ease_in_out_bounce": ease_in_out_bounce,
}

class EnhancedAnimator:
    """Enhanced animation processing with additional effects and keyframe support"""
    
    @staticmethod
    def interpolate_keyframes(keyframes, current_frame, fps):
        """
        Interpolate position, scale, rotation, and opacity between keyframes
        
        Args:
            keyframes: List of keyframe dictionaries
            current_frame: Current frame number
            fps: Frames per second
            
        Returns:
            Dictionary with interpolated values for position, scale, rotation, opacity
        """
        # Log keyframe input parameters for debugging
        logger.info(f"Interpolating keyframes at frame {current_frame}")
        logger.info(f"Number of keyframes: {len(keyframes)}")
        
        if not keyframes or len(keyframes) == 0:
            # Default values if no keyframes provided
            logger.warning("No keyframes provided for interpolation")
            return {
                "position": (0, 0),
                "scale": 1.0,
                "rotation": 0.0,
                "opacity": 1.0
            }
        
        # If only one keyframe, return its values
        if len(keyframes) == 1:
            kf = keyframes[0]
            pos = kf.get("position", (0, 0))
            logger.info(f"Only one keyframe available. Using position: {pos}")
            return {
                "position": pos,
                "scale": kf.get("scale", 1.0),
                "rotation": kf.get("rotation", 0.0),
                "opacity": kf.get("opacity", 1.0)
            }
        
        # Sort keyframes by frame number
        sorted_keyframes = sorted(keyframes, key=lambda k: k.get("frame", 0))
        
        # Log sorted keyframes for debugging
        for i, kf in enumerate(sorted_keyframes):
            logger.info(f"Keyframe {i}: frame={kf.get('frame', 0)}, position={kf.get('position', (0, 0))}")
        
        # Find keyframes that bound the current frame
        for i in range(len(sorted_keyframes) - 1):
            k1 = sorted_keyframes[i]
            k2 = sorted_keyframes[i + 1]
            
            frame1 = k1.get("frame", 0)
            frame2 = k2.get("frame", 0)
            
            if frame1 <= current_frame <= frame2:
                # Calculate progress between keyframes
                if frame1 == frame2:  # Avoid division by zero
                    progress = 0
                else:
                    progress = (current_frame - frame1) / (frame2 - frame1)
                
                # Get easing function
                easing_name = k2.get("easing", "linear")
                easing_func = EASING_FUNCTIONS.get(easing_name, linear)
                
                # Apply easing
                eased_progress = easing_func(progress)
                
                # Interpolate position
                pos1 = k1.get("position", (0, 0))
                pos2 = k2.get("position", (0, 0))
                
                # Ensure positions are tuples with 2 elements
                if not isinstance(pos1, tuple) or len(pos1) != 2:
                    logger.warning(f"Invalid position in keyframe {i}: {pos1}, using (0,0)")
                    pos1 = (0, 0)
                if not isinstance(pos2, tuple) or len(pos2) != 2:
                    logger.warning(f"Invalid position in keyframe {i+1}: {pos2}, using (0,0)")
                    pos2 = (0, 0)
                
                pos_x = pos1[0] + (pos2[0] - pos1[0]) * eased_progress
                pos_y = pos1[1] + (pos2[1] - pos1[1]) * eased_progress
                
                # Interpolate scale
                scale1 = k1.get("scale", 1.0)
                scale2 = k2.get("scale", 1.0)
                scale = scale1 + (scale2 - scale1) * eased_progress
                
                # Interpolate rotation
                rot1 = k1.get("rotation", 0.0)
                rot2 = k2.get("rotation", 0.0)
                # Handle rotation wraparound for shortest path
                if abs(rot2 - rot1) > 180:
                    if rot1 > rot2:
                        rot2 += 360
                    else:
                        rot1 += 360
                rotation = rot1 + (rot2 - rot1) * eased_progress
                rotation %= 360  # Keep in 0-360 range
                
                # Interpolate opacity
                op1 = k1.get("opacity", 1.0)
                op2 = k2.get("opacity", 1.0)
                opacity = op1 + (op2 - op1) * eased_progress
                
                # Log interpolated values for debugging
                logger.info(f"Interpolating between keyframes {i} and {i+1}")
                logger.info(f"Frame1: {frame1}, Pos1: {pos1}, Frame2: {frame2}, Pos2: {pos2}")
                logger.info(f"Current frame: {current_frame}, Progress: {progress}, Eased progress: {eased_progress}")
                logger.info(f"Interpolated position: ({pos_x}, {pos_y})")
                
                return {
                    "position": (pos_x, pos_y),
                    "scale": scale,
                    "rotation": rotation,
                    "opacity": opacity
                }
        
        # If current_frame is before first keyframe
        if current_frame <= sorted_keyframes[0].get("frame", 0):
            kf = sorted_keyframes[0]
            pos = kf.get("position", (0, 0))
            logger.info(f"Frame {current_frame} is before first keyframe. Using position: {pos}")
            return {
                "position": pos,
                "scale": kf.get("scale", 1.0),
                "rotation": kf.get("rotation", 0.0),
                "opacity": kf.get("opacity", 1.0)
            }
        
        # If current_frame is after last keyframe
        kf = sorted_keyframes[-1]
        pos = kf.get("position", (0, 0))
        logger.info(f"Frame {current_frame} is after last keyframe. Using position: {pos}")
        return {
            "position": pos,
            "scale": kf.get("scale", 1.0),
            "rotation": kf.get("rotation", 0.0),
            "opacity": kf.get("opacity", 1.0)
        }
    
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
        phase_shift: float = 0.0,
        keyframe_params: Dict = None
    ) -> Tuple[Image.Image, int, int]:
        """
        Apply animation to an image frame.
        """
        try:
            # Check if we're doing keyframe animation - the string check is more robust than enum comparison
            is_keyframe_animation = animation_type == "keyframe" and keyframe_params is not None
            
            # If animation type is keyframe, use keyframe animation
            if is_keyframe_animation:
                # Log for debugging
                logger.info(f"Using keyframe animation for frame {frame_number}")
                
                # Validate keyframe parameters
                if not keyframe_params or "keyframes" not in keyframe_params:
                    logger.warning("Invalid keyframe_params: missing keyframes list")
                    return element, x_start, y_start
                
                # Get keyframes
                keyframes = keyframe_params.get("keyframes", [])
                
                if not keyframes:
                    logger.warning("Empty keyframes list")
                    return element, x_start, y_start
                
                # Log keyframes for debugging
                logger.info(f"Found {len(keyframes)} keyframes")
                
                # Get FPS
                fps = keyframe_params.get("fps", 30)
                
                # Interpolate values based on current frame
                interpolated = EnhancedAnimator.interpolate_keyframes(keyframes, frame_number, fps)
                
                # Get interpolated values
                x, y = interpolated["position"]
                interp_scale = interpolated["scale"] * scale  # Combine with base scale
                interp_rotation = interpolated["rotation"] + rotation  # Combine with base rotation
                opacity = interpolated["opacity"]
                
                # Log interpolated values for debugging
                logger.info(f"Interpolated values for frame {frame_number}: position=({x}, {y}), scale={interp_scale}, rotation={interp_rotation}")
                
                # Apply scale
                orig_width, orig_height = element.size
                new_size = (int(orig_width * interp_scale), int(orig_height * interp_scale))
                element = element.resize(new_size, Image.LANCZOS)
                
                # Apply rotation
                if interp_rotation != 0:
                    element = element.rotate(
                        interp_rotation,
                        resample=Image.BICUBIC,
                        expand=True,
                        center=(element.width // 2, element.height // 2)
                    )
                
                # Apply opacity if less than 1
                if opacity < 1.0:
                    r, g, b, a = element.split()
                    a = a.point(lambda i: i * opacity)
                    element = Image.merge('RGBA', (r, g, b, a))
                
                return element, int(x), int(y)
            
            # Standard animation processing for non-keyframe animations
            # Adjust frame_number based on delay
            adjusted_frame = frame_number - int(delay * total_frames)
            if adjusted_frame < 0:
                return element, x_start, y_start  # No animation yet
            
            # Handle repeats
            cycle_length = total_frames / max(1, repeat)
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
                center_x = bbox[0] + (bbox[2] - bbox[0]) // 2
                center_y = bbox[1] + (bbox[3] - bbox[1]) // 2
            else:
                center_x, center_y = element.width // 2, element.height // 2
            
            # Apply scaling
            new_size = (int(orig_width * scale), int(orig_height * scale))
            element = element.resize(new_size, Image.LANCZOS)
            
            # Apply rotation around the center of visible pixels
            if rotation != 0:
                element = element.rotate(
                    rotation,
                    resample=Image.BICUBIC,
                    expand=True,
                    center=(center_x, center_y)
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
            
            # Process different animation types
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
                zoom_scale = 0.5 + progress * animation_speed
                new_width = int(orig_width * scale * zoom_scale)
                new_height = int(orig_height * scale * zoom_scale)
                element = element.resize((new_width, new_height), Image.LANCZOS)
                x = x_start - (new_width - orig_width * scale) // 2
                y = y_start - (new_height - orig_height * scale) // 2
            
            elif animation_type == AnimationType.ZOOM_OUT.value:
                zoom_scale = 1.5 - progress * animation_speed
                zoom_scale = max(0.1, zoom_scale)  # Prevent negative or zero scaling
                new_width = int(orig_width * scale * zoom_scale)
                new_height = int(orig_height * scale * zoom_scale)
                element = element.resize((new_width, new_height), Image.LANCZOS)
                x = x_start - (new_width - orig_width * scale) // 2
                y = y_start - (new_height - orig_height * scale) // 2
            
            # New animation types
            elif animation_type == AnimationType.SHAKE.value:
                shake_amplitude = 10 * animation_speed
                x_offset = int(math.sin(progress * 10 * math.pi) * shake_amplitude)
                y_offset = int(math.cos(progress * 10 * math.pi) * shake_amplitude)
                x += x_offset
                y += y_offset
            
            elif animation_type == AnimationType.SLIDE_UP.value:
                start_y = canvas_height
                end_y = y_start
                y = int(start_y - (start_y - end_y) * progress)
            
            elif animation_type == AnimationType.SLIDE_DOWN.value:
                start_y = -element.height
                end_y = y_start
                y = int(start_y + (end_y - start_y) * progress)
            
            elif animation_type == AnimationType.FLIP_HORIZONTAL.value:
                # Create separate copies for before and after the flip
                if progress < 0.5:
                    # Scale horizontally as we approach the middle
                    scale_x = 1 - progress * 2  # 1 -> 0
                    new_width = max(1, int(element.width * scale_x))
                    element = element.resize((new_width, element.height), Image.LANCZOS)
                    x = x_start + (orig_width * scale - new_width) // 2
                else:
                    # Flip the image at the midpoint and scale back up
                    element = ImageOps.mirror(element)
                    scale_x = (progress - 0.5) * 2  # 0 -> 1
                    new_width = max(1, int(element.width * scale_x))
                    element = element.resize((new_width, element.height), Image.LANCZOS)
                    x = x_start + (orig_width * scale - new_width) // 2
            
            elif animation_type == AnimationType.FLIP_VERTICAL.value:
                # Similar approach for vertical flip
                if progress < 0.5:
                    scale_y = 1 - progress * 2  # 1 -> 0
                    new_height = max(1, int(element.height * scale_y))
                    element = element.resize((element.width, new_height), Image.LANCZOS)
                    y = y_start + (orig_height * scale - new_height) // 2
                else:
                    element = ImageOps.flip(element)
                    scale_y = (progress - 0.5) * 2  # 0 -> 1
                    new_height = max(1, int(element.height * scale_y))
                    element = element.resize((element.width, new_height), Image.LANCZOS)
                    y = y_start + (orig_height * scale - new_height) // 2
            
            elif animation_type == AnimationType.WAVE.value:
                # Create a wave effect by offsetting rows of pixels
                img_array = np.array(element)
                height, width = img_array.shape[:2]
                
                # Create a new array for the waved image
                waved = np.zeros_like(img_array)
                
                # Wave amplitude and frequency
                amplitude = 10 * animation_speed
                frequency = 2 + 2 * animation_speed
                
                # Apply horizontal wave
                for i in range(height):
                    offset = int(amplitude * math.sin(2 * math.pi * frequency * i / height + progress * 2 * math.pi))
                    for j in range(width):
                        src_j = (j - offset) % width
                        waved[i, j] = img_array[i, src_j]
                
                element = Image.fromarray(waved)
            
            elif animation_type == AnimationType.PULSE.value:
                pulse_scale = 1 + 0.3 * math.sin(progress * 4 * math.pi) * animation_speed
                new_size = (int(element.width * pulse_scale), int(element.height * pulse_scale))
                element = element.resize(new_size, Image.LANCZOS)
                x -= (new_size[0] - element.width) // 2
                y -= (new_size[1] - element.height) // 2
            
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
                # Flash between normal and bright
                if progress < 0.5:
                    flash_strength = 1 + 3 * progress * animation_speed
                else:
                    flash_strength = 1 + 3 * (1 - progress) * animation_speed
                
                enhancer = ImageEnhance.Brightness(element)
                element = enhancer.enhance(flash_strength)
            
            # Log final position for debugging
            if frame_number % 10 == 0:  # Only log every 10th frame to reduce spam
                logger.info(f"Standard animation frame {frame_number}: final position=({x}, {y})")
                
            return element, x, y
            
        except Exception as e:
            logger.error(f"Animation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return element, x_start, y_start