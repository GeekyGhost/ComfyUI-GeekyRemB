import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from rembg import remove, new_session
from enum import Enum
import math
from tqdm import tqdm
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread

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

class BlendMode:
    @staticmethod
    def _ensure_rgba(img):
        """Convert input array to RGBA format if needed"""
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # RGB
                alpha = np.ones((*img.shape[:2], 1)) * 255
                return np.concatenate([img, alpha], axis=-1)
            return img
        else:  # Single channel
            return np.stack([img] * 4, axis=-1)

    @staticmethod
    def _apply_blend(target, blend, operation, opacity=1.0):
        """Apply blend operation with proper alpha handling"""
        # Ensure both images are RGBA
        target = BlendMode._ensure_rgba(target).astype(np.float32)
        blend = BlendMode._ensure_rgba(blend).astype(np.float32)
        
        # Normalize to 0-1 range
        target = target / 255.0
        blend = blend / 255.0
        
        # Split channels
        target_rgb = target[..., :3]
        blend_rgb = blend[..., :3]
        target_a = target[..., 3:4]
        blend_a = blend[..., 3:4]
        
        # Apply blend operation
        result_rgb = operation(target_rgb, blend_rgb)
        
        # Calculate final alpha
        result_a = target_a + blend_a * (1 - target_a) * opacity
        
        # Combine RGB and alpha
        result = np.concatenate([
            result_rgb * opacity + target_rgb * (1 - opacity),
            result_a
        ], axis=-1)
        
        # Convert back to 0-255 range
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def normal(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: b, opacity)
    
    @staticmethod
    def multiply(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t * b, opacity)
    
    @staticmethod
    def screen(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: 1 - (1 - t) * (1 - b), opacity)
    
    @staticmethod
    def overlay(target, blend, opacity=1.0):
        def overlay_op(t, b):
            return np.where(t > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, overlay_op, opacity)
    
    @staticmethod
    def soft_light(target, blend, opacity=1.0):
        def soft_light_op(t, b):
            return np.where(b > 0.5,
                          t + (2 * b - 1) * (t - t * t),
                          t - (1 - 2 * b) * t * (1 - t))
        return BlendMode._apply_blend(target, blend, soft_light_op, opacity)
    
    @staticmethod
    def hard_light(target, blend, opacity=1.0):
        def hard_light_op(t, b):
            return np.where(b > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, hard_light_op, opacity)
    
    @staticmethod
    def difference(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: np.abs(t - b), opacity)
    
    @staticmethod
    def exclusion(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t + b - 2 * t * b, opacity)
    
    @staticmethod
    def color_dodge(target, blend, opacity=1.0):
        def color_dodge_op(t, b):
            return np.where(b >= 1, 1, np.minimum(1, t / (1 - b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_dodge_op, opacity)
    
    @staticmethod
    def color_burn(target, blend, opacity=1.0):
        def color_burn_op(t, b):
            return np.where(b <= 0, 0, np.maximum(0, 1 - (1 - t) / (b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_burn_op, opacity)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GeekyRemB:
    def __init__(self):
        self.session = None
        self.use_gpu = torch.cuda.is_available()
        self.frame_cache = {}
        self.max_cache_size = 100
        self.batch_size = 4
        self.max_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Instance variables for background removal
        self.enable_background_removal = True
        self.removal_method = "rembg"
        self.chroma_key_color = "green"
        self.chroma_key_tolerance = 0.1
        self.mask_expansion = 0
        self.edge_detection = False
        self.edge_thickness = 1
        self.mask_blur = 0
        self.threshold = 0.5
        self.invert_generated_mask = False
        self.remove_small_regions = False
        self.small_region_size = 100
        
        self.blend_modes = {
            "normal": BlendMode.normal,
            "multiply": BlendMode.multiply,
            "screen": BlendMode.screen,
            "overlay": BlendMode.overlay,
            "soft_light": BlendMode.soft_light,
            "hard_light": BlendMode.hard_light,
            "difference": BlendMode.difference,
            "exclusion": BlendMode.exclusion,
            "color_dodge": BlendMode.color_dodge,
            "color_burn": BlendMode.color_burn
        }

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
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "soft_light", 
                    "hard_light", "difference", "exclusion", "color_dodge", "color_burn"
                ],),
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

    def initialize_model(self, model):
        if self.session is None or self.session.model_name != model:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.session = new_session(model, providers=providers)

    def remove_background_rembg(self, image, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold):
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get the full RGBA result from rembg
        result = remove(
            image,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold
        )
        
        # Return both the RGBA result and its alpha channel as mask
        return result, result.split()[3]

    def remove_background_chroma(self, image, color, tolerance):
        img_np = np.array(image)
        if img_np.shape[2] == 4:
            img_np = img_np[:,:,:3]
        
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        if color == "green":
            lower = np.array([60 - 30*tolerance, 100, 100])
            upper = np.array([60 + 30*tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([120 - 30*tolerance, 100, 100])
            upper = np.array([120 + 30*tolerance, 255, 255])
        else:  # red
            lower = np.array([0, 100, 100])
            upper = np.array([30*tolerance, 255, 255])
            
        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - mask
        
        return Image.fromarray(mask)

    def refine_mask(self, mask, expansion, edge_detection, edge_thickness, blur, threshold, invert, remove_small_regions, small_region_size):
        mask_np = np.array(mask)
        
        mask_np = (mask_np > threshold * 255).astype(np.uint8) * 255
        
        if edge_detection:
            edges = cv2.Canny(mask_np, 100, 200)
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            mask_np = cv2.addWeighted(mask_np, 1, edges, 0.5, 0)
        
        if expansion != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(expansion), abs(expansion)))
            if expansion > 0:
                mask_np = cv2.dilate(mask_np, kernel)
            else:
                mask_np = cv2.erode(mask_np, kernel)
        
        if blur > 0:
            mask_np = cv2.GaussianBlur(mask_np, (blur * 2 + 1, blur * 2 + 1), 0)
        
        if remove_small_regions:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            img2 = np.zeros((output.shape))
            for i in range(0, nb_components):
                if sizes[i] >= small_region_size:
                    img2[output == i + 1] = 255
            mask_np = img2.astype(np.uint8)
        
        mask = Image.fromarray(mask_np)
        
        if invert:
            mask = ImageOps.invert(mask)
        
        return mask

    def parse_aspect_ratio(self, aspect_ratio_input):
        if not aspect_ratio_input:
            return None
        
        if ':' in aspect_ratio_input:
            try:
                w, h = map(float, aspect_ratio_input.split(':'))
                return w / h
            except ValueError:
                return None
        
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

    def calculate_default_position_and_scale(self, fg_size, bg_size):
        fg_aspect = fg_size[0] / fg_size[1]
        bg_aspect = bg_size[0] / bg_size[1]

        if fg_aspect > bg_aspect:
            scale = bg_size[0] / fg_size[0]
        else:
            scale = bg_size[1] / fg_size[1]

        scale = min(scale, 1.0)

        x = (bg_size[0] - fg_size[0] * scale) // 2
        y = (bg_size[1] - fg_size[1] * scale) // 2

        return x, y, scale

    def animate_element(self, element, animation_type, animation_speed, frame_number, total_frames,
                       x_start, y_start, canvas_width, canvas_height, scale, rotation):
        progress = frame_number / total_frames
        orig_width, orig_height = element.size
        
        if element.mode != 'RGBA':
            element = element.convert('RGBA')
        
        new_size = (int(orig_width * scale), int(orig_height * scale))
        element = element.resize(new_size, Image.LANCZOS)
        
        rotated = Image.new('RGBA', element.size, (0, 0, 0, 0))
        rotated.paste(element, (0, 0), element)
        
        if rotation != 0:
            rotated = rotated.rotate(rotation, resample=Image.BICUBIC, expand=True,
                                   center=(rotated.width // 2, rotated.height // 2))
        
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            x, y = x_start, y_start + y_offset
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(canvas_width - (canvas_width + rotated.width) * progress)
            y = y_start
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(-rotated.width + (canvas_width + rotated.width) * progress)
            y = y_start
        elif animation_type == AnimationType.ROTATE.value:
            angle = progress * 360 * animation_speed
            rotated = rotated.rotate(angle, resample=Image.BICUBIC, expand=True,
                                   center=(rotated.width // 2, rotated.height // 2))
            x, y = x_start, y_start
        elif animation_type == AnimationType.FADE_IN.value:
            x, y = x_start, y_start
            opacity = int(progress * 255)
            r, g, b, a = rotated.split()
            a = a.point(lambda i: i * opacity // 255)
            rotated = Image.merge('RGBA', (r, g, b, a))
        elif animation_type == AnimationType.FADE_OUT.value:
            x, y = x_start, y_start
            opacity = int((1 - progress) * 255)
            r, g, b, a = rotated.split()
            a = a.point(lambda i: i * opacity // 255)
            rotated = Image.merge('RGBA', (r, g, b, a))
        elif animation_type in [AnimationType.ZOOM_IN.value, AnimationType.ZOOM_OUT.value]:
            zoom_scale = 1 + progress * animation_speed if animation_type == AnimationType.ZOOM_IN.value else 1 + (1 - progress) * animation_speed
            
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            
            rotated = rotated.resize((new_width, new_height), Image.LANCZOS)
            
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            rotated = rotated.crop((left, top, right, bottom))
            x, y = x_start, y_start
        else:  # NONE
            x, y = x_start, y_start

        return rotated, x, y

    def process_frame(self, frame, background_frame=None, *args):
        try:
            # Convert frame format if needed
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            # Handle background removal
            if self.enable_background_removal:
                if self.removal_method == "rembg":
                    frame_with_alpha, mask = self.remove_background_rembg(
                        frame,
                        args[0],  # alpha_matting
                        args[1],  # alpha_matting_foreground_threshold
                        args[2]   # alpha_matting_background_threshold
                    )
                    frame = frame_with_alpha
                else:
                    mask = self.remove_background_chroma(
                        frame,
                        self.chroma_key_color,
                        self.chroma_key_tolerance
                    )
                    frame = Image.composite(
                        frame, 
                        Image.new('RGBA', frame.size, (0, 0, 0, 0)), 
                        mask
                    )

                mask = self.refine_mask(
                    mask,
                    self.mask_expansion,
                    self.edge_detection,
                    self.edge_thickness,
                    self.mask_blur,
                    self.threshold,
                    self.invert_generated_mask,
                    self.remove_small_regions,
                    self.small_region_size
                )
            else:
                mask = Image.new('L', frame.size, 255)

            # Apply animation
            animated_frame, x, y = self.animate_element(
                frame,
                args[3],   # animation_type
                args[4],   # animation_speed
                args[5],   # frame_number
                args[6],   # total_frames
                args[7],   # x_position
                args[8],   # y_position
                background_frame.width if background_frame else frame.width,
                background_frame.height if background_frame else frame.height,
                args[9],   # scale
                args[10]   # rotation
            )

            # Handle background blending
            if background_frame is not None:
                bg_width, bg_height = background_frame.size
                
                if args[11] != "normal":  # blend_mode
                    # Create a blank canvas matching background size
                    canvas = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
                    
                    # Calculate safe paste position
                    paste_x = max(0, min(int(x), bg_width - animated_frame.width))
                    paste_y = max(0, min(int(y), bg_height - animated_frame.height))
                    
                    # Paste animated frame onto canvas
                    canvas.paste(animated_frame, (paste_x, paste_y), animated_frame)
                    
                    # Convert to numpy arrays for blending
                    bg_array = np.array(background_frame)
                    canvas_array = np.array(canvas)
                    
                    # Apply blend mode with opacity
                    result_array = self.blend_modes[args[11]](
                        bg_array, 
                        canvas_array, 
                        args[12]  # opacity
                    )
                    result = Image.fromarray(result_array)
                else:
                    # Normal blend mode
                    result = background_frame.copy().convert('RGBA')
                    result.alpha_composite(animated_frame, (int(x), int(y)))
            else:
                result = animated_frame

            return result, mask

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
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
            # Set instance variables
            self.enable_background_removal = enable_background_removal
            self.removal_method = removal_method
            self.chroma_key_color = chroma_key_color
            self.chroma_key_tolerance = chroma_key_tolerance
            self.mask_expansion = mask_expansion
            self.edge_detection = edge_detection
            self.edge_thickness = edge_thickness
            self.mask_blur = mask_blur
            self.threshold = threshold
            self.invert_generated_mask = invert_generated_mask
            self.remove_small_regions = remove_small_regions
            self.small_region_size = small_region_size

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

            for frame in tqdm(range(animation_frames), desc="Processing frames"):
                fg_index = frame % len(fg_frames)
                bg_frame = bg_frames[frame % len(bg_frames)] if bg_frames else None

                frame_args = (
                    alpha_matting,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    animation_type,
                    animation_speed,
                    frame,
                    animation_frames,
                    x_position,
                    y_position,
                    scale,
                    rotation,
                    blend_mode,
                    opacity
                )

                result_frame, mask = self.process_frame(fg_frames[fg_index], bg_frame, *frame_args)

                if additional_mask is not None:
                    additional_mask_pil = tensor2pil(additional_mask[frame % len(additional_mask)])
                    if invert_additional_mask:
                        additional_mask_pil = ImageOps.invert(additional_mask_pil)
                    mask = Image.fromarray(np.minimum(np.array(mask), np.array(additional_mask_pil)))

                animated_frames.append(pil2tensor(result_frame))
                masks.append(pil2tensor(mask.convert('L')))

            # Convert output format if needed
            if output_format == "RGB":
                for i in range(len(animated_frames)):
                    frame = tensor2pil(animated_frames[i])
                    frame = frame.convert('RGB')
                    animated_frames[i] = pil2tensor(frame)

            return (torch.cat(animated_frames, dim=0), torch.cat(masks, dim=0))

        except Exception as e:
            print(f"Error in GeekyRemB: {str(e)}")
            return (foreground, torch.zeros_like(foreground[:, :1, :, :]))

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemB
}

# Display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB"
}
