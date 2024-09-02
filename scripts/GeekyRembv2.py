import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import torch
import logging
import cv2
from tqdm import tqdm
import onnxruntime as ort
from transformers import pipeline, AutoModelForImageSegmentation
from enum import Enum
import random
import math
import os
from torchvision import transforms

logging.basicConfig(level=logging.INFO)

class BlendingMode(Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    VIVID_LIGHT = "vivid_light"
    LINEAR_LIGHT = "linear_light"
    PIN_LIGHT = "pin_light"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    SUBTRACT = "subtract"

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

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)

class GeekyRemB:
    def __init__(self):
        self.session = None
        self.bria_pipeline = None
        self.birefnet = None
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        self.custom_model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "enable_background_removal": ("BOOLEAN", {"default": True}),
                "model": (["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "bria", "inspyrenet", "tracer", "basnet", "deeplab", "ormbg", "u2net_custom", "birefnet"],),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "post_process_mask": ("BOOLEAN", {"default": False}),
                "chroma_key": (["none", "green", "blue", "red"],),
                "chroma_threshold": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "color_tolerance": ("INT", {"default": 20, "min": 0, "max": 255, "step": 1}),
                "background_mode": (["transparent", "color", "image", "matrix"],),
                "background_color": ("COLOR", {"default": "#000000"}),
                "background_loop_mode": (["reverse", "loop"],),
                "blending_mode": ([mode.value for mode in BlendingMode],),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "aspect_ratio_preset": (["original", "1:1", "4:3", "16:9", "2:1", "custom"],),
                "custom_aspect_ratio": ("STRING", {"default": ""}),
                "foreground_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "x_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "edge_color": ("COLOR", {"default": "#FFFFFF"}),
                "shadow": ("BOOLEAN", {"default": False}),
                "shadow_blur": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "shadow_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "shadow_direction": ("FLOAT", {"default": 45, "min": 0, "max": 360, "step": 1}),
                "shadow_distance": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "color_adjustment": ("BOOLEAN", {"default": False}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "hue": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "filter": (["none", "blur", "sharpen", "edge_enhance", "emboss", "toon", "sepia", "film_grain", "matrix"],),
                "filter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "matrix_foreground_pattern": (["BINARY", "RANDOM", "CUSTOM"], {"default": "BINARY"}),
                "matrix_custom_text": ("STRING", {"default": "MATRIX"}),
                "matrix_density": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "matrix_fall_speed": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "matrix_glow": (["ENABLED", "DISABLED"], {"default": "DISABLED"}),
                "matrix_glow_intensity": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "animation_type": ([anim.value for anim in AnimationType],),
                "animation_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "animation_frames": ("INT", {"default": 30, "min": 1, "max": 300, "step": 1}),
                "animation_x_start": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "animation_y_start": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "animation_x_end": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "animation_y_end": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
            },
            "optional": {
                "background": ("IMAGE",),
                "input_masks": ("MASK",),
                "output_format": (["RGBA", "RGB"],),
                "only_mask": ("BOOLEAN", {"default": False}),
                "custom_model_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "image/processing"

    def ensure_image_format(self, image):
        img_np = np.array(image)
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        return img_np

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if color == "green":
            lower = np.array([40 - color_tolerance, 40, 40])
            upper = np.array([80 + color_tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([90 - color_tolerance, 40, 40])
            upper = np.array([130 + color_tolerance, 255, 255])
        elif color == "red":
            lower = np.array([0, 40, 40])
            upper = np.array([20 + color_tolerance, 255, 255])
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
        if invert_mask:
            mask = 255 - mask
        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)
        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)
        return mask

    def parse_aspect_ratio(self, aspect_ratio_preset, custom_aspect_ratio):
        if aspect_ratio_preset == "custom":
            if ":" in custom_aspect_ratio:
                try:
                    w, h = map(float, custom_aspect_ratio.split(":"))
                    return w / h
                except ValueError:
                    pass
            try:
                return float(custom_aspect_ratio)
            except ValueError:
                return None
        elif aspect_ratio_preset == "original":
            return None
        else:
            aspect_ratios = {
                "1:1": 1,
                "4:3": 4/3,
                "16:9": 16/9,
                "2:1": 2
            }
            return aspect_ratios.get(aspect_ratio_preset)

    def apply_color_adjustments(self, image, brightness, contrast, saturation, hue, sharpness):
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if saturation != 1.0:
            image = ImageEnhance.Color(image).enhance(saturation)
        if hue != 0.0:
            r, g, b = image.split()
            image = Image.merge("RGB", (
                r.point(lambda x: x + hue * 255),
                g.point(lambda x: x - hue * 255),
                b
            ))
        if sharpness != 1.0:
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
        return image

    def apply_filter(self, image, filter_type, strength):
        if filter_type == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=strength * 2))
        elif filter_type == "sharpen":
            percent = int(100 + (strength * 100))
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))
        elif filter_type == "edge_enhance":
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif filter_type == "emboss":
            return image.filter(ImageFilter.EMBOSS)
        elif filter_type == "toon":
            return self.apply_toon_filter(image, strength)
        elif filter_type == "sepia":
            return self.apply_sepia_filter(image)
        elif filter_type == "film_grain":
            return self.apply_film_grain(image, strength)
        elif filter_type == "matrix":
            return self.apply_matrix_filter(image, strength)
        else:
            return image

    def apply_toon_filter(self, image, strength):
        img_np = self.ensure_image_format(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_np, d=9, sigmaColor=300, sigmaSpace=300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        # Adjust the strength of the effect
        result = cv2.addWeighted(img_np, 1 - strength, cartoon, strength, 0)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def apply_sepia_filter(self, image):
        img_np = self.ensure_image_format(image)
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = cv2.transform(img_np, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)

    def apply_film_grain(self, image, strength):
        img_np = self.ensure_image_format(image)
        h, w, c = img_np.shape
        noise = np.random.randn(h, w) * 10 * strength
        noise = np.dstack([noise] * 3)
        grain_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(grain_img)

    def apply_matrix_filter(self, image, strength):
        img_np = self.ensure_image_format(image)
        h, w, _ = img_np.shape
        
        # Create a black background
        matrix = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate matrix rain effect
        for x in range(w):
            for y in range(h):
                if random.random() < strength * 0.1:
                    value = random.choice([0, 1])
                    color = (0, 255, 0) if value == 1 else (0, 100, 0)  # Bright green for 1, dark green for 0
                    matrix[y, x] = color
        
        # Add a motion blur effect
        kernel_size = int(strength * 5)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
        kernel /= kernel_size
        matrix = cv2.filter2D(matrix, -1, kernel)
        
        # Blend the matrix with the original image
        matrix_img = cv2.addWeighted(img_np, 1-strength, matrix, strength, 0)
        return Image.fromarray(matrix_img)

    def create_matrix_background(self, width, height, foreground_pattern, custom_text, density, fall_speed, glow, glow_intensity):
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        if foreground_pattern == "BINARY":
            chars = "01"
        elif foreground_pattern == "RANDOM":
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        else:  # CUSTOM
            chars = custom_text
        
        drops = [0 for _ in range(width // 10)]
        for _ in range(int(height * density / 10)):
            for i in range(len(drops)):
                if random.random() < 0.1:
                    x = i * 10
                    y = drops[i] * 15
                    char = random.choice(chars)
                    color = (0, 255, 0)
                    draw.text((x, y), char, font=font, fill=color)
                    
                    if glow == "ENABLED":
                        for offset in range(1, 4):
                            alpha = int(255 * (1 - offset / 4) * glow_intensity)
                            glow_color = (0, 255, 0, alpha)
                            draw.text((x-offset, y), char, font=font, fill=glow_color)
                            draw.text((x+offset, y), char, font=font, fill=glow_color)
                            draw.text((x, y-offset), char, font=font, fill=glow_color)
                            draw.text((x, y+offset), char, font=font, fill=glow_color)
                    
                    drops[i] += fall_speed
                    if drops[i] * 15 > height:
                        drops[i] = 0
        
        return image

    def apply_blending_mode(self, bg, fg, mode, strength):
        bg_np = np.array(bg).astype(np.float32) / 255.0
        fg_np = np.array(fg).astype(np.float32) / 255.0

        if bg_np.shape[-1] == 3:
            bg_np = np.dstack([bg_np, np.ones(bg_np.shape[:2], dtype=np.float32)])
        if fg_np.shape[-1] == 3:
            fg_np = np.dstack([fg_np, np.ones(fg_np.shape[:2], dtype=np.float32)])

        bg_rgb, bg_a = bg_np[..., :3], bg_np[..., 3]
        fg_rgb, fg_a = fg_np[..., :3], fg_np[..., 3]

        if mode == BlendingMode.NORMAL.value:
            blended_rgb = fg_rgb
        elif mode == BlendingMode.MULTIPLY.value:
            blended_rgb = bg_rgb * fg_rgb
        elif mode == BlendingMode.SCREEN.value:
            blended_rgb = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif mode == BlendingMode.OVERLAY.value:
            blended_rgb = np.where(bg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.DARKEN.value:
            blended_rgb = np.minimum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.LIGHTEN.value:
            blended_rgb = np.maximum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.COLOR_DODGE.value:
            blended_rgb = np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (1 - fg_rgb)))
        elif mode == BlendingMode.COLOR_BURN.value:
            blended_rgb = np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / fg_rgb))
        elif mode == BlendingMode.LINEAR_DODGE.value:
            blended_rgb = np.minimum(1, bg_rgb + fg_rgb)
        elif mode == BlendingMode.LINEAR_BURN.value:
            blended_rgb = np.maximum(0, bg_rgb + fg_rgb - 1)
        elif mode == BlendingMode.HARD_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.SOFT_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   bg_rgb - (1 - 2 * fg_rgb) * bg_rgb * (1 - bg_rgb),
                                   bg_rgb + (2 * fg_rgb - 1) * (np.sqrt(bg_rgb) - bg_rgb))
        elif mode == BlendingMode.VIVID_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / (2 * fg_rgb))),
                                   np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (2 * (1 - fg_rgb)))))
        elif mode == BlendingMode.LINEAR_LIGHT.value:
            blended_rgb = np.clip(bg_rgb + 2 * fg_rgb - 1, 0, 1)
        elif mode == BlendingMode.PIN_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, np.minimum(bg_rgb, 2 * fg_rgb), np.maximum(bg_rgb, 2 * fg_rgb - 1))
        elif mode == BlendingMode.DIFFERENCE.value:
            blended_rgb = np.abs(bg_rgb - fg_rgb)
        elif mode == BlendingMode.EXCLUSION.value:
            blended_rgb = bg_rgb + fg_rgb - 2 * bg_rgb * fg_rgb
        elif mode == BlendingMode.SUBTRACT.value:
            blended_rgb = np.maximum(0, bg_rgb - fg_rgb)
        else:
            blended_rgb = fg_rgb

        blended_rgb = blended_rgb * strength + bg_rgb * (1 - strength)
        blended_a = fg_a * strength + bg_a * (1 - strength)
        blended = np.dstack([blended_rgb, blended_a])

        return Image.fromarray((blended * 255.0).astype(np.uint8))

    def animate_element(self, element, element_mask, animation_type, animation_speed, frame_number, total_frames,
                        x_start, y_start, x_end, y_end, canvas_width, canvas_height):
        progress = frame_number / total_frames
        
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            x = x_start
            y = y_start + y_offset
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(x_start + (x_end - x_start) * (1 - progress))
            y = y_start
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(x_start + (x_end - x_start) * progress)
            y = y_start
        elif animation_type == AnimationType.ROTATE.value:
            angle = progress * 360 * animation_speed
            element = element.rotate(angle, resample=Image.BICUBIC, expand=True)
            if element_mask:
                element_mask = element_mask.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = x_start
            y = y_start
        elif animation_type == AnimationType.FADE_IN.value:
            x, y = x_start, y_start
            opacity = progress
            element = Image.blend(Image.new('RGBA', element.size, (0, 0, 0, 0)), element, opacity)
        elif animation_type == AnimationType.FADE_OUT.value:
            x, y = x_start, y_start
            opacity = 1 - progress
            element = Image.blend(Image.new('RGBA', element.size, (0, 0, 0, 0)), element, opacity)
        elif animation_type == AnimationType.ZOOM_IN.value:
            scale = 1 + progress * animation_speed
            new_size = (int(element.width * scale), int(element.height * scale))
            element = element.resize(new_size, Image.LANCZOS)
            if element_mask:
                element_mask = element_mask.resize(new_size, Image.LANCZOS)
            x = x_start - (new_size[0] - element.width) // 2
            y = y_start - (new_size[1] - element.height) // 2
        elif animation_type == AnimationType.ZOOM_OUT.value:
            scale = 1 + (1 - progress) * animation_speed
            new_size = (int(element.width * scale), int(element.height * scale))
            element = element.resize(new_size, Image.LANCZOS)
            if element_mask:
                element_mask = element_mask.resize(new_size, Image.LANCZOS)
            x = x_start - (new_size[0] - element.width) // 2
            y = y_start - (new_size[1] - element.height) // 2
        else:  # NONE
            x = x_start
            y = y_start

        return element, element_mask, x, y

    def process_image(self, foreground, enable_background_removal, model, alpha_matting, alpha_matting_foreground_threshold, 
                      alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                      color_tolerance, background_mode, background_color, background_loop_mode, blending_mode,
                      blend_strength, aspect_ratio_preset, custom_aspect_ratio, foreground_scale, x_position, y_position,
                      rotation, opacity, flip_horizontal, flip_vertical, invert_mask, feather_amount, edge_detection,
                      edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity, shadow_direction, shadow_distance,
                      color_adjustment, brightness, contrast, saturation, hue, sharpness,
                      filter, filter_strength, mask_blur, mask_expansion, matrix_foreground_pattern, matrix_custom_text,
                      matrix_density, matrix_fall_speed, matrix_glow, matrix_glow_intensity,
                      animation_type, animation_speed, animation_frames, animation_x_start, animation_y_start,
                      animation_x_end, animation_y_end,
                      background=None, input_masks=None, output_format="RGBA", only_mask=False, custom_model_path=""):
        
        if enable_background_removal:
            if model == "bria":
                if self.bria_pipeline is None:
                    self.bria_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
            elif model == "birefnet":
                if self.birefnet is None:
                    self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
                    self.birefnet.to(self.device)
                    self.birefnet.eval()
            elif self.session is None or self.session.model_name != model:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                if model == 'u2net_custom' and custom_model_path:
                    self.session = new_session('u2net_custom', model_path=custom_model_path, providers=providers)
                else:
                    self.session = new_session(model, providers=providers)

        bg_color = background_color
        edge_color = edge_color
        aspect_ratio = self.parse_aspect_ratio(aspect_ratio_preset, custom_aspect_ratio)

        def process_single_image(fg_image, bg_image=None, input_mask=None, frame_number=0):
            fg_pil = tensor2pil(fg_image)
            bg_pil = tensor2pil(bg_image) if bg_image is not None else None
            original_fg = np.array(fg_pil)
            
            if input_mask is not None:
                input_mask_np = input_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            else:
                input_mask_np = None

            if enable_background_removal:
                if chroma_key != "none":
                    chroma_mask = self.apply_chroma_key(original_fg, chroma_key, chroma_threshold, color_tolerance)
                    if input_mask_np is not None:
                        input_mask_np = cv2.bitwise_and(input_mask_np, chroma_mask)
                    else:
                        input_mask_np = chroma_mask

                if model == "bria":
                    removed_bg = self.bria_pipeline(fg_pil, return_mask=True)
                    rembg_mask = np.array(removed_bg)
                elif model == "birefnet":
                    transform_image = transforms.Compose([
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform_image(fg_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        pred = self.birefnet(input_tensor)[-1].sigmoid().cpu()
                    rembg_mask = (pred[0].squeeze().numpy() * 255).astype(np.uint8)
                else:
                    removed_bg = remove(
                        fg_pil,
                        session=self.session,
                        alpha_matting=alpha_matting,
                        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                        alpha_matting_background_threshold=alpha_matting_background_threshold,
                        post_process_mask=post_process_mask,
                    )
                    rembg_mask = np.array(removed_bg)[:,:,3]

                if input_mask_np is not None:
                    final_mask = cv2.bitwise_and(rembg_mask, input_mask_np)
                else:
                    final_mask = rembg_mask

                final_mask = self.process_mask(final_mask, invert_mask, feather_amount, mask_blur, mask_expansion)
            else:
                if input_mask_np is not None:
                    final_mask = input_mask_np
                else:
                    final_mask = np.full(fg_pil.size[::-1], 255, dtype=np.uint8)

            if only_mask:
                return pil2tensor(Image.fromarray(final_mask)), pil2tensor(Image.fromarray(final_mask))

            orig_width, orig_height = fg_pil.size
            if aspect_ratio is None:
                new_width, new_height = orig_width, orig_height
            else:
                if orig_width / orig_height > aspect_ratio:
                    new_width = int(orig_height * aspect_ratio)
                    new_height = orig_height
                else:
                    new_width = orig_width
                    new_height = int(orig_width / aspect_ratio)

            if background_mode == "transparent":
                result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            elif background_mode == "color":
                result = Image.new("RGBA", (new_width, new_height), bg_color)
            elif background_mode == "matrix":
                result = self.create_matrix_background(new_width, new_height, matrix_foreground_pattern, matrix_custom_text,
                                                       matrix_density, matrix_fall_speed, matrix_glow, matrix_glow_intensity)
            elif background_mode == "image" and bg_pil is not None:
                result = bg_pil.convert("RGBA").resize((new_width, new_height), Image.LANCZOS)
            else:
                result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

            fg_width = int(new_width * foreground_scale)
            fg_height = int(new_height * foreground_scale)

            fg_pil = fg_pil.resize((fg_width, fg_height), Image.LANCZOS)
            fg_mask = Image.fromarray(final_mask).resize((fg_width, fg_height), Image.LANCZOS)

            if flip_horizontal:
                fg_pil = fg_pil.transpose(Image.FLIP_LEFT_RIGHT)
                fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                fg_pil = fg_pil.transpose(Image.FLIP_TOP_BOTTOM)
                fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

            fg_pil = fg_pil.rotate(rotation, resample=Image.BICUBIC, expand=True)
            fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

            fg_pil, fg_mask, paste_x, paste_y = self.animate_element(
                fg_pil, fg_mask, animation_type, animation_speed, frame_number, animation_frames,
                animation_x_start, animation_y_start, animation_x_end, animation_y_end, new_width, new_height
            )

            if color_adjustment:
                fg_pil = self.apply_color_adjustments(fg_pil, brightness, contrast, saturation, hue, sharpness)
            
            if filter != "none":
                fg_pil = self.apply_filter(fg_pil, filter, filter_strength)

            bg_subset = result.crop((paste_x, paste_y, paste_x + fg_pil.width, paste_y + fg_pil.height))
            blended = self.apply_blending_mode(bg_subset, fg_pil, blending_mode, blend_strength)

            blended_rgba = blended.convert("RGBA")
            fg_with_opacity = Image.new("RGBA", blended_rgba.size, (0, 0, 0, 0))
            fg_data = blended_rgba.getdata()
            new_data = [(r, g, b, int(a * opacity)) for r, g, b, a in fg_data]
            fg_with_opacity.putdata(new_data)

            fg_mask_with_opacity = fg_mask.point(lambda p: int(p * opacity))

            if shadow:
                shadow_x = int(shadow_distance * np.cos(np.radians(shadow_direction)))
                shadow_y = int(shadow_distance * np.sin(np.radians(shadow_direction)))
                shadow_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
                shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
                shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (paste_x + shadow_x, paste_y + shadow_y), shadow_mask)
                result = Image.alpha_composite(result, shadow_image)

            result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask_with_opacity)

            if edge_detection:
                edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
                edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
                edge_overlay = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
                edge_overlay.paste(Image.new("RGB", fg_pil.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
                result = Image.alpha_composite(result, edge_overlay)

            if output_format == "RGB":
                result = result.convert("RGB")

            return pil2tensor(result), pil2tensor(fg_mask)

        try:
            fg_batch_size = foreground.shape[0]
            bg_batch_size = background.shape[0] if background is not None else 0
            max_batch_size = max(fg_batch_size, bg_batch_size)

            processed_images = []
            processed_masks = []
            logging.info(f"Processing {animation_frames} frames")
            
            for frame_number in tqdm(range(animation_frames), desc="Processing frames"):
                fg_index = frame_number % fg_batch_size
                bg_index = frame_number % bg_batch_size if bg_batch_size > 0 else None

                single_fg = foreground[fg_index:fg_index+1]
                single_bg = background[bg_index:bg_index+1] if bg_index is not None else None
                single_input_mask = input_masks[fg_index:fg_index+1] if input_masks is not None else None
                
                processed_image, processed_mask = process_single_image(single_fg, single_bg, single_input_mask, frame_number)
                
                processed_images.append(processed_image)
                processed_masks.append(processed_mask)

            logging.info("Finished processing all frames")
            
            stacked_images = torch.cat(processed_images, dim=0)
            stacked_masks = torch.cat(processed_masks, dim=0)
            
            if len(stacked_masks.shape) == 4 and stacked_masks.shape[1] == 1:
                stacked_masks = stacked_masks.squeeze(1)
            
            return (stacked_images, stacked_masks)
        
        except Exception as e:
            logging.error(f"Error during image processing: {str(e)}")
            raise

# Node class mapping
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemB
}

# Optional: Add a display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB"
}
