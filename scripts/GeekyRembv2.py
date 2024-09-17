import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps
from rembg import remove, new_session
from enum import Enum
import math
from tqdm import tqdm
from scipy import ndimage

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
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GeekyRemB:
    def __init__(self):
        self.session = None
        self.use_gpu = torch.cuda.is_available()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "enable_background_removal": ("BOOLEAN", {"default": True}),
                "removal_method": (["rembg", "chroma_key"],),
                "model": (["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use"],),
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
        return remove(
            image,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold
        ).split()[-1]  # Return only the alpha channel as mask

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
        
        # Apply threshold
        mask_np = (mask_np > threshold * 255).astype(np.uint8) * 255
        
        # Edge detection
        if edge_detection:
            edges = cv2.Canny(mask_np, 100, 200)
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            mask_np = cv2.addWeighted(mask_np, 1, edges, 0.5, 0)
        
        # Expand or contract the mask
        if expansion != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(expansion), abs(expansion)))
            if expansion > 0:
                mask_np = cv2.dilate(mask_np, kernel)
            else:
                mask_np = cv2.erode(mask_np, kernel)
        
        # Apply blur
        if blur > 0:
            mask_np = cv2.GaussianBlur(mask_np, (blur * 2 + 1, blur * 2 + 1), 0)
        
        # Remove small regions
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
        
        # Invert if requested
        if invert:
            mask = ImageOps.invert(mask)
        
        return mask

    def calculate_default_position_and_scale(self, fg_size, bg_size):
        fg_aspect = fg_size[0] / fg_size[1]
        bg_aspect = bg_size[0] / bg_size[1]

        if fg_aspect > bg_aspect:
            # Fit to width
            scale = bg_size[0] / fg_size[0]
        else:
            # Fit to height
            scale = bg_size[1] / fg_size[1]

        # Ensure the scaled image isn't larger than the background
        scale = min(scale, 1.0)

        x = (bg_size[0] - fg_size[0] * scale) // 2
        y = (bg_size[1] - fg_size[1] * scale) // 2

        return x, y, scale

    def animate_element(self, element, animation_type, animation_speed, frame_number, total_frames,
                        x_start, y_start, canvas_width, canvas_height, scale, rotation):
        progress = frame_number / total_frames
        orig_width, orig_height = element.size
        
        # Ensure the element is in RGBA mode
        if element.mode != 'RGBA':
            element = element.convert('RGBA')
        
        # Apply initial scale
        new_size = (int(orig_width * scale), int(orig_height * scale))
        element = element.resize(new_size, Image.LANCZOS)
        
        # Apply rotation
        element = element.rotate(rotation, resample=Image.BICUBIC, expand=True)
        
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            x, y = x_start, y_start + y_offset
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(canvas_width - (canvas_width + element.width) * progress)
            y = y_start
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(-element.width + (canvas_width + element.width) * progress)
            y = y_start
        elif animation_type == AnimationType.ROTATE.value:
            angle = progress * 360 * animation_speed
            element = element.rotate(angle, resample=Image.BICUBIC, expand=True)
            x, y = x_start, y_start
        elif animation_type == AnimationType.FADE_IN.value:
            x, y = x_start, y_start
            opacity = int(progress * 255)
            r, g, b, a = element.split()
            a = a.point(lambda i: i * opacity // 255)
            element = Image.merge('RGBA', (r, g, b, a))
        elif animation_type == AnimationType.FADE_OUT.value:
            x, y = x_start, y_start
            opacity = int((1 - progress) * 255)
            r, g, b, a = element.split()
            a = a.point(lambda i: i * opacity // 255)
            element = Image.merge('RGBA', (r, g, b, a))
        elif animation_type == AnimationType.ZOOM_IN.value or animation_type == AnimationType.ZOOM_OUT.value:
            if animation_type == AnimationType.ZOOM_IN.value:
                zoom_scale = 1 + progress * animation_speed
            else:  # ZOOM_OUT
                zoom_scale = 1 + (1 - progress) * animation_speed
            
            new_width = int(orig_width * scale * zoom_scale)
            new_height = int(orig_height * scale * zoom_scale)
            
            element = element.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate the crop box to keep the center of the original image
            left = (new_width - orig_width * scale) / 2
            top = (new_height - orig_height * scale) / 2
            right = left + orig_width * scale
            bottom = top + orig_height * scale
            
            element = element.crop((left, top, right, bottom))
            x, y = x_start, y_start
        else:  # NONE
            x, y = x_start, y_start

        return element, x, y

    def process_image(self, foreground, enable_background_removal, removal_method, model, chroma_key_color,
                      chroma_key_tolerance, mask_expansion, edge_detection, edge_thickness, mask_blur, threshold,
                      invert_generated_mask, remove_small_regions, small_region_size, alpha_matting,
                      alpha_matting_foreground_threshold, alpha_matting_background_threshold,
                      animation_type, animation_speed, animation_frames, x_position, y_position, scale, rotation,
                      background=None, additional_mask=None, invert_additional_mask=False):
        try:
            if enable_background_removal and removal_method == "rembg":
                self.initialize_model(model)

            fg_frames = [tensor2pil(foreground[i]) for i in range(foreground.shape[0])]
            bg_frames = [tensor2pil(background[i]) for i in range(background.shape[0])] if background is not None else None

            if bg_frames and len(bg_frames) < animation_frames:
                bg_frames = bg_frames * (animation_frames // len(bg_frames) + 1)
                bg_frames = bg_frames[:animation_frames]

            # Calculate default position and scale
            if bg_frames:
                default_x, default_y, default_scale = self.calculate_default_position_and_scale(fg_frames[0].size, bg_frames[0].size)
            else:
                default_x, default_y, default_scale = 0, 0, 1.0

            # Use default values if user didn't specify custom ones
            x_position = x_position if x_position != 0 else default_x
            y_position = y_position if y_position != 0 else default_y
            scale = scale if scale != 1.0 else default_scale

            animated_frames = []
            masks = []
            for frame in tqdm(range(animation_frames), desc="Processing frames"):
                fg_index = frame % len(fg_frames)
                fg_frame = fg_frames[fg_index].convert('RGBA')
                
                if enable_background_removal:
                    if removal_method == "rembg":
                        mask = self.remove_background_rembg(fg_frame, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold)
                    else:  # chroma_key
                        mask = self.remove_background_chroma(fg_frame, chroma_key_color, chroma_key_tolerance)
                    
                    # Refine the mask
                    mask = self.refine_mask(mask, mask_expansion, edge_detection, edge_thickness, mask_blur,
                                            threshold, invert_generated_mask, remove_small_regions, small_region_size)
                    
                    # Combine with additional mask if provided
                    if additional_mask is not None:
                        additional_mask_pil = tensor2pil(additional_mask[frame % len(additional_mask)])
                        if invert_additional_mask:
                            additional_mask_pil = ImageOps.invert(additional_mask_pil)
                        mask = Image.fromarray(np.minimum(np.array(mask), np.array(additional_mask_pil)))
                    
                    # Apply the mask to remove the background
                    fg_frame.putalpha(mask)
                else:
                    mask = Image.new('L', fg_frame.size, 255)

                if bg_frames:
                    bg_index = frame % len(bg_frames)
                    result = bg_frames[bg_index].copy().convert('RGBA')
                else:
                    result = Image.new("RGBA", fg_frame.size, (0, 0, 0, 0))

                animated_fg, x, y = self.animate_element(
                    fg_frame, animation_type, animation_speed, frame, animation_frames,
                    x_position, y_position, result.width, result.height, scale, rotation
                )

                # Composite the animated foreground onto the background
                result.alpha_composite(animated_fg, (int(x), int(y)))
                animated_frames.append(pil2tensor(result.convert('RGB')))
                masks.append(pil2tensor(mask.convert('L')))

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
