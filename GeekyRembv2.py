import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch
import logging
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:  # If it's a grayscale image (mask)
        np_image = np_image[None, None, ...]  # Add batch and channel dimensions
    elif np_image.ndim == 3:  # If it's an RGB image
        np_image = np_image[None, ...]  # Add batch dimension
    return torch.from_numpy(np_image)

class GeekyRemB:
    def __init__(self):
        self.session = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"],),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "post_process_mask": ("BOOLEAN", {"default": False}),
                "chroma_key": (["none", "green", "blue", "red"],),
                "chroma_threshold": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "color_tolerance": ("INT", {"default": 20, "min": 0, "max": 255, "step": 1}),
                "background_mode": (["transparent", "color", "image"],),
                "background_loop_mode": (["reverse", "loop"],),
                "background_width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "background_height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "output_format": (["RGBA", "RGB"],),
                "input_masks": ("MASK",),
                "background_images": ("IMAGE",),
                "background_color": ("COLOR", {"default": "#000000"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "edge_color": ("COLOR", {"default": "#FFFFFF"}),
                "shadow": ("BOOLEAN", {"default": False}),
                "shadow_blur": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "shadow_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "color_adjustment": ("BOOLEAN", {"default": False}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "x_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image/processing"

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

    def remove_background(self, images, model, alpha_matting, alpha_matting_foreground_threshold, 
                          alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                          color_tolerance, background_mode, background_loop_mode="loop", background_width=512,
                          background_height=512, output_format="RGBA", input_masks=None, background_images=None, 
                          background_color="#000000", invert_mask=False, feather_amount=0, edge_detection=False, 
                          edge_thickness=1, edge_color="#FFFFFF", shadow=False, shadow_blur=5, shadow_opacity=0.5, 
                          color_adjustment=False, brightness=1.0, contrast=1.0, saturation=1.0, scale=1.0, 
                          x_position=0, y_position=0, rotation=0, opacity=1.0, flip_horizontal=False, 
                          flip_vertical=False, aspect_ratio=1.0):
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        edge_color = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        def get_background_image(index, total_images):
            if background_images is None or len(background_images) == 0:
                return None
            
            bg_count = len(background_images)
            
            if background_loop_mode == "loop":
                bg_image = background_images[index % bg_count]
            elif background_loop_mode == "reverse":
                forward = index % (2 * bg_count)
                if forward < bg_count:
                    bg_image = background_images[forward]
                else:
                    bg_image = background_images[2 * bg_count - forward - 1]
            
            bg_pil = tensor2pil(bg_image)
            return bg_pil.resize((background_width, background_height), Image.LANCZOS)

        def process_single_image(image, input_mask=None, background_image=None):
            pil_image = tensor2pil(image)
            original_image = np.array(pil_image)
            
            if input_mask is not None:
                mask_pil = Image.fromarray(input_mask.squeeze().cpu().numpy().astype(np.uint8) * 255)
                mask_pil = mask_pil.resize(pil_image.size, Image.LANCZOS)
                if invert_mask:
                    mask_pil = ImageOps.invert(mask_pil)
                if feather_amount > 0:
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather_amount))
                input_mask_np = np.array(mask_pil)
            else:
                input_mask_np = None

            if chroma_key != "none":
                chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
                if input_mask_np is not None:
                    input_mask_np = cv2.bitwise_and(input_mask_np, chroma_mask)
                else:
                    input_mask_np = chroma_mask

            removed_bg = remove(
                pil_image,
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

            if background_mode == "transparent":
                result = Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0))
            elif background_mode == "color":
                result = Image.new("RGBA", (background_width, background_height), bg_color)
            else:  # background_mode == "image"
                if background_image is not None:
                    result = background_image.convert("RGBA")
                else:
                    result = Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0))

            # Scale, rotate, and position the foreground image
            fg_image = Image.fromarray(original_image)
            fg_mask = Image.fromarray(final_mask)
            
            if flip_horizontal:
                fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
                fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)
                fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Apply aspect ratio adjustment
            orig_width, orig_height = fg_image.size
            new_width = int(orig_width * scale)
            new_height = int(new_width / aspect_ratio)
            
            fg_image = fg_image.resize((new_width, new_height), Image.LANCZOS)
            fg_mask = fg_mask.resize((new_width, new_height), Image.LANCZOS)

            fg_image = fg_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
            fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

            paste_x = x_position + (background_width - fg_image.width) // 2
            paste_y = y_position + (background_height - fg_image.height) // 2

            scaled_fg = Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0))
            scaled_fg.paste(fg_image, (paste_x, paste_y), fg_mask)

            scaled_fg = Image.blend(Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0)), scaled_fg, opacity)

            result = Image.alpha_composite(result, scaled_fg)

            if edge_detection:
                edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
                edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
                edge_overlay = Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0))
                edge_overlay.paste(Image.new("RGB", fg_image.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
                result = Image.alpha_composite(result, edge_overlay)

            if shadow:
                shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
                shadow_image = Image.new("RGBA", (background_width, background_height), (0, 0, 0, 0))
                shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (paste_x, paste_y), shadow_mask)
                result = Image.alpha_composite(result, shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur)))

            if color_adjustment:
                enhancer = ImageEnhance.Brightness(result)
                result = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Color(result)
                result = enhancer.enhance(saturation)

            if output_format == "RGB":
                result = result.convert("RGB")

            return pil2tensor(result), pil2tensor(fg_mask)

        try:
            batch_size = images.shape[0]
            processed_images = []
            processed_masks = []
            logging.info(f"Processing {batch_size} images")
            
            for i in tqdm(range(batch_size), desc="Removing backgrounds"):
                single_image = images[i:i+1]
                single_input_mask = input_masks[i:i+1] if input_masks is not None else None
                single_background_image = get_background_image(i, batch_size)
                
                processed_image, processed_mask = process_single_image(single_image, single_input_mask, single_background_image)
                
                processed_images.append(processed_image)
                processed_masks.append(processed_mask)

            logging.info("Finished processing all images")
            
            stacked_images = torch.cat(processed_images, dim=0)
            stacked_masks = torch.cat(processed_masks, dim=0)
            
            if len(stacked_masks.shape) == 4 and stacked_masks.shape[1] == 1:
                stacked_masks = stacked_masks.squeeze(1)
            
            return (stacked_images, stacked_masks)
        
        except Exception as e:
            logging.error(f"Error during background removal: {str(e)}")
            raise

# Node class mapping
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemB
}

# Optional: Add a display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB"
}
