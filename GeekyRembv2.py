import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter
import torch
import logging
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    # Convert PIL image to numpy array
    np_image = np.array(image).astype(np.float32) / 255.0
    # Add batch dimension if it's a single image
    if np_image.ndim == 3:
        np_image = np_image[None, ...]
    # Convert to torch tensor
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
            },
            "optional": {
                "output_format": (["RGBA", "RGB"],),
                "input_masks": ("MASK",),
                "background_images": ("IMAGE",),
                "background_color": ("COLOR", {"default": "#000000"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
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
                          color_tolerance, background_mode, output_format="RGBA", input_masks=None, 
                          background_images=None, background_color="#000000", invert_mask=False, feather_amount=0):
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)

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
                result = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            elif background_mode == "color":
                result = Image.new("RGBA", pil_image.size, bg_color)
            else:  # background_mode == "image"
                if background_image is not None:
                    bg_pil = tensor2pil(background_image)
                    result = bg_pil.resize(pil_image.size, Image.LANCZOS).convert("RGBA")
                else:
                    result = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))

            # Use the final_mask to composite the original image onto the result
            result.paste(pil_image, (0, 0), Image.fromarray(final_mask))

            if output_format == "RGB":
                result = result.convert("RGB")

            return pil2tensor(result), pil2tensor(Image.fromarray(final_mask))

        try:
            batch_size = images.shape[0]
            processed_images = []
            processed_masks = []
            logging.info(f"Processing {batch_size} images")
            
            for i in tqdm(range(batch_size), desc="Removing backgrounds"):
                single_image = images[i]
                input_mask = input_masks[i] if input_masks is not None else None
                background_image = background_images[i] if background_images is not None else None
                processed_image, processed_mask = process_single_image(single_image, input_mask, background_image)
                processed_images.append(processed_image)
                processed_masks.append(processed_mask)

            logging.info("Finished processing all images")
            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0))
        
        except Exception as e:
            logging.error(f"Error during background removal: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemB
}
