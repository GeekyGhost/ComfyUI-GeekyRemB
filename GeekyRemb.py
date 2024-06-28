import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter
import torch
import logging
import concurrent.futures
import cv2

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GeekyRemb:
    def __init__(self):
        self.cache = {}
        self.session = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime", "briarmbg", "custom"],),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 40, "step": 1}),
                "post_process_mask": ("BOOLEAN", {"default": True}),
                "only_mask": ("BOOLEAN", {"default": False}),
                "background_color": (["none", "black", "white", "magenta", "chroma green", "chroma blue"],),
                "edge_detection": (["none", "canny", "sobel"],),
                "edge_detection_threshold1": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "edge_detection_threshold2": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "blur_mask": ("BOOLEAN", {"default": False}),
                "blur_kernel_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2}),
                "morphological_operation": (["none", "erosion", "dilation", "opening", "closing"],),
                "morph_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "morph_iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "adaptive_threshold": ("BOOLEAN", {"default": False}),
                "watershed": ("BOOLEAN", {"default": False}),
                "grabcut": ("BOOLEAN", {"default": False}),
                "color_segmentation": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "chroma_key": (["none", "green", "blue", "red"],),
                "chroma_threshold": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "color_tolerance": ("INT", {"default": 20, "min": 0, "max": 255, "step": 1}),
                "despill": ("BOOLEAN", {"default": False}),
                "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image/processing"

    def post_process_mask(self, mask, edge_detection, edge_detection_threshold1, edge_detection_threshold2, 
                          blur_mask, blur_kernel_size, morphological_operation, morph_kernel_size, morph_iterations,
                          adaptive_threshold, watershed, grabcut, color_segmentation, original_image):
        if edge_detection == "canny":
            edges = cv2.Canny(mask, edge_detection_threshold1, edge_detection_threshold2)
            mask = cv2.bitwise_or(mask, edges)
        elif edge_detection == "sobel":
            sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
            mask = cv2.magnitude(sobelx, sobely)
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if blur_mask:
            mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        if morphological_operation == "erosion":
            mask = cv2.erode(mask, kernel, iterations=morph_iterations)
        elif morphological_operation == "dilation":
            mask = cv2.dilate(mask, kernel, iterations=morph_iterations)
        elif morphological_operation == "opening":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        elif morphological_operation == "closing":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        if adaptive_threshold:
            mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if watershed:
            sure_bg = cv2.dilate(mask, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), markers)
            mask = np.zeros(mask.shape, dtype=np.uint8)
            mask[markers == -1] = 255

        if grabcut:
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            rect = (1,1,mask.shape[1]-1,mask.shape[0]-1)
            cv2.grabCut(original_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            mask = mask2*255

        if color_segmentation:
            Z = original_image.reshape((-1,3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            mask = res.reshape((original_image.shape))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        return mask

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

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def despill(self, image, mask, color, strength):
        if color == "green":
            other_channels = [0, 2]  # Red and Blue channels
        elif color == "blue":
            other_channels = [0, 1]  # Red and Green channels
        elif color == "red":
            other_channels = [1, 2]  # Green and Blue channels
        else:
            return image

        channel = 1 if color == "green" else (2 if color == "blue" else 0)
        other_channel1, other_channel2 = other_channels

        min_other = np.minimum(image[:,:,other_channel1], image[:,:,other_channel2])
        overdub = image[:,:,channel] - min_other
        overdub = np.maximum(overdub, 0.0)
        
        image[:,:,channel] = image[:,:,channel] - overdub * strength
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def custom_background_removal(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def remove_background(self, images, model, alpha_matting, alpha_matting_foreground_threshold, 
                          alpha_matting_background_threshold, alpha_matting_erode_size, post_process_mask, 
                          only_mask, background_color, edge_detection, edge_detection_threshold1, 
                          edge_detection_threshold2, blur_mask, blur_kernel_size, morphological_operation, 
                          morph_kernel_size, morph_iterations, adaptive_threshold, watershed, grabcut, 
                          color_segmentation, invert_mask, feather_amount, chroma_key, chroma_threshold,
                          color_tolerance, despill, despill_strength, mask=None):
        
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        # Set bgcolor
        bgrgba = None
        if background_color == "black":
            bgrgba = [0, 0, 0, 255]
        elif background_color == "white":
            bgrgba = [255, 255, 255, 255]
        elif background_color == "magenta":
            bgrgba = [255, 0, 255, 255]
        elif background_color == "chroma green":
            bgrgba = [0, 177, 64, 255]
        elif background_color == "chroma blue":
            bgrgba = [0, 71, 187, 255]

        def process_single_image(image, mask_single=None):
            pil_image = tensor2pil(image)
            original_image = np.array(pil_image)
            
            # Process input mask if provided
            if mask_single is not None:
                mask_pil = Image.fromarray(mask_single.squeeze().cpu().numpy().astype(np.uint8) * 255)
                mask_pil = mask_pil.resize(pil_image.size, Image.LANCZOS)  # Resize mask to match image size
                if invert_mask:
                    mask_pil = ImageOps.invert(mask_pil)
                if feather_amount > 0:
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather_amount))
                mask_np = np.array(mask_pil)
            else:
                mask_np = None

            if chroma_key != "none":
                chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
                if mask_np is not None:
                    mask_np = cv2.bitwise_and(mask_np, chroma_mask)
                else:
                    mask_np = chroma_mask

            if model == "custom":
                rembg_mask = self.custom_background_removal(original_image)
            else:
                removed_bg = remove(
                    pil_image,
                    session=self.session,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    post_process_mask=post_process_mask,
                    only_mask=only_mask,
                    bgcolor=bgrgba,
                )
                
                if only_mask:
                    rembg_mask = np.array(removed_bg)
                else:
                    np_removed_bg = np.array(removed_bg)
                    rembg_mask = np_removed_bg[:,:,3]

            if post_process_mask:
                processed_mask = self.post_process_mask(
                    rembg_mask,
                    edge_detection,
                    edge_detection_threshold1,
                    edge_detection_threshold2,
                    blur_mask,
                    blur_kernel_size,
                    morphological_operation,
                    morph_kernel_size,
                    morph_iterations,
                    adaptive_threshold,
                    watershed,
                    grabcut,
                    color_segmentation,
                    original_image
                )
                
                if mask_np is not None:
                    # Ensure mask_np has the same shape as processed_mask
                    mask_np = cv2.resize(mask_np, (processed_mask.shape[1], processed_mask.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    processed_mask = cv2.bitwise_and(processed_mask, mask_np)
                
                final_mask = processed_mask
            else:
                final_mask = rembg_mask

            if only_mask:
                result = final_mask
            else:
                # Apply the mask to the original image
                result = np.dstack((original_image, final_mask))

                if despill and chroma_key in ["green", "blue", "red"]:
                    result = self.despill(result, final_mask, chroma_key, despill_strength)

            return pil2tensor(Image.fromarray(result))

        try:
            if mask is not None:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    processed_images = list(executor.map(process_single_image, images, mask))
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    processed_images = list(executor.map(process_single_image, images, [None] * len(images)))
        except Exception as e:
            logging.error(f"Error during background removal: {str(e)}")
            raise

        return (torch.cat(processed_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "Geeky Remb": GeekyRemb
}
