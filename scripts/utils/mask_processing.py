import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class MaskProcessingConfig:
    """Configuration for mask processing parameters"""
    mask_expansion: int = 0
    edge_detection: bool = False
    edge_thickness: int = 1
    edge_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    mask_blur: int = 5
    threshold: float = 0.5
    invert_generated_mask: bool = False
    remove_small_regions: bool = False
    small_region_size: int = 100
    edge_refinement: bool = False

class EnhancedMaskProcessor:
    """Enhanced mask processing with advanced refinement techniques"""
    
    @staticmethod
    def refine_mask(mask: Image.Image, config: MaskProcessingConfig, original_image: Image.Image) -> Image.Image:
        """Enhanced mask refinement with improved edge detection and color control"""
        try:
            # Convert mask to numpy array
            mask_np = np.array(mask)
            if len(mask_np.shape) > 2:
                if mask_np.shape[2] == 4:
                    mask_np = mask_np[:, :, 3]
                else:
                    mask_np = mask_np[:, :, 0]

            # Initial binary threshold
            _, binary_mask = cv2.threshold(
                mask_np,
                127,
                255,
                cv2.THRESH_BINARY
            )

            # Enhanced Edge Detection
            if config.edge_detection:
                # Detect edges using Canny
                edges = cv2.Canny(binary_mask, 100, 200)
                
                # Create kernel based on edge_thickness
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (config.edge_thickness, config.edge_thickness)
                )
                
                # Dilate edges
                edges = cv2.dilate(edges, kernel)
                
                # Convert edge color from grayscale to color mask
                edge_mask = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
                edge_mask[edges > 0] = config.edge_color  # Edge color from config
                
                # Blend edges with original mask
                binary_mask = cv2.addWeighted(
                    binary_mask, 
                    0.7,
                    edges,
                    0.3,
                    0
                )

            # Handle mask expansion
            if config.mask_expansion != 0:
                kernel_size = abs(config.mask_expansion) // 10 + 1  # Scale factor for iteration count
                kernel = np.ones((2, 2), np.uint8)
                if config.mask_expansion > 0:
                    binary_mask = cv2.dilate(binary_mask, kernel, iterations=kernel_size)
                else:
                    binary_mask = cv2.erode(binary_mask, kernel, iterations=kernel_size)

            # Apply blur for smoother edges
            if config.mask_blur > 0:
                blur_amount = min(config.mask_blur, 20)  # Limit maximum blur
                blur_kernel_size = blur_amount * 2 + 1  # Ensure odd kernel size
                binary_mask = cv2.GaussianBlur(
                    binary_mask,
                    (blur_kernel_size, blur_kernel_size),
                    0
                )

            # Edge refinement for better quality edges
            if config.edge_refinement:
                # Create erosion and dilation to identify edge areas
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(binary_mask, kernel, iterations=1)
                dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                
                # Edge mask (areas that differ between erosion and dilation)
                edge_mask = dilated - eroded
                
                # Apply stronger blur only to edge areas
                strong_blur = cv2.GaussianBlur(binary_mask, (9, 9), 0)
                
                # Blend the edge areas with strong blur, keep the rest from original mask
                binary_mask = np.where(edge_mask > 0, strong_blur, binary_mask)
            
            if config.remove_small_regions:
                # Remove small disconnected areas
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
                
                # Find the largest component (background is usually labeled as 0)
                largest_label = 1  # Default to first component if no larger one is found
                largest_size = 0
                
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] > largest_size:
                        largest_size = stats[i, cv2.CC_STAT_AREA]
                        largest_label = i
                
                # Keep only regions larger than the configured size
                clean_mask = np.zeros_like(binary_mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] > config.small_region_size:
                        clean_mask[labels == i] = 255
                binary_mask = clean_mask

            if config.invert_generated_mask:
                binary_mask = 255 - binary_mask

            return Image.fromarray(binary_mask.astype(np.uint8), 'L')

        except Exception as e:
            logger.error(f"Mask refinement failed: {str(e)}")
            return mask

@dataclass
class ChromaKeyConfig:
    """Configuration for chroma key processing"""
    chroma_key_color: Union[str, Tuple[int, int, int]] = "green"
    chroma_key_tolerance: float = 0.1
    spill_reduction: float = 0.0
    edge_refinement: bool = False

def remove_background_chroma(image: Image.Image, config: ChromaKeyConfig) -> Image.Image:
    """Enhanced chroma key with spill reduction and edge refinement"""
    try:
        # Convert to numpy array
        img_np = np.array(image)
        original_img = img_np.copy()
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        # Convert to HSV color space
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Basic color ranges
        tolerance = int(30 * config.chroma_key_tolerance)
        saturation_min = 30
        value_min = 30

        # Color-specific ranges
        if config.chroma_key_color == "green":
            lower = np.array([55 - tolerance, saturation_min, value_min])
            upper = np.array([65 + tolerance, 255, 255])
        elif config.chroma_key_color == "blue":
            lower = np.array([110 - tolerance, saturation_min, value_min])
            upper = np.array([130 + tolerance, 255, 255])
        else:  # red
            lower1 = np.array([0, saturation_min, value_min])
            upper1 = np.array([tolerance, 255, 255])
            lower2 = np.array([180 - tolerance, saturation_min, value_min])
            upper2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # Create mask for non-red colors
        if config.chroma_key_color != "red":
            mask = cv2.inRange(hsv, lower, upper)

        # Add spill suppression
        if config.spill_reduction > 0:
            # Create a wider range for spill detection
            spill_tolerance = int(tolerance * 1.5)
            
            if config.chroma_key_color == "green":
                spill_lower = np.array([55 - spill_tolerance, saturation_min//2, value_min])
                spill_upper = np.array([65 + spill_tolerance, 255, 255])
                spill_mask = cv2.inRange(hsv, spill_lower, spill_upper)
            elif config.chroma_key_color == "blue":
                spill_lower = np.array([110 - spill_tolerance, saturation_min//2, value_min])
                spill_upper = np.array([130 + spill_tolerance, 255, 255])
                spill_mask = cv2.inRange(hsv, spill_lower, spill_upper)
            else:  # red
                spill_lower1 = np.array([0, saturation_min//2, value_min])
                spill_upper1 = np.array([spill_tolerance, 255, 255])
                spill_lower2 = np.array([180 - spill_tolerance, saturation_min//2, value_min])
                spill_upper2 = np.array([180, 255, 255])
                spill_mask1 = cv2.inRange(hsv, spill_lower1, spill_upper1)
                spill_mask2 = cv2.inRange(hsv, spill_lower2, spill_upper2)
                spill_mask = cv2.bitwise_or(spill_mask1, spill_mask2)
            
            # Find transition areas (spill areas not fully keyed out)
            transition_mask = cv2.bitwise_and(spill_mask, cv2.bitwise_not(mask))
            
            # Apply spill reduction to transition areas
            if np.any(transition_mask > 0):
                # Convert RGB to grayscale only for spill areas
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                
                # Create an image where spill areas are desaturated
                desaturated_img = img_np.copy()
                
                # Apply weighted desaturation based on spill_reduction setting
                desaturation_strength = config.spill_reduction
                transition_pixels = (transition_mask > 0)
                
                # Replace color channels with grayscale weighted by strength
                for c in range(3):
                    desaturated_img[:, :, c] = np.where(
                        transition_pixels,
                        img_np[:, :, c] * (1 - desaturation_strength) + gray * desaturation_strength,
                        img_np[:, :, c]
                    )
                
                # Use the desaturated image for further processing
                img_np = desaturated_img

        # Invert mask (0 for background, 255 for foreground)
        mask = 255 - mask
        
        # Edge refinement
        if config.edge_refinement:
            # Create eroded and dilated versions to identify edge regions
            kernel = np.ones((3, 3), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            
            # Edge regions are the difference between dilated and eroded masks
            edge_regions = mask_dilated - mask_eroded
            
            # Apply strong blur only to edge regions
            blurred_mask = cv2.GaussianBlur(mask, (9, 9), 0)
            
            # Combine: use blurred mask for edge regions, original mask elsewhere
            refined_mask = np.where(edge_regions > 0, blurred_mask, mask)
            mask = refined_mask

        # Return the processed mask
        return Image.fromarray(mask, 'L')

    except Exception as e:
        logger.error(f"Chroma key removal failed: {str(e)}")
        raise RuntimeError(f"Chroma key removal failed: {str(e)}")