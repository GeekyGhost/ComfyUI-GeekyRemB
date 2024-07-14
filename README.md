This is a custom node for ComfyUI
and can be found here!
https://github.com/comfyanonymous/ComfyUI

# ComfyUI-GeekyRemB ComfyUI-GeekyRemB
Overview
ComfyUI-GeekyRemB is a powerful and versatile image processing node for ComfyUI, designed to remove or replace backgrounds with advanced customization options. This node leverages the rembg library and offers a wide range of features for fine-tuning the background removal process and enhancing the resulting images.
Key Features

Multiple Background Removal Models: Supports various AI models for accurate background removal:

u2net
u2netp
u2net_human_seg
u2net_cloth_seg
silueta
isnet-general-use
isnet-anime


Alpha Matting: Enables fine-tuned edge detection for smoother transitions between foreground and background.
Chroma Key: Offers additional background removal using color-based keying for green, blue, or red backgrounds.
Flexible Background Modes:

Transparent
Solid color
Image replacement


Advanced Mask Processing:

Supports input masks
Mask inversion
Feathering for precise control


Edge Detection: Adds customizable outlines to the extracted foreground.
Shadow Generation: Creates realistic drop shadows for extracted foreground objects.
Color Adjustment: Modify brightness, contrast, and saturation of the output image.
Scaling and Positioning: Resize and reposition the foreground image when using image backgrounds.
Batch Processing: Efficiently handles multiple images in a single operation.

Usage

Input an image or batch of images for background removal.

Select the desired AI model for background removal.

Adjust alpha matting settings for edge refinement if needed.

Apply chroma keying if working with solid color backgrounds.

Choose the background mode (transparent, color, or image).

Fine-tune the result with mask processing, edge detection, and shadow options.

Adjust color settings as needed.

Scale and position the foreground when using image backgrounds.

The node outputs both the processed image(s) and the corresponding mask(s), allowing for further manipulation in your ComfyUI workflow.

Implementation Details

For developers interested in the technical aspects:

Chroma Key Implementation: The apply_chroma_key function uses OpenCV (cv2) to perform color-based background removal. It converts the image to HSV color space and creates a mask based on the specified color range.

Image Processing Core: The process_single_image function handles the main image processing logic, including background removal, mask application, and all post-processing effects.

Batch Processing: The remove_background function manages batch processing and overall node operation. It uses tqdm for progress tracking during batch operations.

Tensor Handling: The node uses PyTorch tensors for efficient GPU processing, with helper functions tensor2pil and pil2tensor for conversion between tensor and PIL image formats.

Image Manipulation: Extensive use of PIL (Python Imaging Library) for various image processing tasks, including resizing, rotating, and applying filters.

Installation
[Include instructions for installing the node in ComfyUI]
Dependencies

numpy
rembg
Pillow
torch
opencv-python (cv2)
tqdm

Acknowledgements
Special thanks to WASasquatch for licensing their repository under the MIT license, which has contributed to the development and enhancement of this node.
License
[MIT]
Contributing
[Claude Sonnet 3.5 fixing my mess]

<img width="600" alt="Screenshot 2024-06-29 134500" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/b09a1833-8bdb-43ba-95db-da6f520e8411">

Output examples can be found here - https://civitai.com/models/546180/geeky-remb




Remember, the effectiveness of each setting can vary depending on the input image. Experimentation is key to achieving the best results for your specific use case within ComfyUI workflows.

Inspired by tools found here https://github.com/WASasquatch/was-node-suite-comfyui.git

