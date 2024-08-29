Update: 8/29, corrected the ini file issue by adding .sripts

This is a custom node for ComfyUI
which can be found here!
https://github.com/comfyanonymous/ComfyUI

<img width="725" alt="Screenshot 2024-07-18 182336" src="https://github.com/user-attachments/assets/dff53dd1-ff4f-48b2-8a96-5f8443cac251">

<img width="688" alt="Screenshot 2024-07-18 191152" src="https://github.com/user-attachments/assets/48281466-9dd7-4dcd-8f1c-3e2bbb69f114">

GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI
Part 1: User Guide
GeekyRemB is a powerful ComfyUI node that offers advanced background removal and image processing capabilities. Here's a detailed guide on how to use each feature:
Input Parameters:

images: The input image(s) to process.
model: Choose the background removal model (e.g., u2net, isnet-anime).
alpha_matting: Enable for improved edge detection (may be slower).
alpha_matting_foreground_threshold: Adjust for alpha matting precision.
alpha_matting_background_threshold: Adjust for alpha matting precision.
post_process_mask: Apply post-processing to the mask for smoother edges.
chroma_key: Remove specific color backgrounds (none, green, blue, red).
chroma_threshold: Adjust chroma key sensitivity.
color_tolerance: Fine-tune chroma key color range.
background_mode: Choose output background (transparent, color, image).
background_loop_mode: How background images cycle (reverse, loop).
background_width: Set output width.
background_height: Set output height.

Optional Parameters:

output_format: Choose between RGBA or RGB output.
input_masks: Provide custom input masks.
background_images: Supply images for image background mode.
background_color: Set color for color background mode.
invert_mask: Invert the generated mask.
feather_amount: Soften mask edges.
edge_detection: Add edges to the foreground.
edge_thickness: Adjust edge thickness.
edge_color: Set edge color.
shadow: Add a shadow effect.
shadow_blur: Adjust shadow softness.
shadow_opacity: Set shadow transparency.
color_adjustment: Enable color modifications.
brightness: Adjust image brightness.
contrast: Modify image contrast.
saturation: Change color saturation.
x_position: Horizontally position the foreground.
y_position: Vertically position the foreground.
rotation: Rotate the foreground image.
opacity: Set foreground transparency.
flip_horizontal: Mirror the image horizontally.
flip_vertical: Flip the image vertically.
mask_blur: Apply blur to the mask.
mask_expansion: Expand or contract the mask.
foreground_scale: Resize the foreground image.
foreground_aspect_ratio: Adjust the foreground's aspect ratio.

Usage Tips:

Experiment with different models for optimal results with various image types.
Use alpha matting for complex edges like hair or fur.
Combine chroma key with model-based removal for challenging backgrounds.
Adjust mask settings (blur, expansion) for refined edges.
Use color adjustment to match foreground with new backgrounds.
Experiment with shadow settings for realistic compositing.

Part 2: Developer Guide
GeekyRemB is built with extensibility and performance in mind. Here's an overview of its structure and notable features:
Key Components:

GeekyRemB Class: The main node class, handling all processing logic.
INPUT_TYPES: Defines all input parameters, their types, and defaults.
remove_background: The primary function orchestrating the entire process.

Notable Functions:

apply_chroma_key: Implements color-based background removal.
process_mask: Handles various mask modifications (inversion, expansion, blurring).
get_background_image: Manages background image selection and looping.
process_single_image: The core function processing each image.

Unique Aspects:

Flexible Scaling: The foreground_scale and foreground_aspect_ratio parameters allow precise control over foreground dimensions while maintaining or adjusting aspect ratios.
Advanced Mask Processing: The process_mask function combines multiple mask operations efficiently.
Optimized Opacity Handling: Applies opacity to both the foreground image and its mask for accurate blending.
Batch Processing: Efficiently processes multiple images in a batch, utilizing tqdm for progress tracking.
Error Handling and Logging: Comprehensive error catching and informative logging for debugging.
Modular Design: Each major step (chroma keying, mask processing, image composition) is separated into functions for easy maintenance and extension.

Performance Considerations:

Uses NumPy and OpenCV for efficient image processing operations.
Leverages PIL for image manipulations that are more efficiently done in that library.
Implements batch processing to maximize GPU utilization when processing multiple images.

Extensibility:
The modular design allows for easy addition of new features:

New background removal models can be added to the model input type.
Additional image processing effects can be implemented by adding new functions and corresponding input parameters.

Integration with ComfyUI:

The NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS dictionaries allow seamless integration with ComfyUI.
The INPUT_TYPES class method provides a clear interface for ComfyUI to generate the appropriate UI elements.

Developers can extend this node by adding new processing functions, integrating additional background removal models, or enhancing the image composition capabilities. The clear structure and comprehensive error handling make it an excellent starting point for further development.

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

