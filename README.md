Save the python file into your custom nodes folder and run ComfyUI. Use the one labled v2. 

# ComfyUI-GeekyRemB v2 (Pretty much complete used Claude Sonnet to help me refine v1 and make v2) 

GeekyRemB Node Description:
GeekyRemB is a powerful and versatile image processing node for ComfyUI, designed to remove backgrounds from images with advanced customization options. This node leverages the rembg library and offers a wide range of features for fine-tuning the background removal process and enhancing the resulting images.
Key Features:

Background Removal Models: Supports multiple AI models (u2net, u2netp, u2net_human_seg, u2net_cloth_seg, silueta, isnet-general-use, isnet-anime) for accurate background removal.
Alpha Matting: Enables fine-tuned edge detection for smoother transitions between foreground and background.
Chroma Key: Offers additional background removal using color-based keying for green, blue, or red backgrounds.
Background Modes: Allows for transparent backgrounds, solid color backgrounds, or image replacement backgrounds.
Mask Processing: Supports input masks, mask inversion, and feathering for precise control over the area of effect.
Edge Detection: Adds outlines to the extracted foreground for enhanced visibility or stylistic effects.
Shadow Generation: Creates realistic drop shadows for the extracted foreground objects.
Color Adjustment: Provides options to modify brightness, contrast, and saturation of the output image.
Scaling and Positioning: Allows resizing and repositioning of the foreground image when using image backgrounds.
Batch Processing: Efficiently handles multiple images in a single operation.

Usage:

Input an image or batch of images to remove backgrounds.
Select the desired AI model for background removal.
Adjust alpha matting settings for edge refinement.
Apply chroma keying if working with solid color backgrounds.
Choose the background mode (transparent, color, or image).
Fine-tune the result with mask processing, edge detection, and shadow options.
Adjust color settings as needed.
Scale and position the foreground when using image backgrounds.

The node outputs both the processed image(s) and the corresponding mask(s), allowing for further manipulation in your ComfyUI workflow.
For Developers:
Key functions to note include:

apply_chroma_key: Implements color-based background removal.
process_single_image: Handles the core image processing logic.
remove_background: Manages batch processing and overall node operation.

The node uses PyTorch tensors for efficient GPU processing and PIL for image manipulation. It's designed to be flexible and extensible, making it a valuable addition to image processing pipelines.
Special thanks to WASasquatch for licensing their repository under the MIT license, which has contributed to the development and enhancement of this node.

<img width="600" alt="Screenshot 2024-06-29 134500" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/b09a1833-8bdb-43ba-95db-da6f520e8411">


# ComfyUI-GeekyRemB v1 (Wip - Rough)

WIP - Some features not fully functional (watershed), but the majority are fully operational.  

Custom Background Remover for ComfyUI to address some issues I've encountered with different background removers. 

GeekyRemb: Advanced Background Removal Node
This custom node provides a powerful and flexible tool for removing backgrounds from images within ComfyUI workflows. It combines multiple techniques and offers various options for fine-tuning the background removal process.
How to Use:

Connect an image output from a previous node to the "images" input of GeekyRemb.
Optionally, connect a mask to the "mask" input if you want to guide the background removal.
Adjust the settings as needed for your specific use case.
The node will output the processed image(s) with the background removed.

Settings Explanation:

Model: Choose the background removal model to use. Options include:

u2net, u2netp, u2net_human_seg, silueta, isnet-general-use, isnet-anime, briarmbg: Different pre-trained models for various types of images.
custom: Uses a basic threshold-based method (useful for simple images).


Alpha Matting: Enable for smoother edges, especially useful for hair or fuzzy objects.

Alpha Matting Foreground Threshold: Adjust to fine-tune foreground detection.
Alpha Matting Background Threshold: Adjust to fine-tune background detection.
Alpha Matting Erode Size: Controls the erosion of the alpha matte.


Post Process Mask: Enable to apply additional processing to the generated mask.
Only Mask: If enabled, outputs only the mask instead of the full image.
Background Color: Choose a color to replace the removed background, or leave transparent.
Edge Detection: Apply edge detection to refine the mask.

Edge Detection Threshold1 and Threshold2: Adjust the sensitivity of edge detection.


Blur Mask: Apply Gaussian blur to smooth the mask edges.

Blur Kernel Size: Controls the strength of the blur.


Morphological Operation: Apply morphological operations to the mask.

Morph Kernel Size: Size of the kernel for morphological operations.
Morph Iterations: Number of times to apply the operation.


Adaptive Threshold: Apply adaptive thresholding to the mask.
Watershed: Use the watershed algorithm for more precise segmentation.
Grabcut: Apply the GrabCut algorithm for improved foreground extraction.
Color Segmentation: Use color-based segmentation to refine the mask.
Invert Mask: Invert the input mask if provided.
Feather Amount: Apply feathering to soften the edges of the mask.
Chroma Key: Enable chroma keying to remove specific colors.

Chroma Threshold: Adjust the sensitivity of chroma keying.
Color Tolerance: Controls the range of colors considered for chroma keying.


Despill: Remove color spill when using chroma keying.

Despill Strength: Adjust the strength of the despill effect.



Tips for Usage:

Start with the default settings and gradually adjust as needed.
For human subjects, try the "u2net_human_seg" model.
Use alpha matting for images with hair or fuzzy edges.
Experiment with post-processing options for challenging images.
Combine with other ComfyUI nodes for more advanced workflows, such as:

Using the output in image composition nodes.
Applying the removed background to different images.
Creating masks for selective processing in other nodes.



Remember, the effectiveness of each setting can vary depending on the input image. Experimentation is key to achieving the best results for your specific use case within ComfyUI workflows.

Inspired by tools found here https://github.com/WASasquatch/was-node-suite-comfyui.git


<img width="907" alt="Screenshot 2024-06-28 113238" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/fc5b0df2-6410-4751-8719-6eb6841574cb">
