# ComfyUI-GeekyRemB

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
