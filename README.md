# GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI

## Overview
GeekyRemB is a powerful custom node for ComfyUI that offers advanced background removal and image processing capabilities. This node has been updated with new features and improvements to enhance its functionality and user experience.

## New Features and Changes

1. **Color Picker Inputs**: 
   - `background_color` and `edge_color` now use a color picker input instead of a string input, making it more intuitive for users to select colors.

2. **Aspect Ratio Handling**:
   - New `aspect_ratio_preset` option allows users to choose from predefined aspect ratios or a custom ratio.
   - `custom_aspect_ratio` input for specifying custom ratios.

3. **Foreground Scaling**:
   - `foreground_scale` parameter added for easy resizing of the foreground image.

4. **Positioning Improvements**:
   - `x_position` and `y_position` parameters now allow for precise positioning of the foreground image.

5. **Rotation**:
   - New `rotation` parameter for rotating the foreground image.

6. **Flip Options**:
   - `flip_horizontal` and `flip_vertical` boolean options added for mirroring the image.

7. **Enhanced Mask Processing**:
   - `mask_blur` and `mask_expansion` parameters for fine-tuning the mask.

8. **Custom Model Support**:
   - `u2net_custom` option in the model selection, with a `custom_model_path` parameter.

9. **Performance Optimization**:
   - Improved batch processing with better error handling and logging.

10. **UI Improvements**:
    - More intuitive parameter grouping and naming in the ComfyUI interface.

## User Guide

### Input Parameters

#### Required Parameters:
- `images`: The input image(s) to process.
- `model`: Choose the background removal model (e.g., u2net, isnet-anime, u2net_custom).
- `alpha_matting`: Enable for improved edge detection (may be slower).
- `alpha_matting_foreground_threshold`: Adjust for alpha matting precision (0-255).
- `alpha_matting_background_threshold`: Adjust for alpha matting precision (0-255).
- `post_process_mask`: Apply post-processing to the mask for smoother edges.
- `chroma_key`: Remove specific color backgrounds (none, green, blue, red).
- `chroma_threshold`: Adjust chroma key sensitivity (0-255).
- `color_tolerance`: Fine-tune chroma key color range (0-255).
- `background_mode`: Choose output background (transparent, color, image).
- `background_color`: Set color for color background mode (now with color picker).
- `background_loop_mode`: How background images cycle (reverse, loop).
- `aspect_ratio_preset`: Choose from predefined aspect ratios or custom.
- `custom_aspect_ratio`: Set a custom aspect ratio (if "custom" is selected).
- `foreground_scale`: Resize the foreground image (0.1-5.0).
- `x_position`: Horizontally position the foreground (-10000 to 10000).
- `y_position`: Vertically position the foreground (-10000 to 10000).
- `rotation`: Rotate the foreground image (-360 to 360 degrees).
- `opacity`: Set foreground transparency (0.0-1.0).
- `flip_horizontal`: Mirror the image horizontally (boolean).
- `flip_vertical`: Flip the image vertically (boolean).
- `invert_mask`: Invert the generated mask (boolean).
- `feather_amount`: Soften mask edges (0-100).
- `edge_detection`: Add edges to the foreground (boolean).
- `edge_thickness`: Adjust edge thickness (1-10).
- `edge_color`: Set edge color (now with color picker).
- `shadow`: Add a shadow effect (boolean).
- `shadow_blur`: Adjust shadow softness (0-20).
- `shadow_opacity`: Set shadow transparency (0.0-1.0).
- `color_adjustment`: Enable color modifications (boolean).
- `brightness`: Adjust image brightness (0.0-2.0).
- `contrast`: Modify image contrast (0.0-2.0).
- `saturation`: Change color saturation (0.0-2.0).
- `mask_blur`: Apply blur to the mask (0-100).
- `mask_expansion`: Expand or contract the mask (-100 to 100).

#### Optional Parameters:
- `input_masks`: Provide custom input masks.
- `background_images`: Supply images for image background mode.
- `output_format`: Choose between RGBA or RGB output.
- `only_mask`: Output only the mask (boolean).
- `custom_model_path`: Path to a custom U2Net model (for u2net_custom).

### Usage Tips

1. **Model Selection**: 
   - Experiment with different models for optimal results with various image types.
   - Use the new `u2net_custom` option with `custom_model_path` for specialized models.

2. **Background Removal**:
   - Use alpha matting for complex edges like hair or fur.
   - Combine chroma key with model-based removal for challenging backgrounds.

3. **Mask Refinement**:
   - Adjust `mask_blur` and `mask_expansion` for refined edges.
   - Use `feather_amount` for softer transitions.

4. **Composition**:
   - Utilize `foreground_scale`, `x_position`, `y_position`, and `rotation` for precise placement.
   - Experiment with `aspect_ratio_preset` and `custom_aspect_ratio` for different layouts.

5. **Effects**:
   - Use color adjustment (brightness, contrast, saturation) to match foreground with new backgrounds.
   - Experiment with shadow settings for realistic compositing.

6. **Performance**:
   - For batch processing, ensure consistent image sizes for optimal performance.

## Developer Guide

### Key Components

- `GeekyRemB` Class: The main node class, handling all processing logic.
- `INPUT_TYPES`: Defines all input parameters, their types, and defaults.
- `remove_background`: The primary function orchestrating the entire process.

### Notable Functions

- `apply_chroma_key`: Implements color-based background removal.
- `process_mask`: Handles various mask modifications (inversion, expansion, blurring).
- `parse_aspect_ratio`: Manages aspect ratio calculations.
- `get_background_image`: Manages background image selection and looping.
- `process_single_image`: The core function processing each image.

### Unique Aspects

1. **Flexible Scaling and Positioning**: 
   - `foreground_scale`, `x_position`, `y_position`, and `rotation` allow precise control.
   - `parse_aspect_ratio` function handles various aspect ratio inputs.

2. **Advanced Mask Processing**: 
   - `process_mask` function combines multiple mask operations efficiently.

3. **Optimized Opacity Handling**: 
   - Applies opacity to both the foreground image and its mask for accurate blending.

4. **Batch Processing**: 
   - Efficiently processes multiple images in a batch, utilizing tqdm for progress tracking.

5. **Error Handling and Logging**: 
   - Comprehensive error catching and informative logging for debugging.

6. **Modular Design**: 
   - Each major step (chroma keying, mask processing, image composition) is separated into functions for easy maintenance and extension.

### Performance Considerations

- Uses NumPy and OpenCV for efficient image processing operations.
- Leverages PIL for image manipulations that are more efficiently done in that library.
- Implements batch processing to maximize GPU utilization when processing multiple images.

### Extensibility

The modular design allows for easy addition of new features:
- New background removal models can be added to the model input type.
- Additional image processing effects can be implemented by adding new functions and corresponding input parameters.

### Integration with ComfyUI

- The `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dictionaries allow seamless integration with ComfyUI.
- The `INPUT_TYPES` class method provides a clear interface for ComfyUI to generate the appropriate UI elements.

## Acknowledgements

Special thanks to WASasquatch for licensing their repository under the MIT license, which has contributed to the development and enhancement of this node.

## License

[MIT]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Examples

Output examples can be found here: [GeekyRemB on CivitAI](https://civitai.com/models/546180/geeky-remb)

Remember, the effectiveness of each setting can vary depending on the input image. Experimentation is key to achieving the best results for your specific use case within ComfyUI workflows.

---

Inspired by tools found in [WAS Node Suite for ComfyUI](https://github.com/WASasquatch/was-node-suite-comfyui.git)




<img width="725" alt="Screenshot 2024-07-18 182336" src="https://github.com/user-attachments/assets/dff53dd1-ff4f-48b2-8a96-5f8443cac251">

<img width="688" alt="Screenshot 2024-07-18 191152" src="https://github.com/user-attachments/assets/48281466-9dd7-4dcd-8f1c-3e2bbb69f114">


<img width="600" alt="Screenshot 2024-06-29 134500" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/b09a1833-8bdb-43ba-95db-da6f520e8411">

Output examples can be found here - https://civitai.com/models/546180/geeky-remb





