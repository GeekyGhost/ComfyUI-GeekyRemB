# GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI
Note, with the most recent update, most new features work, some do not, bear with me please as I make us something that doesn't currently exist lol. 

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [User Guide](#user-guide)
   - [Features](#features)
   - [UI Elements and Their Functions](#ui-elements-and-their-functions)
   - [Use Cases](#use-cases)
   - [Tips and Best Practices](#tips-and-best-practices)
4. [Developer Guide](#developer-guide)
   - [Code Structure](#code-structure)
   - [Key Components](#key-components)
   - [Notable Functions](#notable-functions)
   - [Performance Considerations](#performance-considerations)
   - [Extensibility](#extensibility)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

## Overview

GeekyRemB is a powerful custom node for ComfyUI that offers advanced background removal and image processing capabilities. It combines state-of-the-art AI models with traditional image processing techniques to provide a versatile tool for various image manipulation tasks.

## Installation

1. Navigate to your ComfyUI custom nodes directory.
2. Clone this repository:
   ```
   git clone https://github.com/YourUsername/ComfyUI-GeekyRemB.git
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Restart ComfyUI to load the new node.

## User Guide

### Features

- Advanced background removal using various AI models
- Chroma key functionality
- Flexible image composition with scaling, positioning, and rotation
- Aspect ratio control
- Mask refinement and processing
- Color adjustments and filters
- Edge detection and shadow effects
- Support for batch processing

### UI Elements and Their Functions

1. **Input Image**: The source image(s) to process.

2. **Background Removal Settings**:
   - `enable_background_removal`: Toggle background removal on/off.
   - `model`: Choose the AI model for background removal (e.g., u2net, isnet-anime).
   - `alpha_matting`: Enable for improved edge detection in complex images.
   - `alpha_matting_foreground_threshold`: Fine-tune foreground detection (0-255).
   - `alpha_matting_background_threshold`: Fine-tune background detection (0-255).
   - `post_process_mask`: Apply additional processing to the generated mask.

3. **Chroma Key Settings**:
   - `chroma_key`: Select color to remove (none, green, blue, red).
   - `chroma_threshold`: Adjust sensitivity of color removal (0-255).
   - `color_tolerance`: Fine-tune the range of colors to remove (0-255).

4. **Background Settings**:
   - `background_mode`: Choose output background type (transparent, color, image).
   - `background_color`: Set color for color background mode.
   - `background_loop_mode`: How background images cycle (reverse, loop).

5. **Composition Settings**:
   - `aspect_ratio_preset`: Choose from predefined ratios or custom.
   - `custom_aspect_ratio`: Set a custom ratio (if "custom" is selected).
   - `foreground_scale`: Resize the foreground image (0.1-5.0).
   - `x_position`: Horizontally position the foreground (-10000 to 10000).
   - `y_position`: Vertically position the foreground (-10000 to 10000).
   - `rotation`: Rotate the foreground image (-360 to 360 degrees).
   - `opacity`: Set foreground transparency (0.0-1.0).
   - `flip_horizontal`: Mirror the image horizontally.
   - `flip_vertical`: Flip the image vertically.

6. **Mask Processing**:
   - `invert_mask`: Invert the generated mask.
   - `feather_amount`: Soften mask edges (0-100).
   - `mask_blur`: Apply blur to the mask (0-100).
   - `mask_expansion`: Expand or contract the mask (-100 to 100).

7. **Effects**:
   - `edge_detection`: Add edges to the foreground.
   - `edge_thickness`: Adjust edge thickness (1-10).
   - `edge_color`: Set edge color.
   - `shadow`: Add a shadow effect.
   - `shadow_blur`: Adjust shadow softness (0-20).
   - `shadow_opacity`: Set shadow transparency (0.0-1.0).
   - `shadow_direction`: Set shadow angle (0-360 degrees).
   - `shadow_distance`: Set shadow offset (0-100).

8. **Color Adjustments**:
   - `color_adjustment`: Enable color modifications.
   - `brightness`: Adjust image brightness (0.0-2.0).
   - `contrast`: Modify image contrast (0.0-2.0).
   - `saturation`: Change color saturation (0.0-2.0).
   - `hue`: Shift image hue (-0.5 to 0.5).
   - `sharpness`: Adjust image sharpness (0.0-2.0).

9. **Additional Settings**:
   - `blending_mode`: Choose how foreground blends with background.
   - `blend_strength`: Adjust the intensity of blending (0.0-1.0).
   - `filter`: Apply image filters (blur, sharpen, edge enhance, etc.).
   - `filter_strength`: Adjust the intensity of the chosen filter (0.0-2.0).

10. **Output Options**:
    - `output_format`: Choose between RGBA or RGB output.
    - `only_mask`: Output only the generated mask.

### Use Cases

1. **Product Photography**: Remove backgrounds from product images for e-commerce sites.
2. **Portrait Editing**: Extract subjects from portraits for compositing or style transfer.
3. **Video Game Asset Creation**: Prepare sprites or textures with transparent backgrounds.
4. **Meme Generation**: Quickly remove backgrounds for creating memes or reaction images.
5. **Architectural Visualization**: Extract buildings or elements for architectural mockups.
6. **Fashion Design**: Remove backgrounds from clothing items for lookbooks or catalogs.
7. **Social Media Content**: Create eye-catching posts with custom backgrounds.
8. **Educational Materials**: Prepare images for textbooks, presentations, or online courses.

### Tips and Best Practices

1. **Model Selection**: 
   - Use `u2net` for general-purpose removal.
   - Try `isnet-anime` for cartoon or anime-style images.
   - Experiment with different models for optimal results.

2. **Complex Edges**:
   - Enable `alpha_matting` for images with hair or fur.
   - Adjust `feather_amount` for smoother transitions.

3. **Fine-tuning Removal**:
   - Use `chroma_key` in combination with AI removal for challenging backgrounds.
   - Adjust `mask_blur` and `mask_expansion` for refined edges.

4. **Composition**:
   - Use `foreground_scale`, `x_position`, and `y_position` for precise placement.
   - Experiment with `aspect_ratio_preset` for different layouts.

5. **Realistic Integration**:
   - Apply subtle `shadow` effects for grounded compositions.
   - Use `color_adjustment` to match foreground with new backgrounds.

6. **Batch Processing**:
   - Ensure consistent image sizes for optimal performance.
   - Use `background_loop_mode` for creative batch outputs.

7. **Performance**:
   - Disable `alpha_matting` for faster processing when not needed.
   - Use lower resolution images for rapid prototyping, then switch to high-res for final output.

## Developer Guide

### Code Structure

The main class `GeekyRemB` is defined in `GeekyRembv2.py`. It inherits from a base node class in ComfyUI and overrides key methods for integration.

### Key Components

1. **INPUT_TYPES**: Defines all input parameters, their types, and defaults.
2. **RETURN_TYPES**: Specifies the output types (IMAGE and MASK).
3. **process_image**: The main method that orchestrates the entire image processing pipeline.

### Notable Functions

1. **apply_chroma_key**:
   ```python
   def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
       # ... implementation ...
   ```
   This function uses OpenCV to create a mask based on the specified color range. It's efficient for removing solid color backgrounds.

2. **process_mask**:
   ```python
   def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
       # ... implementation ...
   ```
   Combines multiple mask operations (inversion, feathering, blurring, expansion) into a single efficient function.

3. **apply_blending_mode**:
   ```python
   def apply_blending_mode(self, bg, fg, mode, strength):
       # ... implementation ...
   ```
   Implements various blending modes using NumPy operations, allowing for creative compositing effects.

4. **apply_color_adjustments**:
   ```python
   def apply_color_adjustments(self, image, brightness, contrast, saturation, hue, sharpness):
       # ... implementation ...
   ```
   Uses PIL's `ImageEnhance` for efficient color adjustments.

### Performance Considerations

1. **GPU Acceleration**: 
   ```python
   self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
   ```
   Automatically detects and uses GPU when available for faster processing.

2. **Batch Processing**:
   ```python
   for i in tqdm(range(batch_size), desc="Processing images"):
       # ... process each image ...
   ```
   Efficiently handles multiple images, using tqdm for progress tracking.

3. **Numpy Operations**:
   Extensive use of NumPy for fast array operations, e.g., in blending modes and mask processing.

### Extensibility

1. **Model Support**:
   ```python
   if model == 'u2net_custom' and custom_model_path:
       self.session = new_session('u2net_custom', model_path=custom_model_path, providers=providers)
   ```
   Allows for easy addition of custom models.

2. **Blending Modes**:
   ```python
   class BlendingMode(Enum):
       # ... blending mode definitions ...
   ```
   New blending modes can be easily added by extending this enum and implementing the corresponding logic in `apply_blending_mode`.

3. **Filters**:
   ```python
   def apply_filter(self, image, filter_type, strength):
       # ... implementation ...
   ```
   New filters can be added by extending this function and the corresponding UI option.

## Examples

[Include screenshots or links to example outputs here]

## Troubleshooting

- **Issue**: Slow processing times
  **Solution**: Ensure GPU acceleration is enabled, reduce image size, or disable alpha matting for faster results.

- **Issue**: Poor edge detection
  **Solution**: Try different models, enable alpha matting, or adjust mask processing parameters.

- **Issue**: Unexpected blending results
  **Solution**: Experiment with different blending modes and adjust blend strength.

[Add more common issues and solutions as they arise]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Include your chosen license here]

## Acknowledgements

Special thanks to the creators of rembg, OpenCV, and PIL for their excellent libraries which power much of this node's functionality.

---

Inspired by tools found in [WAS Node Suite for ComfyUI](https://github.com/WASasquatch/was-node-suite-comfyui.git)




<img width="725" alt="Screenshot 2024-07-18 182336" src="https://github.com/user-attachments/assets/dff53dd1-ff4f-48b2-8a96-5f8443cac251">

<img width="688" alt="Screenshot 2024-07-18 191152" src="https://github.com/user-attachments/assets/48281466-9dd7-4dcd-8f1c-3e2bbb69f114">


<img width="600" alt="Screenshot 2024-06-29 134500" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/b09a1833-8bdb-43ba-95db-da6f520e8411">

Output examples can be found here - https://civitai.com/models/546180/geeky-remb





