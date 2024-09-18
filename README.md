# GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI

GeekyRemB is a powerful custom node for ComfyUI that offers advanced background removal, image processing, and animation capabilities. It combines state-of-the-art AI models with traditional image processing techniques to provide a versatile tool for complex image manipulation tasks.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Advanced Usage](#advanced-usage)
5. [Developer's Guide](#developers-guide)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Features

- Advanced background removal using multiple AI models (u2net, u2netp, u2net_human_seg, u2net_cloth_seg, silueta, isnet-general-use, isnet-anime)
- Chroma key functionality for color-based removal
- Comprehensive mask processing (expansion, edge detection, blurring, thresholding)
- Flexible image composition with scaling, positioning, and rotation
- Multiple animation types (bounce, travel, rotate, fade, zoom)
- Support for batch processing and handling multiple frames
- Alpha matting for improved edge detection
- Ability to remove small regions from the mask
- Option to use additional masks and invert them

## Installation

1. Ensure you have ComfyUI installed and set up.
2. Clone the GeekyRemB repository into your ComfyUI custom nodes directory:
   ```
   git clone https://github.com/YourUsername/ComfyUI-GeekyRemB.git
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Restart ComfyUI to load the new node.

## Usage

### Basic Usage

1. In ComfyUI, locate the "GeekyRemB" node in the node browser.
2. Connect your input image to the "foreground" input of the GeekyRemB node.
3. Configure the node parameters according to your needs (see Parameter Details below).
4. Connect the outputs to your desired destination nodes (e.g., Save Image, Preview Image).

### Parameter Details

- **enable_background_removal**: Toggle background removal on/off.
- **removal_method**: Choose between "rembg" (AI-based) or "chroma_key".
- **model**: Select the AI model for background removal (if using "rembg").
- **chroma_key_color**: Select the color to be removed (if using "chroma_key").
- **chroma_key_tolerance**: Adjust the tolerance for chroma keying.
- **mask_expansion**: Expand or contract the mask (-100 to 100).
- **edge_detection**: Enable edge detection on the mask.
- **edge_thickness**: Set the thickness of detected edges (1-10).
- **mask_blur**: Apply Gaussian blur to the mask (0-100).
- **threshold**: Set the threshold for mask generation (0.0-1.0).
- **invert_generated_mask**: Invert the generated mask.
- **remove_small_regions**: Remove small isolated regions from the mask.
- **small_region_size**: Set the size threshold for small region removal.
- **alpha_matting**: Enable alpha matting for improved edge detection.
- **alpha_matting_foreground_threshold**: Set foreground threshold for alpha matting.
- **alpha_matting_background_threshold**: Set background threshold for alpha matting.
- **animation_type**: Choose the type of animation to apply.
- **animation_speed**: Set the speed of the animation.
- **animation_frames**: Set the number of frames for the animation.
- **x_position**, **y_position**: Set the position of the foreground image.
- **scale**: Scale the foreground image.
- **rotation**: Rotate the foreground image.

### Optional Inputs

- **background**: Connect a background image or sequence.
- **additional_mask**: Provide an additional mask to combine with the generated one.
- **invert_additional_mask**: Invert the additional mask before combining.

## Advanced Usage

### Handling Multiple Frames

GeekyRemB can process multiple input frames, useful for creating animations or batch processing:

1. Connect a sequence of images to the "foreground" input.
2. Set the "animation_frames" parameter to the desired number of output frames.
3. Choose an animation type or set it to "none" for static processing of each frame.

### Combining Multiple Masks

To use an additional mask with the generated one:

1. Connect your mask to the "additional_mask" input.
2. The node will combine this mask with the generated one using the minimum operation.
3. Use "invert_additional_mask" if you need to invert the additional mask before combining.

### Creating Complex Animations

Experiment with different animation types and parameters to create complex effects:

1. Set "animation_type" to your desired animation (e.g., "rotate", "zoom_in").
2. Adjust "animation_speed" and "animation_frames" to control the animation.
3. Use "x_position", "y_position", "scale", and "rotation" for fine-tuning.

## Developer's Guide

### Key Components

1. **Background Removal**:
   - `remove_background_rembg`: Handles AI-based removal using the rembg library.
   - `remove_background_chroma`: Implements chroma key-based removal.

2. **Mask Processing**:
   - `refine_mask`: Applies various refinements to the generated mask.

3. **Animation**:
   - `animate_element`: Handles different types of animations on the foreground image.

4. **Image Composition**:
   - `process_image`: The main method that orchestrates the entire process.

### Extending GeekyRemB

To add new features or modify existing ones:

1. **Adding a New Animation Type**:
   - Add the new type to the `AnimationType` enum.
   - Implement the animation logic in the `animate_element` method.

2. **Implementing a New Background Removal Method**:
   - Create a new method similar to `remove_background_rembg` or `remove_background_chroma`.
   - Add the new method as an option in the `process_image` method.

3. **Adding New Mask Processing Techniques**:
   - Extend the `refine_mask` method with new processing options.
   - Update the `INPUT_TYPES` to include parameters for the new techniques.

4. **Optimizing Performance**:
   - Consider using GPU acceleration for heavy computations.
   - Implement caching mechanisms for frequently used data or results.

### Important Considerations

- Ensure compatibility with various image formats and sizes.
- Handle errors gracefully and provide informative error messages.
- Optimize memory usage, especially when dealing with large images or multiple frames.
- Maintain backwards compatibility when adding new features.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Note: Some included models may require separate licensing for commercial use. Ensure you have appropriate licensing if you intend to use these models commercially.

## Acknowledgements

GeekyRemB builds upon the work of many in the open-source community. We extend our sincere gratitude to:

- Daniel Gatis (https://github.com/danielgatis/rembg), creator of the Rembg library, which forms the backbone of our background removal capabilities.
- The creators and contributors of ComfyUI (https://github.com/comfyanonymous/ComfyUI), whose innovative work has made this project possible.
- WAS Node Suite (https://github.com/WASasquatch/was-node-suite-comfyui), particularly the Layer Utility Node, which provided inspiration for some of our features.
- The broader open-source community, whose collective efforts continue to push the boundaries of what's possible in image processing and AI.

Your work has been instrumental in the development of GeekyRemB, and we're deeply appreciative of your contributions to the field.

---

This project is continually evolving. We welcome feedback, bug reports, and contributions as we work to improve GeekyRemB.
