# GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI

GeekyRemB is a powerful custom node for ComfyUI that offers advanced background removal and comprehensive image processing capabilities. It combines state-of-the-art AI models with traditional image processing techniques to provide a versatile tool for various image manipulation tasks.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Key Components](#key-components)
5. [Advanced Features](#advanced-features)
6. [Performance Considerations](#performance-considerations)
7. [Extensibility](#extensibility)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Features

1. **Advanced Background Removal**
   - Multiple AI models: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use, isnet-anime, bria, birefnet
   - Chroma key functionality for color-based removal
   - Alpha matting for improved edge detection

2. **Image Composition**
   - Flexible scaling, positioning, and rotation
   - Aspect ratio control
   - Flip horizontal and vertical options

3. **Mask Processing**
   - Mask inversion, feathering, blurring, and expansion
   - Edge detection with customizable thickness and color

4. **Color Adjustments and Filters**
   - Brightness, contrast, saturation, hue, and sharpness adjustments
   - Various filters: blur, sharpen, edge enhance, emboss, toon, sepia, film grain, matrix effect

5. **Blending Modes**
   - Multiple blending modes for creative compositing
   - Adjustable blend strength

6. **Shadow Effects**
   - Customizable shadow with adjustable blur, opacity, direction, and distance

7. **Animation Capabilities**
   - Various animation types: bounce, travel, rotate, fade, zoom
   - Adjustable animation speed and frame count

8. **Matrix Background Generation**
   - Customizable matrix rain effect
   - Options for foreground pattern, density, fall speed, and glow

9. **Batch Processing**
   - Support for processing multiple images in a batch

10. **Output Options**
    - Choice between RGBA and RGB output formats
    - Option to output only the generated mask

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

## Usage

1. In ComfyUI, locate the "GeekyRemB" node in the node browser.
2. Connect your input image to the "foreground" input of the GeekyRemB node.
3. Adjust the parameters as needed:
   - Choose a background removal method and model
   - Set composition options (scale, position, rotation)
   - Apply color adjustments and filters
   - Configure animation settings if desired
4. Connect the output to your desired destination node (e.g., Save Image, Preview Image).

## Key Components

- **Background Removal**: Choose between AI-based removal (using models like u2net) or chroma key.
- **Mask Refinement**: Fine-tune the generated mask with expansion, blurring, and feathering options.
- **Composition Tools**: Precise control over image placement, scaling, and rotation.
- **Color and Filter Effects**: Enhance or stylize your images with various adjustments and filters.
- **Animation**: Create dynamic effects with multiple animation types.
- **Matrix Background**: Generate custom Matrix-style rain effect backgrounds.

## Advanced Features

- **Alpha Matting**: Improve edge detection for complex images like hair or fur.
- **Custom Aspect Ratios**: Define your own aspect ratios for precise composition.
- **Blending Modes**: Experiment with different blending modes for creative effects.
- **Shadow Generation**: Add realistic shadows to composited images.
- **Batch Processing**: Efficiently process multiple images or create animations.

## Performance Considerations

- GPU acceleration is automatically used when available for faster processing.
- Efficient NumPy operations are used for image manipulations.
- Caching mechanisms are implemented to avoid reloading models for repeated use.

## Extensibility

Developers can extend GeekyRemB by:
- Adding new background removal models
- Implementing additional blending modes
- Creating new filters and effects
- Expanding animation capabilities

Refer to the developer guide in the documentation for detailed instructions on extending functionality.

## Examples



https://github.com/user-attachments/assets/3405d87f-cb67-4875-a121-7b9bbfcc8d8c

https://github.com/user-attachments/assets/a5de952c-6411-4932-a074-3b64a503c4f8

https://github.com/user-attachments/assets/b6aa7384-7ee0-4611-82ac-00f330226407




More output examples can be found [here](https://civitai.com/models/546180/geeky-remb).

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are correctly installed.
2. Check that your input image is in a supported format.
3. Try using different models or settings for background removal.
4. For performance issues, consider using smaller images or disabling GPU acceleration if it's causing problems.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Note: The 'bria' model is included for its good performance, but requires a license for commercial use. Ensure you have appropriate licensing if you intend to use this model commercially.

## Acknowledgements

GeekyRemB builds upon various open-source projects and research. We're grateful to the entire open-source community for their contributions. A full list of acknowledgements is in progress and will be included in future updates.

Inspired by tools found in [WAS Node Suite for ComfyUI](https://github.com/WASasquatch/was-node-suite-comfyui.git).

---

This project is a work in progress. Some features may not be fully functional, and we appreciate your patience and contributions as we continue to improve GeekyRemB.
