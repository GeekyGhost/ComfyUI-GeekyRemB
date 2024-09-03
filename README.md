# GeekyRemB: Advanced Background Removal and Image Processing Node for ComfyUI
Note, with the most recent update, most new features work, some do not, bear with me please as I make us something that doesn't currently exist lol. briefnet is a new addition I'm working to add, the zoom in and out features aren't functional yet. Most filters and animations work, but some do not. Acknowledgements is a work in progress. The list of MIT repos is long at this stage. There is a massive community with a massive amount of tools with various opensource friendly licensing that make so much of this all work, like all of it. Our infrastructure itself, many systems we all rely on, many of our tools, especially AI tools. Without the repository of open knowledge everyone has created, none of the things we do tomorrow would be possible, none of the things we do today we be so grand in comparison to what we once thought our limitaions were. I should probably have Claude or Grok rewrite this so I don't sound so cheesy, but everyones work mattered. It was cool to see you guys make so many awesome things, so much so I wanted to give back with what I learned. So, even the acknowledgements are a work in progress. 

# GeekyRemB: Comprehensive User and Developer Guide

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
   - [New Features and Functions](#new-features-and-functions)
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

1. **Advanced Background Removal**: 
   - Multiple AI models including u2net, isnet-anime, bria, and birefnet
   - Chroma key functionality for color-based removal
   - Alpha matting for improved edge detection

2. **Image Composition**:
   - Flexible scaling, positioning, and rotation
   - Aspect ratio control
   - Flip horizontal and vertical options

3. **Mask Processing**:
   - Mask inversion, feathering, blurring, and expansion
   - Edge detection with customizable thickness and color

4. **Color Adjustments and Filters**:
   - Brightness, contrast, saturation, hue, and sharpness adjustments
   - Various filters including blur, sharpen, edge enhance, emboss, toon, sepia, film grain, and matrix effect

5. **Blending Modes**:
   - Multiple blending modes for creative compositing
   - Adjustable blend strength

6. **Shadow Effects**:
   - Customizable shadow with adjustable blur, opacity, direction, and distance

7. **Animation Capabilities**:
   - Various animation types including bounce, travel, rotate, fade, and zoom
   - Adjustable animation speed and frame count

8. **Matrix Background Generation**:
   - Customizable matrix rain effect
   - Options for foreground pattern, density, fall speed, and glow

9. **Batch Processing**:
   - Support for processing multiple images in a batch

10. **Output Options**:
    - Choice between RGBA and RGB output formats
    - Option to output only the generated mask

### UI Elements and Their Functions

1. **Input Image**: The source image(s) to process.

2. **Background Removal Settings**:
   - `enable_background_removal`: Toggle background removal on/off.
   - `model`: Choose the AI model for background removal (e.g., u2net, isnet-anime, bria, birefnet).
   - `alpha_matting`: Enable for improved edge detection in complex images.
   - `alpha_matting_foreground_threshold`: Fine-tune foreground detection (0-255).
   - `alpha_matting_background_threshold`: Fine-tune background detection (0-255).
   - `post_process_mask`: Apply additional processing to the generated mask.

3. **Chroma Key Settings**:
   - `chroma_key`: Select color to remove (none, green, blue, red).
   - `chroma_threshold`: Adjust sensitivity of color removal (0-255).
   - `color_tolerance`: Fine-tune the range of colors to remove (0-255).

4. **Background Settings**:
   - `background_mode`: Choose output background type (transparent, color, image, matrix).
   - `background_color`: Set color for color background mode.
   - `background_loop_mode`: How background images cycle (reverse, loop).

5. **Composition Settings**:
   - `aspect_ratio_preset`: Choose from predefined ratios or custom.
   - `custom_aspect_ratio`: Set a custom ratio (if "custom" is selected).
   - `foreground_scale`: Resize the foreground image (0.1-5.0).
   - `x_position`, `y_position`: Position the foreground (-10000 to 10000).
   - `rotation`: Rotate the foreground image (-360 to 360 degrees).
   - `opacity`: Set foreground transparency (0.0-1.0).
   - `flip_horizontal`, `flip_vertical`: Mirror or flip the image.

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
   - `brightness`, `contrast`, `saturation`, `hue`, `sharpness`: Adjust respective properties (0.0-2.0, hue: -0.5 to 0.5).

9. **Additional Settings**:
   - `blending_mode`: Choose how foreground blends with background.
   - `blend_strength`: Adjust the intensity of blending (0.0-1.0).
   - `filter`: Apply image filters (blur, sharpen, edge enhance, etc.).
   - `filter_strength`: Adjust the intensity of the chosen filter (0.0-2.0).

10. **Matrix Background Settings**:
    - `matrix_foreground_pattern`: Choose between BINARY, RANDOM, or CUSTOM patterns.
    - `matrix_custom_text`: Set custom text for the matrix background.
    - `matrix_density`: Adjust the density of the matrix rain effect (0.1-10.0).
    - `matrix_fall_speed`: Set the speed of the falling characters (1-20).
    - `matrix_glow`: Enable or disable the glow effect.
    - `matrix_glow_intensity`: Adjust the intensity of the glow effect (0.1-1.0).

11. **Animation Settings**:
    - `animation_type`: Choose from NONE, BOUNCE, TRAVEL_LEFT, TRAVEL_RIGHT, ROTATE, FADE_IN, FADE_OUT, ZOOM_IN, ZOOM_OUT.
    - `animation_speed`: Set the speed of the animation (0.1-10.0).
    - `animation_frames`: Number of frames to generate for the animation (1-300).
    - `animation_x_start`, `animation_y_start`: Starting position for animations.
    - `animation_x_end`, `animation_y_end`: Ending position for animations.

12. **Output Options**:
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
9. **Animated GIFs**: Create animated images with background removal and effects.
10. **VFX Compositing**: Prepare elements for visual effects compositing in film or video production.

### Tips and Best Practices

1. **Model Selection**: 
   - Use `u2net` for general-purpose removal.
   - Try `isnet-anime` for cartoon or anime-style images.
   - Use `bria` or `birefnet` for more complex scenes or when other models fail.
   - Experiment with different models for optimal results.

2. **Complex Edges**:
   - Enable `alpha_matting` for images with hair or fur.
   - Adjust `feather_amount` for smoother transitions.
   - Use `mask_expansion` to fine-tune the mask edge.

3. **Fine-tuning Removal**:
   - Use `chroma_key` in combination with AI removal for challenging backgrounds.
   - Adjust `mask_blur` and `mask_expansion` for refined edges.
   - Experiment with `post_process_mask` for improved results.

4. **Composition**:
   - Use `foreground_scale`, `x_position`, and `y_position` for precise placement.
   - Experiment with `aspect_ratio_preset` for different layouts.
   - Try different `blending_mode` options for creative compositing.

5. **Realistic Integration**:
   - Apply subtle `shadow` effects for grounded compositions.
   - Use `color_adjustment` to match foreground with new backgrounds.
   - Experiment with `edge_detection` for more defined subjects.

6. **Animation**:
   - Use `animation_type` to add movement to your compositions.
   - Adjust `animation_speed` and `animation_frames` for smooth animations.
   - Combine different animation types for complex movements.

7. **Matrix Background**:
   - Use `matrix_foreground_pattern` and `matrix_custom_text` for unique effects.
   - Adjust `matrix_density` and `matrix_fall_speed` for desired visual impact.
   - Enable `matrix_glow` for a more ethereal look.

8. **Batch Processing**:
   - Ensure consistent image sizes for optimal performance.
   - Use `background_loop_mode` for creative batch outputs.
   - Utilize animation settings for batch-created GIFs or video frames.

9. **Performance**:
   - Disable `alpha_matting` for faster processing when not needed.
   - Use lower resolution images for rapid prototyping, then switch to high-res for final output.
   - Consider using GPU acceleration for faster processing of large batches or high-res images.

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
       # Implementation...
   ```
   This function uses OpenCV to create a mask based on the specified color range. It's efficient for removing solid color backgrounds.

2. **process_mask**:
   ```python
   def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
       # Implementation...
   ```
   Combines multiple mask operations (inversion, feathering, blurring, expansion) into a single efficient function.

3. **apply_blending_mode**:
   ```python
   def apply_blending_mode(self, bg, fg, mode, strength):
       # Implementation...
   ```
   Implements various blending modes using NumPy operations, allowing for creative compositing effects.

4. **apply_color_adjustments**:
   ```python
   def apply_color_adjustments(self, image, brightness, contrast, saturation, hue, sharpness):
       # Implementation...
   ```
   Uses PIL's `ImageEnhance` for efficient color adjustments.

### New Features and Functions

1. **create_matrix_background**:
   ```python
   def create_matrix_background(self, width, height, foreground_pattern, custom_text, density, fall_speed, glow, glow_intensity):
       # Implementation...
   ```
   This function generates a Matrix-style rain effect background. It uses PIL to create an image with falling characters, supporting custom patterns and glow effects.

2. **animate_element**:
   ```python
   def animate_element(self, element, element_mask, animation_type, animation_speed, frame_number, total_frames,
                       x_start, y_start, x_end, y_end, canvas_width, canvas_height):
       # Implementation...
   ```
   Handles various animation types including bounce, travel, rotate, fade, and zoom effects. It calculates the position and transformation of an element for each frame of the animation.

3. **apply_filter**:
   ```python
   def apply_filter(self, image, filter_type, strength):
       # Implementation...
   ```
   Applies various image filters including blur, sharpen, edge enhance, emboss, toon, sepia, film grain, and matrix effects. Each filter is implemented as a separate method for modularity.

4. **parse_aspect_ratio**:
   ```python
   def parse_aspect_ratio(self, aspect_ratio_preset, custom_aspect_ratio):
       # Implementation...
   ```
   Parses and calculates the aspect ratio based on presets or custom input, allowing for flexible image sizing.

### Performance Considerations

1. **GPU Acceleration**: 
   ```python
   self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
   ```
   Automatically detects and uses GPU when available for faster processing, especially beneficial for AI-based background removal models.

2. **Batch Processing**:
   ```python
   for frame_number in tqdm(range(animation_frames), desc="Processing frames"):
       # Process each frame...
   ```
   Efficiently handles multiple frames or images, using tqdm for progress tracking.

3. **Numpy Operations**:
   Extensive use of NumPy for fast array operations, e.g., in blending modes and mask processing, significantly speeding up image manipulations.

4. **Caching Mechanisms**:
   ```python
   if self.session is None or self.session.model_name != model:
       # Initialize session...
   ```
   Caches AI model sessions to avoid reloading models for each operation, greatly improving performance for repeated uses.

### Extensibility

1. **Model Support**:
   ```python
   if model == 'u2net_custom' and custom_model_path:
       self.session = new_session('u2net_custom', model_path=custom_model_path, providers=providers)
   ```
   Allows for easy addition of custom models. Developers can add new model types by extending the model selection logic and implementing the corresponding session initialization.

2. **Blending Modes**:
   ```python
   class BlendingMode(Enum):
       # Blending mode definitions...
   ```
   New blending modes can be easily added by extending this enum and implementing the corresponding logic in `apply_blending_mode`.

3. **Filters**:
   ```python
   def apply_filter(self, image, filter_type, strength):
       # Implementation...
   ```

### Filters (continued)

```python
def apply_filter(self, image, filter_type, strength):
    if filter_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=strength * 2))
    elif filter_type == "sharpen":
        percent = int(100 + (strength * 100))
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))
    elif filter_type == "edge_enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_type == "emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type == "toon":
        return self.apply_toon_filter(image, strength)
    elif filter_type == "sepia":
        return self.apply_sepia_filter(image)
    elif filter_type == "film_grain":
        return self.apply_film_grain(image, strength)
    elif filter_type == "matrix":
        return self.apply_matrix_filter(image, strength)
    else:
        return image
```

The `apply_filter` function is designed to be easily extensible. To add a new filter:

1. Add a new `elif` condition for your filter type.
2. Implement the filter logic, either directly in this function or by calling a separate method.
3. Update the `INPUT_TYPES` dictionary to include the new filter option.

For example, to add a new "vintage" filter:

```python
elif filter_type == "vintage":
    return self.apply_vintage_filter(image, strength)

# Then implement the new method:
def apply_vintage_filter(self, image, strength):
    # Implement vintage filter logic here
    pass

# Update INPUT_TYPES:
"filter": (["none", "blur", ..., "vintage"],),
```

### Animation Types

```python
class AnimationType(Enum):
    NONE = "none"
    BOUNCE = "bounce"
    TRAVEL_LEFT = "travel_left"
    TRAVEL_RIGHT = "travel_right"
    ROTATE = "rotate"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
```

To add a new animation type:

1. Add a new entry to the `AnimationType` enum.
2. Implement the animation logic in the `animate_element` method.
3. Update the `INPUT_TYPES` dictionary to include the new animation option.

For example, to add a new "spiral" animation:

```python
class AnimationType(Enum):
    # Existing types...
    SPIRAL = "spiral"

# In animate_element method:
elif animation_type == AnimationType.SPIRAL.value:
    # Implement spiral animation logic
    pass

# Update INPUT_TYPES:
"animation_type": ([anim.value for anim in AnimationType],),
```

### Matrix Background Generation

```python
def create_matrix_background(self, width, height, foreground_pattern, custom_text, density, fall_speed, glow, glow_intensity):
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Font selection logic...
    
    if foreground_pattern == "BINARY":
        chars = "01"
    elif foreground_pattern == "RANDOM":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
    else:  # CUSTOM
        chars = custom_text
    
    drops = [0 for _ in range(width // 10)]
    for _ in range(int(height * density / 10)):
        for i in range(len(drops)):
            if random.random() < 0.1:
                x = i * 10
                y = drops[i] * 15
                char = random.choice(chars)
                color = (0, 255, 0)
                draw.text((x, y), char, font=font, fill=color)
                
                if glow == "ENABLED":
                    # Glow effect implementation...
                
                drops[i] += fall_speed
                if drops[i] * 15 > height:
                    drops[i] = 0
    
    return image
```

To extend the matrix background generation:

1. Add new parameters to the `create_matrix_background` method signature.
2. Implement the new features within the method.
3. Update the `INPUT_TYPES` dictionary to include new options for matrix customization.

For example, to add color customization:

```python
def create_matrix_background(self, ..., matrix_color):
    # Existing code...
    color = tuple(int(matrix_color[i:i+2], 16) for i in (1, 3, 5))  # Convert hex to RGB
    draw.text((x, y), char, font=font, fill=color)
    # Rest of the implementation...

# Update INPUT_TYPES:
"matrix_color": ("COLOR", {"default": "#00FF00"}),
```

### Performance Optimizations

1. **Numpy Operations**: 
   The code extensively uses NumPy for array operations, which is crucial for performance. For example, in the `apply_blending_mode` method:

   ```python
   def apply_blending_mode(self, bg, fg, mode, strength):
       bg_np = np.array(bg).astype(np.float32) / 255.0
       fg_np = np.array(fg).astype(np.float32) / 255.0
       # Blending operations...
   ```

   This approach allows for fast, vectorized operations on entire images.

2. **GPU Acceleration**:
   The code checks for GPU availability and uses it when possible:

   ```python
   self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
   ```

   To further optimize GPU usage, consider implementing batch processing for GPU operations, especially for the AI-based background removal models.

3. **Caching**:
   The code implements basic caching for AI model sessions:

   ```python
   if self.session is None or self.session.model_name != model:
       # Initialize session...
   ```

   Consider expanding this caching mechanism to other computationally expensive operations, such as preprocessed images or frequently used masks.

### Extensibility Best Practices

1. **Modular Design**: Keep functions focused on single tasks. This makes the code easier to understand, test, and extend.

2. **Configuration-Driven**: Use configuration dictionaries or files to define parameters that might change frequently, such as filter options or animation types.

3. **Abstract Base Classes**: Consider creating abstract base classes for major components (e.g., filters, animations) to enforce a consistent interface for all implementations.

4. **Plugin Architecture**: Implement a plugin system that allows users to add their own filters, animations, or background removal models without modifying the core code.

5. **Event System**: Implement an event system that allows plugins or external code to hook into different stages of the image processing pipeline.

### Testing and Validation

To ensure the reliability and correctness of the GeekyRemB node:

1. Implement unit tests for individual functions, especially for critical operations like mask processing and blending modes.

2. Create integration tests that process sample images through the entire pipeline and compare the results to expected outputs.

3. Implement property-based testing for operations that should maintain certain invariants (e.g., mask operations should always produce valid masks).

4. Add performance benchmarks to track the efficiency of key operations and ensure that optimizations actually improve performance.

### Documentation

Maintain comprehensive documentation:

1. Include detailed docstrings for all methods, explaining parameters, return values, and any side effects.

2. Provide usage examples for common scenarios and edge cases.

3. Create a developer guide that explains the overall architecture, key concepts, and how to extend different parts of the system.

4. Use type hints throughout the code to improve readability and enable static type checking.

By following these practices and continually refining the code structure, the GeekyRemB node can remain flexible, performant, and easy to extend as new features and use cases arise.



Inspired by tools found in [WAS Node Suite for ComfyUI](https://github.com/WASasquatch/was-node-suite-comfyui.git)




<img width="725" alt="Screenshot 2024-07-18 182336" src="https://github.com/user-attachments/assets/dff53dd1-ff4f-48b2-8a96-5f8443cac251">

<img width="688" alt="Screenshot 2024-07-18 191152" src="https://github.com/user-attachments/assets/48281466-9dd7-4dcd-8f1c-3e2bbb69f114">


<img width="600" alt="Screenshot 2024-06-29 134500" src="https://github.com/GeekyGhost/ComfyUI-GeekyRemB/assets/111990299/b09a1833-8bdb-43ba-95db-da6f520e8411">

Output examples can be found here - https://civitai.com/models/546180/geeky-remb





