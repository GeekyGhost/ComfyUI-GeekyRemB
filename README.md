# GeekyRemB: Advanced Background Removal & Image Processing Node Suite for ComfyUI

GeekyRemB is a sophisticated image processing node suite that brings professional-grade background removal, blending, animation capabilities, and lighting effects to ComfyUI. It combines AI-powered processing with traditional image manipulation techniques to offer a comprehensive solution for complex image processing tasks.

<img width="1139" alt="Screenshot 2025-03-12 070830" src="https://github.com/user-attachments/assets/dc6dd79c-b661-4c6c-91fc-af2780a4a967" />


## Table of Contents
1. [Installation](#installation)
2. [Node Overview](#node-overview)
3. [Core Node: Geeky RemB](#core-node-geeky-remb)
4. [Animation Node: Geeky RemB Animator](#animation-node-geeky-remb-animator)
5. [Lighting & Shadow Node: Geeky RemB Light & Shadow](#lighting--shadow-node-geeky-remb-light--shadow)
6. [Keyframe Position Node: Geeky RemB Keyframe Position](#keyframe-position-node-geeky-remb-keyframe-position)
7. [Workflow Examples](#workflow-examples)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

---

## Installation

1. **Install ComfyUI** if you haven't already. Follow the [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI) for detailed instructions.

2. **Clone the GeekyRemB repository** into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/GeekyGhost/ComfyUI-GeekyRemB.git
   ```

3. **Install dependencies**:
   ```bash
   cd ComfyUI-GeekyRemB
   pip install -r requirements.txt
   ```

4. **Restart ComfyUI** to load the new nodes.

---

## Node Overview

GeekyRemB consists of four interconnected nodes that work together to provide a complete image processing system:

1. **Geeky RemB** - The core node handling background removal, mask processing, and composition
2. **Geeky RemB Animator** - Provides animation parameters for creating dynamic sequences
3. **Geeky RemB Light & Shadow** - Controls lighting effects and shadow generation
4. **Geeky RemB Keyframe Position** - Enables precise control through keyframe-based animation

These nodes can be used independently or connected together for advanced workflows.

---

## Core Node: Geeky RemB

The main node responsible for background removal, image processing, and composition.

### Key Features

- **AI-Powered Background Removal**
  - Multiple rembg models optimized for different subjects
  - Professional chroma keying with color selection and tolerance control
  
- **Advanced Mask Processing**
  - Mask expansion/contraction
  - Edge detection and refinement
  - Blur and threshold controls
  - Small region removal
  
- **Image Composition**
  - Position control for foreground elements
  - Aspect ratio and scaling options
  - Alpha channel management
  
- **Animation Support**
  - Multi-frame processing
  - Integration with animation and lighting nodes

### Usage Instructions

1. **Basic Background Removal**:
   - Connect your input image to the `foreground` input
   - Set `enable_background_removal` to true
   - Choose your removal method (`rembg` or `chroma_key`)
   - For `rembg`, select an appropriate model for your subject
   - For `chroma_key`, select the key color and adjust tolerance
   
2. **Mask Refinement**:
   - Use `mask_expansion` to grow or shrink the mask (positive values expand, negative contract)
   - Enable `edge_detection` and adjust `edge_thickness` for sharper outlines
   - Use `mask_blur` to smooth edges
   - Enable `remove_small_regions` and set `small_region_size` to clean up noise
   
3. **Advanced Composition**:
   - Connect a background image to the `background` input
   - Set `x_position` and `y_position` to place the foreground
   - Adjust `scale` to resize the foreground
   - Specify `aspect_ratio` for consistent dimensions
   
4. **Using Alpha Matting**:
   - Enable `alpha_matting` for high-quality edge refinement
   - Adjust `alpha_matting_foreground_threshold` and `alpha_matting_background_threshold` for fine control
   - Useful for subjects with fine details like hair or fur

5. **Animation Setup**:
   - Set `frames` to the desired number of output frames
   - Connect an animator node to the `animator` input
   - Connect a light & shadow node to the `lightshadow` input
   - Connect keyframe nodes to the `keyframe` input for precise control

### Parameters Guide

#### Essential Parameters
- **output_format**: Choose between RGBA (with transparency) or RGB output
- **foreground**: The input image to process
- **enable_background_removal**: Toggle background removal processing
- **removal_method**: Choose between AI-based (`rembg`) or color-based (`chroma_key`)
- **model**: Select AI model for the `rembg` method
- **chroma_key_color**: Select the color to key out (`green`, `blue`, or `red`)
- **chroma_key_tolerance**: Adjust sensitivity of color keying (0.0-1.0)
- **spill_reduction**: Remove color spill from edges (0.0-1.0)

#### Mask Processing Parameters
- **mask_expansion**: Expand or contract the mask (-100 to 100)
- **edge_detection**: Enable edge detection for sharper outlines
- **edge_thickness**: Control thickness of detected edges (1-10)
- **mask_blur**: Apply blur to mask edges (0-100)
- **threshold**: Set the threshold for mask generation (0.0-1.0)
- **invert_generated_mask**: Invert the mask (switch foreground/background)
- **remove_small_regions**: Remove small artifacts from the mask
- **small_region_size**: Set minimum size of regions to keep (1-1000)

#### Composition Parameters
- **aspect_ratio**: Set output aspect ratio (e.g., "16:9", "4:3", "1:1", "portrait", "landscape")
- **scale**: Scale the foreground (0.1-5.0)
- **frames**: Number of frames to generate (1-1000)
- **x_position**: Horizontal position of foreground (-2048 to 2048)
- **y_position**: Vertical position of foreground (-2048 to 2048)

#### Advanced Parameters
- **alpha_matting**: Enable advanced edge refinement
- **alpha_matting_foreground_threshold**: Threshold for foreground detection (0-255)
- **alpha_matting_background_threshold**: Threshold for background detection (0-255)
- **edge_refinement**: Enable additional edge refinement for chroma key

#### Optional Inputs
- **background**: Secondary image for background composition
- **additional_mask**: Extra mask for complex selections
- **invert_additional_mask**: Invert the additional mask
- **animator**: Connection to GeekyRemB_Animator node
- **lightshadow**: Connection to GeekyRemB_LightShadow node
- **keyframe**: Connection to GeekyRemB_KeyframePosition node

---

## Animation Node: Geeky RemB Animator

Provides animation parameters to the main GeekyRemB node, controlling movement, transitions, and timing.

### Key Features

- **Multiple Animation Types**:
  - Movement animations (bounce, travel, slide)
  - Transform animations (scale, rotate, flip)
  - Opacity animations (fade in/out)
  - Special effects (shake, wave, pulse)
  
- **Animation Control**:
  - Speed and duration settings
  - Repeat and reverse options
  - Easing functions for natural motion
  
- **Keyframe Support**:
  - Integration with keyframe position nodes
  - Frame-accurate positioning

### Usage Instructions

1. **Basic Animation Setup**:
   - Add the Geeky RemB Animator node to your workflow
   - Select the desired `animation_type`
   - Set `animation_speed` and `animation_duration`
   - Connect the node's output to the main GeekyRemB node's `animator` input
   
2. **Animation Timing**:
   - Set `fps` to control the smoothness of animation
   - Adjust `repeats` to loop the animation
   - Enable `reverse` to alternate direction on repeats
   - Set `delay` to postpone the start of animation
   
3. **Motion Control**:
   - Set the initial position with `x_position` and `y_position`
   - Adjust `scale` and `rotation` for the starting state
   - Use `steps` to create multi-step animations
   - Set `phase_shift` for staggered animations
   
4. **Keyframe Animation**:
   - Enable `use_keyframes` to use keyframe-based animation
   - Connect keyframe position nodes to the `keyframe1` through `keyframe5` inputs
   - Set `easing_function` to control the interpolation between keyframes

### Parameters Guide

#### Animation Type
- **animation_type**: Choose from various animation types:
  - `none`: No animation
  - `bounce`: Up and down movement
  - `travel_left`/`travel_right`: Horizontal movement
  - `rotate`: Continuous rotation
  - `fade_in`/`fade_out`: Opacity transitions
  - `zoom_in`/`zoom_out`: Scale transitions
  - `scale_bounce`: Pulsing size changes
  - `spiral`: Combined rotation and movement
  - `shake`: Quick oscillating movements
  - `slide_up`/`slide_down`: Vertical movement
  - `flip_horizontal`/`flip_vertical`: Mirroring effects
  - `wave`: Sinusoidal movement
  - `pulse`: Periodic scaling
  - `swing`: Pendulum-like rotation
  - `spin`: Continuous spinning
  - `flash`: Brightness fluctuation

#### Timing Controls
- **animation_speed**: Rate of animation (0.1-10.0)
- **animation_duration**: Length of one animation cycle (0.1-10.0)
- **repeats**: Number of times to repeat the animation (1-10)
- **reverse**: Toggle direction reversal on repeats
- **fps**: Frames per second (1-120)
- **delay**: Wait time before animation starts (0.0-5.0)

#### Motion Parameters
- **x_position**: Initial horizontal position (-1000 to 1000)
- **y_position**: Initial vertical position (-1000 to 1000)
- **scale**: Initial scale factor (0.1-5.0)
- **rotation**: Initial rotation angle (-360.0 to 360.0)
- **steps**: Number of steps in multi-step animations (1-10)
- **phase_shift**: Phase shift for staggered animations (0.0-1.0)

#### Easing Functions
- **easing_function**: Controls how animations accelerate/decelerate:
  - `linear`: Constant speed
  - `ease_in_quad`/`ease_out_quad`/`ease_in_out_quad`: Quadratic easing
  - `ease_in_cubic`/`ease_out_cubic`/`ease_in_out_cubic`: Cubic easing
  - `ease_in_sine`/`ease_out_sine`/`ease_in_out_sine`: Sinusoidal easing
  - `ease_in_expo`/`ease_out_expo`/`ease_in_out_expo`: Exponential easing
  - `ease_in_bounce`/`ease_out_bounce`/`ease_in_out_bounce`: Bouncy easing

#### Keyframe Controls
- **use_keyframes**: Enable keyframe-based animation
- **keyframe1** through **keyframe5**: Connect to keyframe position nodes

---

## Lighting & Shadow Node: Geeky RemB Light & Shadow

Controls lighting effects and shadow generation for the processed images.

### Key Features

- **Realistic Lighting Effects**:
  - Directional lighting with intensity control
  - Light color and falloff adjustment
  - Normal mapping for 3D-like lighting
  - Specular highlights for reflective surfaces
  
- **Dynamic Shadow Generation**:
  - Shadow opacity and blur control
  - Shadow direction and color customization
  - Perspective shadows for realism
  - Distance-based shadow fading

### Usage Instructions

1. **Basic Lighting Setup**:
   - Add the Geeky RemB Light & Shadow node to your workflow
   - Enable `enable_lighting` to activate lighting effects
   - Adjust `light_intensity` to control strength
   - Set `light_direction_x` and `light_direction_y` to position the light source
   - Connect the node's output to the main GeekyRemB node's `lightshadow` input
   
2. **Light Customization**:
   - Choose between RGB color control or Kelvin temperature
   - Enable `use_kelvin_temperature` and set `kelvin_temperature` for natural lighting
   - Or adjust `light_color_r`, `light_color_g`, and `light_color_b` for custom colors
   - Set `light_radius` and `light_falloff` to control illumination area
   
3. **Advanced Lighting**:
   - Enable `enable_normal_mapping` for 3D-like lighting effects
   - Turn on `enable_specular` and adjust `specular_intensity` for highlights
   - Set `specular_shininess` to control highlight sharpness
   - Adjust `ambient_light` for global illumination level
   
4. **Shadow Configuration**:
   - Enable `enable_shadow` to activate shadow generation
   - Set `shadow_opacity` and `shadow_blur` for shadow appearance
   - Adjust `shadow_direction_x` and `shadow_direction_y` for shadow placement
   - Customize `shadow_color_r`, `shadow_color_g`, and `shadow_color_b`
   
5. **Realistic Shadows**:
   - Enable `perspective_shadow` for distance-based perspective effects
   - Set `light_source_height` to control shadow length
   - Turn on `distance_fade` and adjust `fade_distance` for natural fading
   - Toggle `soft_edges` for realistic shadow edges

### Parameters Guide

#### Lighting Controls
- **enable_lighting**: Toggle lighting effects
- **light_intensity**: Strength of lighting effect (0.0-1.0)
- **light_direction_x**: Horizontal light direction (-200 to 200)
- **light_direction_y**: Vertical light direction (-200 to 200)
- **light_radius**: Area of light effect (10-500)
- **light_falloff**: Rate of light falloff (0.1-3.0)
- **light_from_behind**: Toggle backlighting effect

#### Light Color
- **use_kelvin_temperature**: Use color temperature instead of RGB
- **kelvin_temperature**: Light color temperature (2000-10000K)
- **light_color_r/g/b**: RGB components of light color (0-255)

#### Advanced Lighting
- **enable_normal_mapping**: Enable 3D-like lighting effects
- **enable_specular**: Add specular highlights
- **specular_intensity**: Strength of highlights (0.0-1.0)
- **specular_shininess**: Sharpness of highlights (1-128)
- **ambient_light**: Global illumination level (0.0-1.0)
- **light_source_height**: Height of light source (50-500)

#### Shadow Controls
- **enable_shadow**: Toggle shadow generation
- **shadow_opacity**: Shadow transparency (0.0-1.0)
- **shadow_blur**: Shadow edge softness (0-50)
- **shadow_direction_x**: Horizontal shadow offset (-50 to 50)
- **shadow_direction_y**: Vertical shadow offset (-50 to 50)
- **shadow_expansion**: Shadow size adjustment (-10 to 20)
- **shadow_color_r/g/b**: RGB components of shadow color (0-255)

#### Advanced Shadow
- **perspective_shadow**: Enable perspective-based shadows
- **distance_fade**: Fade shadow with distance
- **fade_distance**: Distance at which shadow begins to fade (10-500)
- **soft_edges**: Toggle soft shadow edges

---

## Keyframe Position Node: Geeky RemB Keyframe Position

Provides precise control over animation through keyframe-based positioning.

### Key Features

- **Frame-Specific Controls**:
  - Position, scale, and rotation settings for specific frames
  - Opacity control for visibility transitions
  - Easing function selection for smooth interpolation

### Usage Instructions

1. **Creating Keyframes**:
   - Add the Geeky RemB Keyframe Position node to your workflow
   - Set `frame_number` to the target frame
   - Adjust `x_position` and `y_position` for placement
   - Set `scale` and `rotation` as needed
   - Connect multiple keyframe nodes to the Animator's keyframe inputs
   
2. **Keyframe Configuration**:
   - Define the canvas size with `width` and `height`
   - Set `opacity` for transparency control
   - Select an `easing` function for interpolation between keyframes
   
3. **Building Keyframe Sequences**:
   - Create multiple keyframe nodes with different frame numbers
   - Connect them to consecutive keyframe inputs on the Animator node
   - Enable `use_keyframes` on the Animator node

### Parameters Guide

#### Canvas Settings
- **width**: Width of the animation canvas (64-4096)
- **height**: Height of the animation canvas (64-4096)

#### Keyframe Controls
- **frame_number**: Target frame for this keyframe (0-1000)
- **x_position**: Horizontal position at this keyframe (-2048 to 2048)
- **y_position**: Vertical position at this keyframe (-2048 to 2048)
- **scale**: Scale factor at this keyframe (0.1-5.0)
- **rotation**: Rotation angle at this keyframe (-360.0 to 360.0)
- **opacity**: Transparency at this keyframe (0.0-1.0)

#### Interpolation
- **easing**: Easing function for interpolation to the next keyframe
  - Options match the easing functions available in the Animator node

---

## Workflow Examples

### Basic Background Removal

1. Connect an image source to Geeky RemB's `foreground` input
2. Set `enable_background_removal` to true
3. Choose `rembg` as the removal method
4. Select an appropriate model (e.g., `u2net` for general purposes, `isnet-anime` for anime images)
5. Adjust mask processing parameters as needed
6. Connect the output to your workflow

### Animated Character with Lighting

1. Add a Geeky RemB Animator node
   - Set `animation_type` to `bounce`
   - Set `animation_speed` to 1.0
   - Set `repeats` to 2
   - Set `easing_function` to `ease_in_out_sine`

2. Add a Geeky RemB Light & Shadow node
   - Enable `enable_lighting` and `enable_shadow`
   - Set `light_direction_x` to -50 and `light_direction_y` to -100
   - Set `shadow_opacity` to 0.4
   - Set `shadow_blur` to 15

3. Connect both to a Geeky RemB node
   - Connect your character image to `foreground`
   - Connect a background image to `background`
   - Set `frames` to 30
   - Set `enable_background_removal` to true

### Keyframe Animation Sequence

1. Create multiple Geeky RemB Keyframe Position nodes:
   - Keyframe 1: `frame_number`: 0, `x_position`: 0, `y_position`: 0
   - Keyframe 2: `frame_number`: 15, `x_position`: 200, `y_position`: -50, `rotation`: 45
   - Keyframe 3: `frame_number`: 30, `x_position`: 400, `y_position`: 0, `rotation`: 0

2. Add a Geeky RemB Animator node:
   - Set `use_keyframes` to true
   - Connect the keyframe nodes to `keyframe1`, `keyframe2`, and `keyframe3`
   - Set `fps` to 30

3. Connect to a Geeky RemB node:
   - Set `frames` to 30
   - Set other parameters as needed

---

## Advanced Features

### Multi-frame Processing with Thread Pooling

GeekyRemB optimizes performance by using thread pools to process multiple frames in parallel. This makes it efficient for handling animations and batch processing.

### Sophisticated Caching System

An LRU (Least Recently Used) cache system is implemented to store and reuse processed frames, reducing redundant computations and improving performance.

### Edge Refinement Techniques

Multiple edge processing methods are available, including alpha matting, edge detection, and mask refinement, enabling high-quality results even with complex subjects.

### Perspective Shadow Generation

The shadow system can create realistic perspective-based shadows that simulate the effect of a 3D light source, adding depth to compositions.

### Normal Mapping for 3D-like Lighting

Advanced lighting effects include normal mapping, which simulates surface details for more realistic illumination without requiring actual 3D models.

---

## Troubleshooting

### Common Issues and Solutions

1. **Slow Background Removal**
   - Try using a lighter model like `u2netp` instead of `u2net`
   - Reduce the image size before processing
   - Ensure GPU acceleration is available and enabled

2. **Poor Edge Quality**
   - Enable `alpha_matting` for better edge refinement
   - Adjust `mask_blur` for smoother edges
   - Try different `mask_expansion` values

3. **Memory Issues with Animation**
   - Reduce the number of frames
   - Lower the resolution of input images
   - Close other memory-intensive applications

4. **Missing Shadow or Light Effects**
   - Verify that both `enable_lighting` and `enable_shadow` are turned on
   - Check that the Light & Shadow node is connected to the main node
   - Adjust direction values to ensure effects are visible

5. **Keyframes Not Working**
   - Confirm that `use_keyframes` is enabled on the Animator node
   - Check that keyframe nodes are connected in the correct order
   - Verify that frame numbers are set correctly and within range

### Performance Optimization

- Use the appropriate rembg model for your needs - lighter models like `u2netp` are faster
- For batch processing, set a reasonable number of frames to avoid memory issues
- Adjust thread count if needed by modifying the `max_workers` value in the code
- Pre-process images to reduce resolution before applying effects

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Note**: Some included models may have separate licensing requirements for commercial use.

---

## Acknowledgements

GeekyRemB builds upon several outstanding open-source projects:

- [Rembg](https://github.com/danielgatis/rembg) by Daniel Gatis: Core background removal capabilities
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): The foundation of our node system
- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui): Inspiration for layer utility features

Special thanks to:

- **The ComfyUI Community**: For valuable feedback and suggestions
- **Open-Source Contributors**: Who help improve the nodes continuously
- **AI Model Creators**: Whose work enables our advanced background removal features

---

For updates, issues, or contributions, please visit the [GitHub repository](https://github.com/GeekyGhost/ComfyUI-GeekyRemB). We welcome feedback and contributions from the community.
