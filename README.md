# GeekyRemB v4.0: Ultimate AI Background Removal & Video Layering Node for ComfyUI

GeekyRemB v4.0 is a revolutionary image and video processing node suite that brings state-of-the-art AI models, professional compositing, advanced animation, and cutting-edge multimodal capabilities to ComfyUI. Built on the latest AI architectures including RMBG v2.0 with BiRefNet, it represents the pinnacle of background removal and video layering technology.

<img width="3072" height="2195" alt="workflow (7)" src="https://github.com/user-attachments/assets/f3e63915-4355-4b34-9f1b-2568c921e7ae" />


## 🚀 What's New in v4.0

- **Latest AI Models**: RMBG v2.0 with BiRefNet architecture for superior background removal
- **Advanced Multimodal Processing**: Cross-modal analysis and generation capabilities
- **Professional Animation System**: 20+ animation types with advanced easing functions
- **Physics-Aware Generation**: Realistic simulation for next-generation content
- **Professional Blending Modes**: 15+ industry-standard compositing modes
- **Self-Optimizing Workflows**: AI-powered workflow optimization and generation
- **GPU Acceleration**: Optimized performance with intelligent memory management
- **Video Batch Processing**: Frame-consistent processing for video sequences

## Table of Contents
1. [Installation](#installation)
2. [Core Features](#core-features)
3. [AI Models & Methods](#ai-models--methods)
4. [Professional Animation System](#professional-animation-system)
5. [Advanced Positioning & Blending](#advanced-positioning--blending)
6. [Multimodal Processing](#multimodal-processing)
7. [Workflow Examples](#workflow-examples)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- ComfyUI (latest version recommended)
- Python 3.8+ 
- PyTorch 2.0+
- CUDA-capable GPU (recommended for optimal performance)

### Quick Installation

1. **Clone the repository**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/GeekyGhost/ComfyUI-GeekyRemB.git
   cd ComfyUI-GeekyRemB
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Acceleration Setup** (Optional but recommended):
   ```bash
   # For NVIDIA GPUs
   pip install onnxruntime-gpu>=1.16.0
   
   # For AMD ROCm
   pip install torch-audio-rocm>=2.0.0
   ```

4. **Restart ComfyUI** to load the new nodes.

### Advanced Installation Options

For development or advanced users:
```bash
# Create virtual environment
python -m venv geeky_remb_env
source geeky_remb_env/bin/activate  # Linux/Mac
# or: geeky_remb_env\Scripts\activate  # Windows

# Install with optional features
pip install -r requirements.txt
pip install matplotlib imageio ffmpeg-python  # Optional enhancements
```

---

## Core Features

### 🎯 State-of-the-Art Background Removal

**Supported AI Models:**
- **RMBG v2.0 (BiRefNet)** - Latest architecture for superior quality
- **RMBG v1.4 (BRIA AI)** - Professional-grade removal
- **BiRefNet Direct** - High-performance segmentation
- **Advanced Chroma Key** - Professional green/blue screen with spill suppression
- **Hybrid AI+Chroma** - Combines AI and chroma key for optimal results

**Advanced Features:**
- Adaptive model selection based on content analysis
- Real-time edge refinement with alpha matting
- Intelligent mask processing with small object removal
- Professional spill suppression and edge feathering
- Batch processing with frame consistency

### 🎬 Professional Video Processing

- **Multi-frame Processing**: Optimized for video sequences with frame consistency
- **Thread Pool Optimization**: Parallel processing for maximum performance  
- **Advanced Caching**: LRU cache system reduces redundant computations
- **Memory Management**: Intelligent VRAM handling from 1GB to 64GB+ systems
- **Export Options**: Multiple video formats with quality optimization

### 🎨 Advanced Compositing Engine

- **15+ Blend Modes**: Industry-standard modes including Multiply, Screen, Overlay, Soft Light
- **Professional Positioning**: 11 positioning modes from absolute pixels to custom anchors
- **Transform Controls**: Scale, rotation, flip with sub-pixel precision
- **Layer Effects**: Drop shadows, perspective shadows, lighting effects
- **Alpha Channel Management**: Sophisticated transparency handling

---

## AI Models & Methods

### Model Selection Guide

| Model | Best For | Performance | Quality | GPU Memory |
|-------|----------|-------------|---------|------------|
| RMBG v2.0 (BiRefNet) | General use, highest quality | Medium | Excellent | 4-8GB |
| RMBG v1.4 (BRIA AI) | Professional workflows | Fast | Very Good | 2-4GB |
| BiRefNet Direct | Complex subjects, fine details | Slow | Excellent | 6-12GB |
| Advanced Chroma Key | Studio environments | Very Fast | Good | <1GB |
| Hybrid AI+Chroma | Mixed environments | Medium | Excellent | 4-6GB |

### Configuration Examples

```python
# High-quality portrait processing
removal_config = BackgroundRemovalConfig(
    method=RemovalMethod.RMBG_V2,
    model_precision="fp16",
    processing_resolution=1024,
    edge_feathering=3.0,
    remove_small_objects=True,
    anti_aliasing=True
)

# Speed-optimized processing  
removal_config = BackgroundRemovalConfig(
    method=RemovalMethod.RMBG_V1_4,
    processing_resolution=512,
    batch_size=4,
    use_gpu_acceleration=True
)

# Professional chroma key
removal_config = BackgroundRemovalConfig(
    method=RemovalMethod.CHROMA_KEY,
    chroma_color="green",
    tolerance=0.15,
    spill_suppression=0.9,
    edge_feathering=2.0
)
```

---

## Professional Animation System

### 20+ Animation Types

**Movement Animations:**
- `slide_left`, `slide_right`, `slide_up`, `slide_down`, `slide_diagonal`
- `orbit_circular`, `spiral_in`, `bounce_in`, `bounce_out`

**Transform Animations:**
- `scale_in`, `scale_out`, `scale_pulse`
- `rotate_clockwise`, `rotate_counter_clockwise`
- `flip_horizontal`, `flip_vertical`

**Special Effects:**
- `fade_in`, `fade_out`, `fade_pulse`
- `elastic_in`, `elastic_out`
- `shake_subtle`, `wobble`

### Advanced Easing Functions

Professional easing for natural motion:
- **Linear**: Constant speed movement
- **Quadratic**: `ease_in_quad`, `ease_out_quad`, `ease_in_out_quad`
- **Cubic**: Smooth acceleration/deceleration
- **Sine**: Natural sinusoidal motion
- **Exponential**: Dramatic speed changes
- **Elastic**: Spring-like bouncing effects
- **Bounce**: Realistic bouncing motion
- **Back**: Overshoot and settle effects

### Animation Configuration

```python
# Smooth bounce animation
animation_config = AnimationConfig(
    animation_type=AnimationType.BOUNCE_IN,
    duration=2.0,
    easing=EasingFunction.EASE_OUT_BOUNCE,
    amplitude=150.0,
    frequency=1.0,
    loop_count=3,
    ping_pong=True
)

# Complex orbital motion
animation_config = AnimationConfig(
    animation_type=AnimationType.ORBIT_CIRCULAR,
    duration=4.0,
    easing=EasingFunction.SINE_IN_OUT,
    amplitude=200.0,
    frequency=2.0,
    start_delay=0.5
)
```

---

## Advanced Positioning & Blending

### 11 Positioning Modes

1. **Absolute Pixels** - Precise pixel positioning
2. **Relative Percent** - Percentage-based positioning
3. **Center** - Automatic centering
4. **Corner Positions** - Top/bottom + left/center/right combinations
5. **Custom Anchor** - User-defined anchor points with offset

### Professional Blend Modes

| Blend Mode | Effect | Best For |
|------------|--------|----------|
| **Normal** | Standard overlay | General compositing |
| **Multiply** | Darkening effect | Creating shadows, depth |
| **Screen** | Lightening effect | Adding highlights, glow |
| **Overlay** | Contrast enhancement | Dramatic effects |
| **Soft Light** | Subtle lighting | Natural lighting effects |
| **Hard Light** | Strong contrast | Bold, dramatic looks |
| **Color Dodge/Burn** | Extreme lighting | Special effects |
| **Darken/Lighten** | Selective blending | Texture blending |
| **Difference** | Inversion effects | Abstract, artistic looks |

### Advanced Layer Effects

```python
# Professional drop shadow
blend_config = BlendingConfig(
    blend_mode=BlendMode.NORMAL,
    opacity=0.9,
    drop_shadow=True,
    shadow_offset_x=8.0,
    shadow_offset_y=12.0,
    shadow_blur=15.0,
    shadow_opacity=0.6,
    shadow_color=(0, 0, 0)
)

# Cinematic lighting effect
blend_config = BlendingConfig(
    blend_mode=BlendMode.SOFT_LIGHT,
    opacity=0.8,
    preserve_luminosity=True,
    knock_out=False
)
```

---

## Multimodal Processing

### Cross-Modal Analysis

GeekyRemB v4.0 introduces cutting-edge multimodal capabilities:

- **Vision-Audio Synchronization**: Automatic sync detection for video content
- **Text-to-Visual Generation**: AI-powered visual generation from descriptions
- **Cross-Modal Search**: Find content across different modalities
- **Multimodal Reasoning**: Advanced AI reasoning across image, audio, and text

### Usage Examples

```python
# Multimodal content analysis
result = multimodal_processor.process_multimodal(
    processing_mode="analysis",
    image=input_image,
    audio_path="audio_track.wav",
    text_prompt="Analyze the emotional content",
    modality_weights={"vision": 0.4, "audio": 0.4, "text": 0.2}
)

# Cross-modal generation
result = multimodal_processor.process_multimodal(
    processing_mode="generation",
    text_prompt="A serene sunset over calm waters",
    reference_style=style_image,
    quality_target="cinema",
    output_modalities="all"
)
```

---

## Workflow Examples

### 1. Basic Professional Background Removal

```python
# Input: Portrait photo with complex background
# Output: Clean subject with transparent background

workflow = {
    "nodes": {
        "1": {
            "class_type": "GeekyRemB",
            "inputs": {
                "foreground": ["load_image", 0],
                "skip_background_removal": False,
                "removal_method": "rmbg_v2_birefnet",
                "processing_resolution": 1024,
                "mask_blur": 1.5,
                "edge_feathering": 3.0,
                "remove_small_objects": True
            }
        }
    }
}
```

### 2. Cinematic Character Animation

```python
# Create a cinematic character entrance with lighting
workflow = {
    "character_removal": {
        "removal_method": "birefnet_zhengpeng7",
        "quality_target": "cinema",
        "edge_feathering": 2.0
    },
    "animation": {
        "animation_type": "slide_right",
        "duration": 3.0,
        "easing": "ease_out_cubic",
        "amplitude": 400.0
    },
    "lighting": {
        "enable_lighting": True,
        "light_direction_x": -100,
        "light_direction_y": -150,
        "light_intensity": 0.8,
        "enable_shadow": True,
        "shadow_opacity": 0.4
    },
    "composition": {
        "blend_mode": "soft_light",
        "position_mode": "center",
        "scale_x": 1.2,
        "scale_y": 1.2
    }
}
```

### 3. Multi-Subject Video Processing

```python
# Process video with multiple subjects and complex backgrounds
workflow = {
    "batch_processing": {
        "total_frames": 120,
        "video_fps": 24.0,
        "removal_method": "hybrid_ai_chroma",
        "batch_processing": True
    },
    "optimization": {
        "use_gpu_acceleration": True,
        "processing_resolution": 720,
        "thread_count": 8
    },
    "output": {
        "video_format": "mp4",
        "quality": "high",
        "maintain_frame_consistency": True
    }
}
```

### 4. Advanced Keyframe Animation

```python
# Create complex keyframe-based animation sequence
keyframes = [
    {
        "frame": 0,
        "x_position": -200,
        "y_position": 0,
        "scale": 0.8,
        "rotation": 0,
        "opacity": 0.0
    },
    {
        "frame": 30,
        "x_position": 0,
        "y_position": -50,
        "scale": 1.0,
        "rotation": 15,
        "opacity": 1.0
    },
    {
        "frame": 60,
        "x_position": 150,
        "y_position": 0,
        "scale": 1.1,
        "rotation": 0,
        "opacity": 0.9
    },
    {
        "frame": 90,
        "x_position": 0,
        "y_position": 0,
        "scale": 1.0,
        "rotation": 0,
        "opacity": 1.0
    }
]

animation_config = {
    "use_keyframes": True,
    "easing_function": "ease_in_out_cubic",
    "total_frames": 90,
    "fps": 30
}
```

---

## Performance Optimization

### GPU Memory Management

GeekyRemB v4.0 includes intelligent memory management for various GPU configurations:

| GPU Memory | Recommended Settings | Performance |
|------------|---------------------|-------------|
| **1-2GB** | LOW_VRAM mode, 512px resolution | Basic processing |
| **4-6GB** | NORMAL_VRAM mode, 1024px resolution | Good performance |
| **8-12GB** | HIGH_VRAM mode, 1536px resolution | Optimal performance |
| **16GB+** | Maximum settings, 2048px+ resolution | Maximum quality |

### Optimization Settings

```python
# For 4GB GPU
optimization_config = {
    "processing_resolution": 1024,
    "batch_size": 2,
    "use_gpu_acceleration": True,
    "model_precision": "fp16",
    "memory_management": "NORMAL_VRAM"
}

# For 16GB+ GPU  
optimization_config = {
    "processing_resolution": 2048,
    "batch_size": 8,
    "use_gpu_acceleration": True,
    "model_precision": "fp32",
    "memory_management": "HIGH_VRAM",
    "enable_model_caching": True
}

# CPU-optimized (no GPU)
optimization_config = {
    "processing_resolution": 512,
    "batch_size": 1,
    "use_gpu_acceleration": False,
    "thread_count": 8,
    "memory_management": "CPU_ONLY"
}
```

### Performance Tips

1. **Model Selection**: Use lighter models (RMBG v1.4) for real-time applications
2. **Resolution**: Process at lower resolution, then upscale if needed
3. **Batch Processing**: Group similar operations for efficiency
4. **Caching**: Enable model and result caching for repeated operations
5. **Threading**: Adjust thread count based on CPU cores (recommended: CPU cores - 1)

---

## API Reference

### Core Classes

#### `GeekyRemBv4`
The main processing node with comprehensive background removal and compositing capabilities.

**Key Parameters:**
- `removal_method`: Choose AI model or chroma key method
- `processing_resolution`: Balance quality vs. performance
- `animation_type`: Select from 20+ animation types
- `blend_mode`: Professional compositing modes
- `position_mode`: Flexible positioning system

#### `BackgroundRemovalConfig`
Advanced configuration for background removal algorithms.

```python
config = BackgroundRemovalConfig(
    method=RemovalMethod.RMBG_V2,
    model_precision="fp16",
    batch_size=4,
    chroma_color="green",
    tolerance=0.15,
    spill_suppression=0.9,
    edge_feathering=3.0,
    processing_resolution=1024,
    anti_aliasing=True
)
```

#### `AnimationConfig`
Professional animation control system.

```python
config = AnimationConfig(
    animation_type=AnimationType.BOUNCE_IN,
    duration=2.0,
    easing=EasingFunction.EASE_OUT_BOUNCE,
    amplitude=100.0,
    frequency=1.0,
    loop_count=3,
    ping_pong=True
)
```

#### `PositionConfig` 
Comprehensive positioning and transform system.

```python
config = PositionConfig(
    mode=PositionMode.CUSTOM_ANCHOR,
    x_offset=100.0,
    y_offset=-50.0,
    anchor_x=0.5,
    anchor_y=0.3,
    scale_x=1.2,
    scale_y=1.0,
    rotation=15.0,
    flip_horizontal=False
)
```

### Input/Output Types

**Supported Input Types:**
- `IMAGE`: Standard ComfyUI image tensors
- `STRING`: Text prompts and file paths
- `FLOAT`: Numerical parameters with validation
- `INT`: Integer values with range checking
- `BOOLEAN`: Toggle switches for features

**Output Types:**
- `IMAGE`: Processed image sequences
- `MASK`: Generated alpha masks
- `STRING`: Processing metadata and logs
- `FLOAT`: Confidence scores and metrics

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Background Removal Quality Issues

**Problem**: Poor edge quality or missed details
**Solutions**:
- Try RMBG v2.0 (BiRefNet) for best quality
- Enable `alpha_matting` for fine details like hair
- Adjust `edge_feathering` (2.0-5.0 for complex edges)
- Increase `processing_resolution` to 1024 or higher
- Enable `anti_aliasing` for smoother edges

#### 2. Performance & Memory Issues

**Problem**: Slow processing or out-of-memory errors
**Solutions**:
- Reduce `processing_resolution` (try 512px)
- Switch to lighter model (RMBG v1.4)
- Enable `LOW_VRAM` mode in settings
- Reduce `batch_size` to 1-2
- Close other GPU-intensive applications
- Use CPU mode for very limited systems

#### 3. Animation Issues

**Problem**: Jerky or incorrect animations
**Solutions**:
- Check `fps` setting matches your target frame rate
- Verify keyframe `frame_number` values are in correct sequence
- Ensure `easing_function` is appropriate for animation type
- Check that `total_frames` encompasses all keyframes
- Verify node connections are correct

#### 4. Video Processing Issues

**Problem**: Frame inconsistency or export failures
**Solutions**:
- Enable `batch_processing` for video sequences
- Check that all frames have consistent dimensions
- Verify `video_fps` matches source material
- Ensure sufficient disk space for output
- Try different output formats (mp4, webm)

#### 5. Chroma Key Issues

**Problem**: Color spill or incomplete removal
**Solutions**:
- Adjust `tolerance` (start with 0.15, increase gradually)
- Enable `spill_suppression` (0.7-0.9)
- Increase `edge_feathering` (2.0-5.0)
- Check lighting conditions in source material
- Consider `hybrid_ai_chroma` method for difficult scenarios

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### System Requirements Verification

**Minimum Requirements:**
- Python 3.8+
- PyTorch 2.0+
- 4GB RAM
- 2GB storage space

**Recommended:**
- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU with 8GB+ VRAM
- 16GB+ system RAM
- SSD storage

**Optimal:**
- Latest Python 3.11+
- PyTorch 2.2+ with CUDA 12+
- RTX 4080/4090 or equivalent
- 32GB+ system RAM
- NVMe SSD storage

---

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

### Development Setup

```bash
# Clone repository
git clone https://github.com/GeekyGhost/ComfyUI-GeekyRemB.git
cd ComfyUI-GeekyRemB

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Code formatting
black --check scripts/
flake8 scripts/
mypy scripts/
```

---

## License & Acknowledgements

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Note**: Some AI models may have separate licensing requirements for commercial use. Please review individual model licenses.

### Acknowledgements

GeekyRemB v4.0 builds upon exceptional open-source projects:

- **[RMBG](https://github.com/ZhengPeng7/BiRefNet)** - BiRefNet architecture for background removal
- **[BRIA AI](https://github.com/BRIA-AI/BRIA-RMBG-1.4)** - Professional-grade RMBG models  
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - The incredible node-based UI framework
- **[OpenCV](https://opencv.org/)** - Computer vision processing
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

### Special Thanks

- **ComfyUI Community** - Feedback, testing, and feature requests
- **AI Researchers** - Advancing the state-of-the-art in background removal
- **Open Source Contributors** - Making this technology accessible to everyone
- **Beta Testers** - Helping refine v4.0 before release

---

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/GeekyGhost/ComfyUI-GeekyRemB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GeekyGhost/ComfyUI-GeekyRemB/discussions)
- **Documentation**: [Wiki](https://github.com/GeekyGhost/ComfyUI-GeekyRemB/wiki)

---

**GeekyRemB v4.0** - Pushing the boundaries of what's possible in AI-powered image and video processing.

*Made with ❤️ by [GeekyGhost](https://github.com/GeekyGhost) and the open-source community.*
