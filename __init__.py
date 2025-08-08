"""
GeekyRemB v4.0 - Ultimate Background Removal and Video Layering Node for ComfyUI
==============================================================================

This package provides state-of-the-art background removal and professional video layering
capabilities using the latest AI models including RMBG v2.0 with BiRefNet architecture.

Features:
- Latest RMBG v2.0 with BiRefNet architecture for superior background removal
- Advanced chroma key with spill suppression and edge feathering  
- Professional positioning system with multiple modes (absolute, relative, anchored)
- Comprehensive animation system with professional easing functions
- Advanced blending modes for professional compositing
- Video batch processing with frame consistency
- GPU acceleration and memory optimization
- Professional layer effects including drop shadows
- Support for both single images and video sequences

Author: GeekyGhost
Version: 4.0
License: MIT
"""

from .scripts.GeekyRembv2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optionally, you can also add:
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version information
__version__ = "4.0.0"
__author__ = "GeekyGhost"
__license__ = "MIT"
__description__ = "Ultimate AI Background Removal & Video Layering Node for ComfyUI"

# Supported models and features
SUPPORTED_MODELS = [
    "RMBG v2.0 (BiRefNet)",
    "RMBG v1.4 (BRIA AI)", 
    "BiRefNet Direct",
    "Advanced Chroma Key",
    "Hybrid AI+Chroma"
]

FEATURES = [
    "Professional positioning system",
    "Advanced animation system",
    "Professional blending modes",
    "Video batch processing",
    "GPU acceleration",
    "Professional layer effects",
    "Comprehensive error handling"
]
