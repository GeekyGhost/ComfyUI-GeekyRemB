# Import all node classes
from .scripts.geekyremb_core import GeekyRemB
from .scripts.geekyremb_animator import GeekyRemB_Animator
from .scripts.geekyremb_lightshadow import GeekyRemB_LightShadow
from .scripts.geekyremb_keyframe_position import GeekyRemB_KeyframePosition

# Create node mappings
NODE_CLASS_MAPPINGS = {
    "GeekyRemB": GeekyRemB,
    "GeekyRemB_Animator": GeekyRemB_Animator,
    "GeekyRemB_LightShadow": GeekyRemB_LightShadow,
    "GeekyRemB_KeyframePosition": GeekyRemB_KeyframePosition
}

# Create display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyRemB": "Geeky RemB",
    "GeekyRemB_Animator": "Geeky RemB Animator",
    "GeekyRemB_LightShadow": "Geeky RemB Light & Shadow",
    "GeekyRemB_KeyframePosition": "Geeky RemB Keyframe Position"
}

# Register custom types for ComfyUI
custom_output_types = {
    "ANIMATOR": ("ANIMATOR", ),
    "LIGHTSHADOW": ("LIGHTSHADOW", ),
    "KEYFRAME": ("KEYFRAME", )
}

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'custom_output_types']
