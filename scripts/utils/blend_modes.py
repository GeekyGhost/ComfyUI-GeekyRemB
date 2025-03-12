import numpy as np
from typing import Dict, Callable

class EnhancedBlendMode:
    """Enhanced blend mode operations with optimized processing"""
    
    @staticmethod
    def _ensure_rgba(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype) * 255
                return np.concatenate([img, alpha], axis=-1)
            return img
        return np.stack([img] * 4, axis=-1)

    @staticmethod
    def _apply_blend(target: np.ndarray, blend: np.ndarray, operation, opacity: float = 1.0) -> np.ndarray:
        target = EnhancedBlendMode._ensure_rgba(target).astype(np.float32)
        blend = EnhancedBlendMode._ensure_rgba(blend).astype(np.float32)
        
        target = target / 255.0
        blend = blend / 255.0
        
        target_rgb = target[..., :3]
        blend_rgb = blend[..., :3]
        target_a = target[..., 3:4]
        blend_a = blend[..., 3:4]
        
        result_rgb = operation(target_rgb, blend_rgb)
        result_a = target_a + blend_a * (1 - target_a) * opacity
        
        result = np.concatenate([
            result_rgb * opacity + target_rgb * (1 - opacity),
            result_a
        ], axis=-1)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    @classmethod
    def get_blend_modes(cls) -> Dict[str, Callable]:
        return {
            "normal": cls.normal,
            "multiply": cls.multiply,
            "screen": cls.screen,
            "overlay": cls.overlay,
            "soft_light": cls.soft_light,
            "hard_light": cls.hard_light,
            "difference": cls.difference,
            "exclusion": cls.exclusion,
            "color_dodge": cls.color_dodge,
            "color_burn": cls.color_burn,
            "linear_light": cls.linear_light,
            "pin_light": cls.pin_light,
        }

    @staticmethod
    def normal(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: b, opacity)

    @staticmethod
    def multiply(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: t * b, opacity)

    @staticmethod
    def screen(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: 1 - (1 - t) * (1 - b), opacity)

    @staticmethod
    def overlay(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def overlay_op(t, b):
            return np.where(t > 0.5, 1 - 2 * (1 - t) * (1 - b), 2 * t * b)
        return EnhancedBlendMode._apply_blend(target, blend, overlay_op, opacity)

    @staticmethod
    def soft_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def soft_light_op(t, b):
            return np.where(b > 0.5,
                          t + (2 * b - 1) * (t - t * t),
                          t - (1 - 2 * b) * t * (1 - t))
        return EnhancedBlendMode._apply_blend(target, blend, soft_light_op, opacity)

    @staticmethod
    def hard_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def hard_light_op(t, b):
            return np.where(b > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return EnhancedBlendMode._apply_blend(target, blend, hard_light_op, opacity)

    @staticmethod
    def difference(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: np.abs(t - b), opacity)

    @staticmethod
    def exclusion(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        return EnhancedBlendMode._apply_blend(target, blend, lambda t, b: t + b - 2 * t * b, opacity)

    @staticmethod
    def color_dodge(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def color_dodge_op(t, b):
            return np.where(b >= 1, 1, np.minimum(1, t / (1 - b + 1e-6)))
        return EnhancedBlendMode._apply_blend(target, blend, color_dodge_op, opacity)

    @staticmethod
    def color_burn(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def color_burn_op(t, b):
            return np.where(b <= 0, 0, np.maximum(0, 1 - (1 - t) / (b + 1e-6)))
        return EnhancedBlendMode._apply_blend(target, blend, color_burn_op, opacity)

    @staticmethod
    def linear_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def linear_light_op(t, b):
            return np.clip(2 * b + t - 1, 0, 1)
        return EnhancedBlendMode._apply_blend(target, blend, linear_light_op, opacity)

    @staticmethod
    def pin_light(target: np.ndarray, blend: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        def pin_light_op(t, b):
            return np.where(b > 0.5,
                          np.maximum(t, 2 * (b - 0.5)),
                          np.minimum(t, 2 * b))
        return EnhancedBlendMode._apply_blend(target, blend, pin_light_op, opacity)