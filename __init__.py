"""
ComfyUI HunyuanWorld - Custom nodes for immersive 3D world generation using Tencent's HunyuanWorld 1.0.

Nodes included:
- HunyuanWorld Text to Panorama: Generate 360° panoramic images from text prompts
- HunyuanWorld Image to Panorama: Generate 360° panoramic images from input images
- HunyuanWorld Panorama to 3D World: Generate explorable 3D world meshes from panorama images
- HunyuanWorld Full Pipeline: End-to-end text/image to 3D world generation
- HunyuanWorld Model Loader: Load and cache HunyuanWorld models for reuse
"""

__version__ = "0.1.0"

from .hunyuan_world_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# For ComfyUI to detect
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
