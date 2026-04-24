import os
import torch
import numpy as np
import cv2
import json
import tempfile
from PIL import Image

# Try to import HunyuanWorld dependencies
try:
    from hy3dworld import Text2PanoramaPipelines, Image2PanoramaPipelines, Perspective
    from hy3dworld import LayerDecomposition, WorldComposer, process_file
    from hy3dworld.AngelSlim.gemm_quantization_processor import FluxFp8GeMMProcessor
    from hy3dworld.AngelSlim.attention_quantization_processor import FluxFp8AttnProcessor2_0
    from hy3dworld.AngelSlim.cache_helper import DeepCacheHelper
    HY3D_AVAILABLE = True
except ImportError:
    HY3D_AVAILABLE = False
    print("[ComfyUI-HunyuanWorld] hy3dworld not available - please install HunyuanWorld-1.0")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("[ComfyUI-HunyuanWorld] open3d not available - 3D world generation will not work")


# ─────────────────────────────────────────────
# Shared model cache (singleton across nodes)
# ─────────────────────────────────────────────
_MODEL_CACHE = {}


def _get_output_dir():
    """Get a temp output directory for intermediate files."""
    base = os.path.join(tempfile.gettempdir(), "comfyui_hunyuan_world")
    os.makedirs(base, exist_ok=True)
    return base


# ═══════════════════════════════════════════════
#  NODE 1 — Text to Panorama
# ═══════════════════════════════════════════════
class HYWorldText2PanoNode:
    """
    Generate a 360° panoramic image from a text prompt using HunyuanWorld PanoDiT-Text.
    """

    def __init__(self):
        self.pipe = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the 3D world scene you want to generate"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "What to avoid in the generation"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                }),
                "height": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1920, "min": 512, "max": 4096, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1}),
                "fp8_quantization": ("BOOLEAN", {"default": False}),
                "use_cache": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panorama", "panorama_path")
    FUNCTION = "generate_panorama"
    CATEGORY = "3d/hunyuan-world"

    def _load_pipeline(self, fp8_quantization=False, fp8_gemm=False):
        """Load the Text2Panorama pipeline (cached)."""
        if not HY3D_AVAILABLE:
            raise RuntimeError("hy3dworld is not installed. Please install HunyuanWorld-1.0.")

        cache_key = "text2pano"
        if cache_key not in _MODEL_CACHE:
            print("[HunyuanWorld] Loading Text2Panorama pipeline...")
            lora_path = "tencent/HunyuanWorld-1"
            model_path = "black-forest-labs/FLUX.1-dev"

            pipe = Text2PanoramaPipelines.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
            pipe.load_lora_weights(
                lora_path,
                subfolder="HunyuanWorld-PanoDiT-Text",
                weight_name="lora.safetensors",
                torch_dtype=torch.bfloat16,
            )
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()

            if fp8_quantization:
                pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
                FluxFp8GeMMProcessor(pipe.transformer)

            _MODEL_CACHE[cache_key] = pipe
            print("[HunyuanWorld] Text2Panorama pipeline loaded.")

        return _MODEL_CACHE[cache_key]

    def generate_panorama(self, prompt, negative_prompt="", seed=42,
                          height=960, width=1920, guidance_scale=30.0,
                          num_inference_steps=50, blend_extend=6,
                          fp8_quantization=False, use_cache=False):
        if not HY3D_AVAILABLE:
            raise RuntimeError("hy3dworld is not installed.")

        pipe = self._load_pipeline(fp8_quantization)

        helper = None
        if use_cache:
            helper = DeepCacheHelper(
                pipe_model=pipe.transformer,
                no_cache_steps=list(range(0, 10)) + list(range(10, 40, 3)) + list(range(40, 50)),
                no_cache_block_id={"single": [38]},
            )
            helper.start_timestep = 0
            helper.enable()

        image = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt or None,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            blend_extend=blend_extend,
            true_cfg_scale=0.0,
            helper=helper,
        ).images[0]

        # Save panorama to disk
        output_dir = _get_output_dir()
        pano_path = os.path.join(output_dir, "panorama.png")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.save(pano_path)

        # Convert to ComfyUI tensor (B, H, W, C)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor, pano_path)


# ═══════════════════════════════════════════════
#  NODE 2 — Image to Panorama
# ═══════════════════════════════════════════════
class HYWorldImage2PanoNode:
    """
    Generate a 360° panoramic image from an input image using HunyuanWorld PanoDiT-Image.
    """

    def __init__(self):
        self.pipe = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional text guidance for panorama generation"
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "height": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1920, "min": 512, "max": 4096, "step": 64}),
                "fov": ("INT", {"default": 80, "min": 30, "max": 180, "step": 5}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1}),
                "fp8_quantization": ("BOOLEAN", {"default": False}),
                "use_cache": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panorama", "panorama_path")
    FUNCTION = "generate_panorama"
    CATEGORY = "3d/hunyuan-world"

    def _load_pipeline(self, fp8_quantization=False):
        if not HY3D_AVAILABLE:
            raise RuntimeError("hy3dworld is not installed.")

        cache_key = "img2pano"
        if cache_key not in _MODEL_CACHE:
            print("[HunyuanWorld] Loading Image2Panorama pipeline...")
            lora_path = "tencent/HunyuanWorld-1"
            model_path = "black-forest-labs/FLUX.1-Fill-dev"

            pipe = Image2PanoramaPipelines.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
            pipe.load_lora_weights(
                lora_path,
                subfolder="HunyuanWorld-PanoDiT-Image",
                weight_name="lora.safetensors",
                torch_dtype=torch.bfloat16,
            )
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()

            if fp8_quantization:
                pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
                FluxFp8GeMMProcessor(pipe.transformer)

            _MODEL_CACHE[cache_key] = pipe
            print("[HunyuanWorld] Image2Panorama pipeline loaded.")

        return _MODEL_CACHE[cache_key]

    def generate_panorama(self, image, prompt="", negative_prompt="",
                          seed=42, height=960, width=1920, fov=80,
                          guidance_scale=30.0, num_inference_steps=50,
                          blend_extend=6, fp8_quantization=False,
                          use_cache=False):
        if not HY3D_AVAILABLE:
            raise RuntimeError("hy3dworld is not installed.")

        pipe = self._load_pipeline(fp8_quantization)

        general_neg = "human, person, people, messy, low-quality, blur, noise, low-resolution"
        general_pos = "high-quality, high-resolution, sharp, clear, 8k"

        full_prompt = (prompt + ", " + general_pos) if prompt else general_pos
        full_neg = general_neg + (", " + negative_prompt if negative_prompt else "")

        # Convert ComfyUI image tensor to OpenCV BGR
        if isinstance(image, torch.Tensor):
            img_np = image[0].cpu().numpy() if len(image.shape) == 4 else image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)

        perspective_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Resize perspective image
        h_fov, w_fov = perspective_img.shape[:2]
        if w_fov > h_fov:
            ratio = w_fov / h_fov
            w = int((fov / 360) * width)
            h = int(w / ratio)
        else:
            ratio = h_fov / w_fov
            h = int((fov / 180) * height)
            w = int(h / ratio)
        perspective_img = cv2.resize(perspective_img, (w, h), interpolation=cv2.INTER_AREA)

        # Create equirectangular projection
        equ = Perspective(perspective_img, fov, 0, 0, crop_bound=False)
        img_equ, mask = equ.GetEquirec(height, width)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5)
        img_equ = img_equ * mask
        mask = 255 - (mask.astype(np.uint8) * 255)

        mask_pil = Image.fromarray(mask[:, :, 0])
        img_pil = Image.fromarray(cv2.cvtColor(img_equ.astype(np.uint8), cv2.COLOR_BGR2RGB))

        helper = None
        if use_cache:
            helper = DeepCacheHelper(
                pipe.transformer,
                no_cache_steps=list(range(0, 10)) + list(range(10, 40, 3)) + list(range(40, 50)),
                no_cache_block_id={"single": [38]},
            )
            helper.start_timestep = 0
            helper.enable()

        result = pipe(
            prompt=full_prompt,
            image=img_pil,
            mask_image=mask_pil,
            height=height,
            width=width,
            negative_prompt=full_neg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            blend_extend=blend_extend,
            shifting_extend=0,
            true_cfg_scale=2.0,
            helper=helper,
        ).images[0]

        # Save
        output_dir = _get_output_dir()
        pano_path = os.path.join(output_dir, "panorama.png")
        result.save(pano_path)

        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor, pano_path)


# ═══════════════════════════════════════════════
#  NODE 3 — Panorama to 3D World (Scene Gen)
# ═══════════════════════════════════════════════
class HYWorldSceneGenNode:
    """
    Generate an explorable 3D world mesh from a panoramic image using
    HunyuanWorld's LayerDecomposition + WorldComposer pipeline.
    """

    def __init__(self):
        self.decomposer = None
        self.composer = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to panorama PNG (from Text/Image-to-Pano node)"
                }),
            },
            "optional": {
                "labels_fg1": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Layer 1 foreground labels (space-separated): stones flowers"
                }),
                "labels_fg2": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Layer 2 foreground labels (space-separated): trees mountains"
                }),
                "scene_class": (["outdoor", "indoor"],),
                "target_resolution": ("INT", {"default": 3840, "min": 1920, "max": 7680, "step": 960}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "fp8_quantization": ("BOOLEAN", {"default": False}),
                "export_draco": ("BOOLEAN", {"default": False}),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional output directory (auto-generated if empty)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("mesh_paths_json", "output_directory", "layer_count")
    FUNCTION = "generate_world"
    CATEGORY = "3d/hunyuan-world"

    def _build_args_namespace(self, fp8_quantization=False, cache=False):
        """Build an argparse-like namespace for HunyuanWorld internals."""
        import types
        args = types.SimpleNamespace()
        args.fp8_attention = fp8_quantization
        args.fp8_gemm = fp8_quantization
        args.cache = cache
        return args

    def generate_world(self, panorama_path, labels_fg1="", labels_fg2="",
                       scene_class="outdoor", target_resolution=3840,
                       seed=42, fp8_quantization=False, export_draco=False,
                       output_dir=""):
        if not HY3D_AVAILABLE:
            raise RuntimeError("hy3dworld is not installed.")
        if not O3D_AVAILABLE:
            raise RuntimeError("open3d is not installed.")

        if not panorama_path or not os.path.exists(panorama_path):
            raise FileNotFoundError(f"Panorama image not found: {panorama_path}")

        # Parse foreground labels
        fg1 = labels_fg1.strip().split() if labels_fg1.strip() else []
        fg2 = labels_fg2.strip().split() if labels_fg2.strip() else []

        # Output directory
        if not output_dir:
            output_dir = os.path.join(_get_output_dir(), "scene_output")
        os.makedirs(output_dir, exist_ok=True)

        args = self._build_args_namespace(fp8_quantization)

        # Initialize components
        kernel_scale = max(1, int(target_resolution / 1920))
        decomposer = LayerDecomposition(args)
        composer = WorldComposer(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            resolution=(target_resolution, target_resolution // 2),
            seed=seed,
            filter_mask=True,
            kernel_scale=kernel_scale,
        )

        if fp8_quantization:
            decomposer.inpaint_fg_model.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
            decomposer.inpaint_sky_model.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
            FluxFp8GeMMProcessor(decomposer.inpaint_fg_model.transformer)
            FluxFp8GeMMProcessor(decomposer.inpaint_sky_model.transformer)

        # Build layer infos
        fg1_infos = [{"image_path": panorama_path, "output_path": output_dir,
                      "labels": fg1, "class": scene_class}]
        fg2_infos = [{"image_path": os.path.join(output_dir, "remove_fg1_image.png"),
                      "output_path": output_dir, "labels": fg2, "class": scene_class}]

        # Layer decomposition
        print("[HunyuanWorld] Running layer decomposition...")
        decomposer(fg1_infos, layer=0)
        decomposer(fg2_infos, layer=1)
        decomposer(fg2_infos, layer=2)

        separate_pano, fg_bboxes = composer._load_separate_pano_from_dir(output_dir, sr=True)

        # World reconstruction
        print("[HunyuanWorld] Generating 3D world mesh...")
        layered_world_mesh = composer.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type="mesh"
        )

        # Save meshes
        mesh_paths = []
        for layer_idx, layer_info in enumerate(layered_world_mesh):
            ply_path = os.path.join(output_dir, f"mesh_layer{layer_idx}.ply")
            o3d.io.write_triangle_mesh(ply_path, layer_info["mesh"])
            mesh_paths.append(ply_path)

            if export_draco:
                drc_path = os.path.join(output_dir, f"mesh_layer{layer_idx}.drc")
                process_file(ply_path, drc_path)
                mesh_paths.append(drc_path)

        layer_count = len(layered_world_mesh)
        mesh_json = json.dumps({"meshes": mesh_paths, "layers": layer_count}, indent=2)
        print(f"[HunyuanWorld] Generated {layer_count} mesh layers in {output_dir}")

        return (mesh_json, output_dir, layer_count)


# ═══════════════════════════════════════════════
#  NODE 4 — Full Pipeline (Text/Image → 3D World)
# ═══════════════════════════════════════════════
class HYWorldFullPipelineNode:
    """
    End-to-end pipeline: Text or Image → Panorama → 3D World.
    Combines panorama generation and scene generation in a single node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Text prompt for world generation"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "labels_fg1": ("STRING", {"default": "", "placeholder": "stones flowers"}),
                "labels_fg2": ("STRING", {"default": "", "placeholder": "trees mountains"}),
                "scene_class": (["outdoor", "indoor"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 100}),
                "target_resolution": ("INT", {"default": 3840, "min": 1920, "max": 7680, "step": 960}),
                "fp8_quantization": ("BOOLEAN", {"default": False}),
                "use_cache": ("BOOLEAN", {"default": False}),
                "export_draco": ("BOOLEAN", {"default": False}),
                "output_dir": ("STRING", {"default": "", "placeholder": "Output directory"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("panorama", "mesh_paths_json", "output_directory", "layer_count")
    FUNCTION = "run_pipeline"
    CATEGORY = "3d/hunyuan-world"

    def run_pipeline(self, prompt, image=None, negative_prompt="",
                     labels_fg1="", labels_fg2="", scene_class="outdoor",
                     seed=42, guidance_scale=30.0, num_inference_steps=50,
                     target_resolution=3840, fp8_quantization=False,
                     use_cache=False, export_draco=False, output_dir=""):
        if not output_dir:
            output_dir = os.path.join(_get_output_dir(), "full_pipeline")
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Generate panorama
        if image is not None:
            pano_node = HYWorldImage2PanoNode()
            pano_tensor, pano_path = pano_node.generate_panorama(
                image=image, prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                fp8_quantization=fp8_quantization, use_cache=use_cache,
            )
        else:
            pano_node = HYWorldText2PanoNode()
            pano_tensor, pano_path = pano_node.generate_panorama(
                prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                fp8_quantization=fp8_quantization, use_cache=use_cache,
            )

        # Copy panorama to output dir
        import shutil
        final_pano_path = os.path.join(output_dir, "panorama.png")
        shutil.copy(pano_path, final_pano_path)

        # Step 2: Generate 3D world
        scene_node = HYWorldSceneGenNode()
        mesh_json, out_dir, layer_count = scene_node.generate_world(
            panorama_path=final_pano_path,
            labels_fg1=labels_fg1, labels_fg2=labels_fg2,
            scene_class=scene_class, target_resolution=target_resolution,
            seed=seed, fp8_quantization=fp8_quantization,
            export_draco=export_draco, output_dir=output_dir,
        )

        return (pano_tensor, mesh_json, out_dir, layer_count)


# ═══════════════════════════════════════════════
#  NODE 5 — Unload Models (VRAM management)
# ═══════════════════════════════════════════════
class HYWorldUnloadModelsNode:
    """
    Unload cached HunyuanWorld models to free GPU VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "3d/hunyuan-world"

    def unload(self, confirm=True):
        if not confirm:
            return ("Skipped — confirm was False.",)

        global _MODEL_CACHE
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (f"Unloaded {count} cached model(s). VRAM freed.",)


# ═══════════════════════════════════════════════
#  Node Registration
# ═══════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "HYWorldText2Pano": HYWorldText2PanoNode,
    "HYWorldImage2Pano": HYWorldImage2PanoNode,
    "HYWorldSceneGen": HYWorldSceneGenNode,
    "HYWorldFullPipeline": HYWorldFullPipelineNode,
    "HYWorldUnloadModels": HYWorldUnloadModelsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorldText2Pano": "HunyuanWorld Text to Panorama",
    "HYWorldImage2Pano": "HunyuanWorld Image to Panorama",
    "HYWorldSceneGen": "HunyuanWorld Panorama to 3D World",
    "HYWorldFullPipeline": "HunyuanWorld Full Pipeline (Text/Image → 3D)",
    "HYWorldUnloadModels": "HunyuanWorld Unload Models",
}
