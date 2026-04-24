# ComfyUI HunyuanWorld

A ComfyUI custom node pack for **immersive 3D world generation** using [Tencent HunyuanWorld 1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

Generate explorable, interactive 3D worlds from text prompts or images — directly inside ComfyUI.

## Features

- **Text → 360° Panorama**: Generate immersive panoramic images from text descriptions
- **Image → 360° Panorama**: Extend a single perspective image into a full panorama
- **Panorama → 3D World**: Convert panoramas into layered, explorable 3D mesh worlds
- **Full Pipeline**: End-to-end text/image to 3D world in a single node
- **FP8 Quantization**: Run on consumer GPUs (e.g., RTX 4090) with quantization support
- **DeepCache Acceleration**: Speed up inference with caching
- **VRAM Management**: Unload models when done to free GPU memory

## Nodes

### 1. HunyuanWorld Text to Panorama

Generate a 360° panoramic image from a text prompt.

| Input | Type | Description |
|-------|------|-------------|
| `prompt` | STRING | Text description of the scene |
| `negative_prompt` | STRING | What to avoid |
| `seed` | INT | Random seed (default: 42) |
| `height` / `width` | INT | Panorama dimensions (default: 960×1920) |
| `guidance_scale` | FLOAT | CFG scale (default: 30.0) |
| `num_inference_steps` | INT | Diffusion steps (default: 50) |
| `fp8_quantization` | BOOLEAN | Enable FP8 for lower VRAM |
| `use_cache` | BOOLEAN | Enable DeepCache acceleration |

| Output | Type | Description |
|--------|------|-------------|
| `panorama` | IMAGE | Generated panoramic image |
| `panorama_path` | STRING | File path to saved panorama |

### 2. HunyuanWorld Image to Panorama

Extend a perspective image into a full 360° panorama.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | Input perspective image |
| `prompt` | STRING | Optional text guidance |
| `fov` | INT | Field of view (default: 80°) |
| *(+ same options as Text to Panorama)* | | |

### 3. HunyuanWorld Panorama to 3D World

Generate layered 3D meshes from a panoramic image.

| Input | Type | Description |
|-------|------|-------------|
| `panorama_path` | STRING | Path to panorama image |
| `labels_fg1` | STRING | Foreground layer 1 labels (space-separated) |
| `labels_fg2` | STRING | Foreground layer 2 labels (space-separated) |
| `scene_class` | ENUM | `outdoor` or `indoor` |
| `target_resolution` | INT | Output resolution (default: 3840) |
| `export_draco` | BOOLEAN | Export compressed .drc format |

| Output | Type | Description |
|--------|------|-------------|
| `mesh_paths_json` | STRING | JSON with paths to generated .ply meshes |
| `output_directory` | STRING | Directory containing all outputs |
| `layer_count` | INT | Number of mesh layers generated |

### 4. HunyuanWorld Full Pipeline

End-to-end: text or image → panorama → 3D world.

### 5. HunyuanWorld Unload Models

Free GPU VRAM by clearing cached model pipelines.

## Example Workflows

> **Drag & drop** any `.json` file from the `examples/` folder into ComfyUI to load the workflow.

| Workflow | File | Description |
|----------|------|-------------|
| Text → Panorama | `examples/text_to_panorama.json` | Generate a 360° panorama from a text prompt |
| Image → Panorama | `examples/image_to_panorama.json` | Extend a perspective image into a full panorama |
| Text → 3D World | `examples/text_to_3d_world.json` | Two-stage: text → panorama → layered 3D meshes |
| Image → 3D World | `examples/image_to_3d_world.json` | Two-stage: image → panorama → layered 3D meshes |
| Full Pipeline | `examples/full_pipeline.json` | Single-node end-to-end text → 3D world |
| FP8 Quantized | `examples/fp8_quantized_pipeline.json` | RTX 4090 optimized with FP8 + DeepCache (indoor scene) |

### Text to 3D World

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────┐
│ HunyuanWorld            │────▶│ HunyuanWorld             │────▶│ Save/View 3D Mesh    │
│ Text to Panorama        │     │ Panorama to 3D World     │     │ (.ply files)         │
│ prompt: "A serene       │     │ labels_fg1: stones       │     └──────────────────────┘
│  mountain lake..."      │     │ labels_fg2: trees        │
└─────────────────────────┘     └──────────────────────────┘
```

### Image to 3D World

```
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│ Load Image  │────▶│ HunyuanWorld             │────▶│ HunyuanWorld             │
└─────────────┘     │ Image to Panorama        │     │ Panorama to 3D World     │
                    └──────────────────────────┘     └──────────────────────────┘
```

## Installation

### Prerequisites

1. **Python 3.10+** with **PyTorch 2.5.0+cu124**

2. **HunyuanWorld 1.0** — Install from source:
   ```bash
   git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
   cd HunyuanWorld-1.0
   conda env create -f docker/HunyuanWorld.yaml
   ```

3. **Real-ESRGAN** (for super-resolution):
   ```bash
   git clone https://github.com/xinntao/Real-ESRGAN.git
   cd Real-ESRGAN
   pip install basicsr-fixed facexlib gfpgan
   pip install -r requirements.txt
   python setup.py develop
   ```

4. **ZIM** (for segmentation):
   ```bash
   git clone https://github.com/naver-ai/ZIM.git
   cd ZIM && pip install -e .
   ```

5. **HuggingFace login** (for model downloads):
   ```bash
   huggingface-cli login --token YOUR_TOKEN
   ```

### Install the Node

1. Navigate to ComfyUI custom nodes:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone:
   ```bash
   git clone https://github.com/krmahil/comfyui-hunyuan-world.git
   ```

3. Install dependencies:
   ```bash
   pip install -r comfyui-hunyuan-world/requirements.txt
   ```

4. Restart ComfyUI

### Models — What to Download & Where to Put Them

> **Note:** Models are downloaded automatically from HuggingFace on first run via `diffusers`.
> You **must** be logged in to HuggingFace first (see step 5 above).
> If you prefer to pre-download, follow the manual instructions below.

#### Automatic Download (Recommended)

Just run any workflow — the nodes will download models to your HuggingFace cache
(`~/.cache/huggingface/hub/` on Linux, `C:\Users\<you>\.cache\huggingface\hub\` on Windows).

No manual model placement is needed. The first run will be slow (~10-30 min depending on bandwidth).

#### Manual Download (Optional)

If you want to pre-download or use a custom model directory:

```bash
# Login to HuggingFace first (required — FLUX.1-dev is gated)
huggingface-cli login --token YOUR_HF_TOKEN

# Download all HunyuanWorld LoRA weights (~1.9 GB total)
huggingface-cli download tencent/HunyuanWorld-1 --local-dir ./models/HunyuanWorld-1

# Download FLUX.1-dev base model (~23 GB — required for Text2Pano)
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./models/FLUX.1-dev

# Download FLUX.1-Fill-dev base model (~23 GB — required for Image2Pano)
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir ./models/FLUX.1-Fill-dev
```

#### ZIM Segmentation Model (Required for Scene Generation)

```bash
# Download ZIM encoder/decoder ONNX files
mkdir -p zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx
```

Place the `zim_vit_l_2092/` folder inside your HunyuanWorld-1.0 installation directory.

#### Complete Model Reference

| Model | HuggingFace Repo | File(s) | Size | Used By Node |
|-------|-------------------|---------|------|-------------|
| **PanoDiT-Text** (LoRA) | [`tencent/HunyuanWorld-1`](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text) | `HunyuanWorld-PanoDiT-Text/lora.safetensors` | 478 MB | Text to Panorama |
| **PanoDiT-Image** (LoRA) | [`tencent/HunyuanWorld-1`](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image) | `HunyuanWorld-PanoDiT-Image/lora.safetensors` | 478 MB | Image to Panorama |
| **PanoInpaint-Scene** (LoRA) | [`tencent/HunyuanWorld-1`](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) | `HunyuanWorld-PanoInpaint-Scene/lora.safetensors` | 478 MB | Panorama to 3D World |
| **PanoInpaint-Sky** (LoRA) | [`tencent/HunyuanWorld-1`](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky) | `HunyuanWorld-PanoInpaint-Sky/lora.safetensors` | 478 MB | Panorama to 3D World |
| **FLUX.1-dev** (base) | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Full diffusers model | ~23 GB | Text to Panorama |
| **FLUX.1-Fill-dev** (base) | [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | Full diffusers model | ~23 GB | Image to Panorama |
| **ZIM ViT-L** (segmentation) | [`naver-iv/zim-anything-vitl`](https://huggingface.co/naver-iv/zim-anything-vitl) | `encoder.onnx` + `decoder.onnx` | ~1.2 GB | Panorama to 3D World |

> ⚠️ **FLUX.1-dev is a gated model.** You must accept the license at
> [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
> before downloading.

## GPU Requirements

| Mode | VRAM Required |
|------|---------------|
| Full precision | ~40 GB (A100/H100) |
| FP8 quantization | ~24 GB (RTX 4090) |
| FP8 + cache | ~20 GB |

## Requirements

- Python 3.10+
- PyTorch >= 2.5.0 (CUDA 12.4)
- HunyuanWorld 1.0 (`hy3dworld`)
- Open3D >= 0.18.0
- Diffusers >= 0.28.0
- OpenCV, NumPy, Pillow

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on top of [HunyuanWorld 1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) by Tencent.
Thanks to [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), and [Open3D](http://www.open3d.org/).
