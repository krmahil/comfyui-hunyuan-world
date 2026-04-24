from setuptools import setup, find_packages

setup(
    name="comfyui-hunyuan-world",
    version="0.1.0",
    description="ComfyUI custom nodes for immersive 3D world generation using Tencent HunyuanWorld 1.0",
    author="MAHIL K R",
    author_email="mahilkr246810@gmail.com",
    url="https://github.com/krmahil/comfyui-hunyuan-world",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "torch>=2.5.0",
        "open3d>=0.18.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
