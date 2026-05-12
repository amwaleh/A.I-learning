# Project 5: Build a Multi-modal Generation Agent — Hands-on Tutorial

## Overview

In this project, you'll build a **Multi-modal Generation Agent** — an interactive application that can:
- Generate images from text prompts using Stable Diffusion
- Fine-tune generation parameters (steps, guidance scale, scheduler)
- Create image variations from existing images
- Generate simple text-to-video clips
- Provide a web UI for easy interaction

```
┌─────────────────────────────────────────────────────────────┐
│              WHAT YOU'LL BUILD                                │
│                                                             │
│  User: "A cyberpunk city at sunset"                         │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                   │
│  │  Multi-modal Agent  │                                   │
│  │  • Text-to-Image    │──► 🖼️ Generated Image             │
│  │  • Image Variations │──► 🖼️ Modified Image              │
│  │  • Text-to-Video    │──► 🎬 Generated Video             │
│  │  • Parameter Tuning │                                   │
│  └─────────────────────┘                                   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                   │
│  │   Gradio Web UI     │                                   │
│  └─────────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Step 1: Basic Text-to-Image Generation](#step-1-basic-text-to-image-generation)
4. [Step 2: Parameter Tuning](#step-2-parameter-tuning)
5. [Step 3: Image Variation Pipeline](#step-3-image-variation-pipeline)
6. [Step 4: Text-to-Video Generation](#step-4-text-to-video-generation)
7. [Step 5: Building the Gradio UI](#step-5-building-the-gradio-ui)
8. [Step 6: Complete Multi-modal Agent](#step-6-complete-multi-modal-agent)
9. [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)

---

## Prerequisites

**Hardware Requirements:**
- GPU with at least 8GB VRAM (NVIDIA recommended)
- 16GB+ system RAM
- 20GB+ free disk space (for model weights)
- If no GPU: You can use Google Colab (free tier has T4 GPU)

**Software Requirements:**
- Python 3.9 or later
- pip (Python package manager)
- Git (for cloning repos)

**Knowledge Requirements:**
- Basic Python programming
- Understanding of command line / terminal
- Completed the README.md learning material (recommended)

---

## Environment Setup

### Step 1: Create a Virtual Environment

```bash
# Create project directory
mkdir multimodal-agent
cd multimodal-agent

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face libraries
pip install diffusers transformers accelerate

# Image/Video processing
pip install Pillow opencv-python imageio imageio-ffmpeg

# UI framework
pip install gradio

# Utilities
pip install numpy scipy safetensors
```

### Step 3: Verify Installation

Create a file called `verify_setup.py`:

```python
"""verify_setup.py - Verify all dependencies are installed correctly."""

import sys

def check_imports():
    """Check that all required packages can be imported."""
    packages = {
        "torch": "PyTorch",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "accelerate": "Accelerate",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "gradio": "Gradio",
        "numpy": "NumPy",
        "imageio": "ImageIO",
    }

    all_good = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_good = False

    return all_good


def check_gpu():
    """Check GPU availability."""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  ✓ GPU available: {gpu_name} ({vram:.1f} GB VRAM)")
        return True
    else:
        print("  ⚠ No GPU detected. Generation will be SLOW on CPU.")
        print("    Consider using Google Colab for free GPU access.")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Multi-modal Agent - Environment Check")
    print("=" * 50)

    print("\n📦 Checking packages...")
    packages_ok = check_imports()

    print("\n🖥️  Checking GPU...")
    gpu_ok = check_gpu()

    print("\n" + "=" * 50)
    if packages_ok:
        print("✅ All packages installed! You're ready to go.")
    else:
        print("❌ Some packages are missing. Run: pip install -r requirements.txt")

    if not gpu_ok:
        print("💡 Tip: Code will work on CPU but expect 5-10x slower generation.")
    print("=" * 50)
```

**Run it:**
```bash
python verify_setup.py
```

**Expected output:**
```
==================================================
Multi-modal Agent - Environment Check
==================================================

📦 Checking packages...
  ✓ PyTorch installed
  ✓ Diffusers installed
  ✓ Transformers installed
  ✓ Accelerate installed
  ✓ Pillow installed
  ✓ OpenCV installed
  ✓ Gradio installed
  ✓ NumPy installed
  ✓ ImageIO installed

🖥️  Checking GPU...
  ✓ GPU available: NVIDIA GeForce RTX 3080 (10.0 GB VRAM)

==================================================
✅ All packages installed! You're ready to go.
==================================================
```

---

## Step 1: Basic Text-to-Image Generation

Our first milestone: generate an image from a text prompt using Stable Diffusion.

Create a file called `step1_basic_t2i.py`:

```python
"""step1_basic_t2i.py - Basic text-to-image generation with Stable Diffusion."""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def load_model(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion model.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Loaded pipeline ready for generation.
    """
    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Use half precision on GPU (saves VRAM)
    else:
        device = "cpu"
        dtype = torch.float32  # CPU requires full precision

    print(f"Loading model on {device}...")

    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # Disable for faster loading (re-enable for production)
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()  # Reduces VRAM usage

    print("Model loaded successfully!")
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str = "blurry, bad quality, distorted",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> Image.Image:
    """
    Generate an image from a text prompt.

    Args:
        pipe: The Stable Diffusion pipeline.
        prompt: What you want to generate.
        negative_prompt: What you want to avoid.
        width: Image width in pixels (must be multiple of 8).
        height: Image height in pixels (must be multiple of 8).
        num_inference_steps: Number of denoising steps (more = better quality, slower).
        guidance_scale: How closely to follow the prompt (higher = more faithful).
        seed: Random seed for reproducibility (None = random).

    Returns:
        Generated PIL Image.
    """
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    print(f"Generating: '{prompt}'")
    print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")

    # Generate!
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    print("Generation complete!")
    return image


if __name__ == "__main__":
    # Load the model (first run will download ~4GB)
    pipe = load_model()

    # Generate an image
    image = generate_image(
        pipe,
        prompt="A majestic mountain landscape at sunset, oil painting style",
        seed=42,
    )

    # Save the result
    output_path = "output_step1.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
```

**Run it:**
```bash
python step1_basic_t2i.py
```

**Expected output:**
```
Loading model on cuda...
Model loaded successfully!
Generating: 'A majestic mountain landscape at sunset, oil painting style'
  Steps: 30, Guidance: 7.5, Size: 512x512
Generation complete!
Image saved to: output_step1.png
```

**🎉 Milestone 1 Complete!** You should see a generated landscape image saved as `output_step1.png`.

---

## Step 2: Parameter Tuning

Now let's explore how different parameters affect generation quality.

Create a file called `step2_parameter_tuning.py`:

```python
"""step2_parameter_tuning.py - Explore how parameters affect image generation."""

import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
)
from PIL import Image
import os


def load_model(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    """Load Stable Diffusion pipeline."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


# Available schedulers (samplers)
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
    "lms": LMSDiscreteScheduler,
}


def set_scheduler(pipe: StableDiffusionPipeline, scheduler_name: str):
    """
    Change the scheduler (sampler) used for generation.

    Different schedulers produce different results and may need
    different numbers of steps.

    Args:
        pipe: The pipeline to modify.
        scheduler_name: One of 'euler', 'euler_a', 'dpm++_2m', 'ddim', 'lms'.
    """
    if scheduler_name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Choose from: {list(SCHEDULERS.keys())}")

    scheduler_class = SCHEDULERS[scheduler_name]
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    print(f"Scheduler set to: {scheduler_name}")


def compare_guidance_scales(pipe, prompt: str, scales: list, seed: int = 42):
    """
    Generate images at different guidance scales to see the effect.

    Low guidance (1-3): More creative/diverse but may not match prompt.
    Medium guidance (5-8): Good balance of quality and prompt adherence.
    High guidance (10-20): Very literal interpretation, may be oversaturated.
    """
    os.makedirs("comparisons", exist_ok=True)
    images = []

    for scale in scales:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            guidance_scale=scale,
            num_inference_steps=25,
            generator=generator,
        )
        image = result.images[0]
        images.append(image)

        filename = f"comparisons/guidance_{scale:.1f}.png"
        image.save(filename)
        print(f"  Saved: {filename} (guidance_scale={scale})")

    return images


def compare_steps(pipe, prompt: str, step_counts: list, seed: int = 42):
    """
    Generate images with different numbers of denoising steps.

    Fewer steps (5-10): Fast but lower quality.
    Medium steps (20-30): Good balance.
    Many steps (50+): Diminishing returns, slightly better details.
    """
    os.makedirs("comparisons", exist_ok=True)
    images = []

    for steps in step_counts:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]
        images.append(image)

        filename = f"comparisons/steps_{steps}.png"
        image.save(filename)
        print(f"  Saved: {filename} (steps={steps})")

    return images


def compare_schedulers(pipe, prompt: str, seed: int = 42):
    """
    Generate images using different schedulers to see quality differences.
    """
    os.makedirs("comparisons", exist_ok=True)
    images = []

    for name in SCHEDULERS:
        set_scheduler(pipe, name)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]
        images.append(image)

        filename = f"comparisons/scheduler_{name}.png"
        image.save(filename)
        print(f"  Saved: {filename}")

    return images


def create_comparison_grid(images: list, labels: list, output_path: str):
    """
    Combine multiple images into a single comparison grid.

    Args:
        images: List of PIL Images.
        labels: List of text labels for each image.
        output_path: Where to save the grid image.
    """
    from PIL import ImageDraw, ImageFont

    n = len(images)
    img_size = images[0].size[0]
    padding = 10
    label_height = 30

    # Create grid (single row)
    grid_width = n * img_size + (n - 1) * padding
    grid_height = img_size + label_height

    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * (img_size + padding)
        grid.paste(img, (x_offset, label_height))
        # Draw label
        draw.text((x_offset + 5, 5), label, fill="black")

    grid.save(output_path)
    print(f"Comparison grid saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Parameter Tuning Exploration")
    print("=" * 60)

    pipe = load_model()
    prompt = "A cute robot reading a book in a cozy library, digital art"

    # Comparison 1: Guidance Scale
    print("\n📊 Comparing guidance scales...")
    scales = [1.5, 5.0, 7.5, 12.0, 20.0]
    guidance_images = compare_guidance_scales(pipe, prompt, scales)
    create_comparison_grid(
        guidance_images,
        [f"CFG={s}" for s in scales],
        "comparisons/grid_guidance.png",
    )

    # Comparison 2: Number of Steps
    print("\n📊 Comparing step counts...")
    step_counts = [5, 10, 20, 30, 50]
    step_images = compare_steps(pipe, prompt, step_counts)
    create_comparison_grid(
        step_images,
        [f"{s} steps" for s in step_counts],
        "comparisons/grid_steps.png",
    )

    # Comparison 3: Schedulers
    print("\n📊 Comparing schedulers...")
    scheduler_images = compare_schedulers(pipe, prompt)
    create_comparison_grid(
        scheduler_images,
        list(SCHEDULERS.keys()),
        "comparisons/grid_schedulers.png",
    )

    print("\n✅ All comparisons complete! Check the 'comparisons/' folder.")
    print("\n💡 Key takeaways:")
    print("  • Guidance 7-8 is usually the sweet spot")
    print("  • 20-30 steps is sufficient for most schedulers")
    print("  • DPM++ 2M and Euler are fast and high-quality")
```

**Run it:**
```bash
python step2_parameter_tuning.py
```

**Expected output:**
```
============================================================
Step 2: Parameter Tuning Exploration
============================================================

📊 Comparing guidance scales...
  Saved: comparisons/guidance_1.5.png (guidance_scale=1.5)
  Saved: comparisons/guidance_5.0.png (guidance_scale=5.0)
  ...

📊 Comparing step counts...
  Saved: comparisons/steps_5.png (steps=5)
  Saved: comparisons/steps_10.png (steps=10)
  ...

📊 Comparing schedulers...
  Saved: comparisons/scheduler_euler.png
  ...

✅ All comparisons complete! Check the 'comparisons/' folder.
```

**🎉 Milestone 2 Complete!** You now understand how parameters affect generation.

---

## Step 3: Image Variation Pipeline

Create variations of an existing image — modify style, add elements, or create alternatives.

Create a file called `step3_image_variations.py`:

```python
"""step3_image_variations.py - Generate variations of existing images."""

import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    AutoPipelineForImage2Image,
)
from PIL import Image
import os


def load_img2img_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> StableDiffusionImg2ImgPipeline:
    """Load the image-to-image pipeline."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    print(f"Img2Img pipeline loaded on {device}")
    return pipe


def generate_base_image(prompt: str, seed: int = 42) -> Image.Image:
    """Generate a base image to create variations from."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator, num_inference_steps=30)
    return result.images[0]


def create_variation(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> Image.Image:
    """
    Create a variation of an existing image.

    Args:
        pipe: The img2img pipeline.
        image: Input image to modify.
        prompt: Description of desired output.
        strength: How much to change (0.0 = no change, 1.0 = completely new).
                  - 0.2-0.4: Subtle changes (color adjustments, minor edits)
                  - 0.5-0.7: Moderate changes (style transfer, add elements)
                  - 0.8-1.0: Major changes (almost completely regenerated)
        guidance_scale: How closely to follow the prompt.
        seed: Random seed for reproducibility.

    Returns:
        Modified PIL Image.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Ensure image is the right size
    image = image.resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=30,
    )

    return result.images[0]


def style_transfer(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    style: str,
    strength: float = 0.65,
    seed: int = 42,
) -> Image.Image:
    """
    Apply a style to an existing image.

    Args:
        pipe: The img2img pipeline.
        image: Input image.
        style: Style description (e.g., "watercolor", "pixel art", "van gogh").
        strength: How strongly to apply the style.
        seed: Random seed.

    Returns:
        Styled PIL Image.
    """
    prompt = f"({style} style), high quality, detailed"
    return create_variation(pipe, image, prompt, strength=strength, seed=seed)


def create_variation_series(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    strengths: list = None,
) -> list:
    """
    Create a series of variations at different strength levels.

    This helps visualize how the strength parameter affects output.
    """
    if strengths is None:
        strengths = [0.2, 0.4, 0.6, 0.8, 1.0]

    variations = []
    for strength in strengths:
        variation = create_variation(
            pipe, image, prompt, strength=strength, seed=42
        )
        variations.append((variation, strength))

    return variations


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Image Variation Pipeline")
    print("=" * 60)

    os.makedirs("variations", exist_ok=True)

    # Generate a base image (or load one)
    print("\n🎨 Generating base image...")
    base_prompt = "A serene lake surrounded by mountains, photograph"
    base_image = generate_base_image(base_prompt, seed=42)
    base_image.save("variations/base_image.png")
    print("  Base image saved to: variations/base_image.png")

    # Load img2img pipeline
    print("\n📦 Loading img2img pipeline...")
    pipe = load_img2img_pipeline()

    # Create variations with different strengths
    print("\n🔄 Creating strength comparison...")
    strengths = [0.3, 0.5, 0.7, 0.9]
    for strength in strengths:
        variation = create_variation(
            pipe,
            base_image,
            prompt="A serene lake surrounded by mountains at sunset, golden light",
            strength=strength,
            seed=42,
        )
        filename = f"variations/strength_{strength:.1f}.png"
        variation.save(filename)
        print(f"  Saved: {filename}")

    # Style transfer examples
    print("\n🎨 Applying different styles...")
    styles = [
        "watercolor painting",
        "pixel art retro",
        "japanese ukiyo-e woodblock print",
        "cyberpunk neon",
    ]

    for style in styles:
        styled = style_transfer(pipe, base_image, style, strength=0.7)
        safe_name = style.replace(" ", "_")[:20]
        filename = f"variations/style_{safe_name}.png"
        styled.save(filename)
        print(f"  Saved: {filename} (style: {style})")

    print("\n✅ All variations complete! Check the 'variations/' folder.")
    print("\n💡 Key takeaways:")
    print("  • strength=0.3: Subtle changes, keeps most of original")
    print("  • strength=0.7: Good for style transfer")
    print("  • strength=0.9+: Almost completely new image")
```

**Run it:**
```bash
python step3_image_variations.py
```

**Expected output:**
```
============================================================
Step 3: Image Variation Pipeline
============================================================

🎨 Generating base image...
  Base image saved to: variations/base_image.png

📦 Loading img2img pipeline...
Img2Img pipeline loaded on cuda

🔄 Creating strength comparison...
  Saved: variations/strength_0.3.png
  Saved: variations/strength_0.5.png
  Saved: variations/strength_0.7.png
  Saved: variations/strength_0.9.png

🎨 Applying different styles...
  Saved: variations/style_watercolor_painting.png (style: watercolor painting)
  Saved: variations/style_pixel_art_retro.png (style: pixel art retro)
  ...

✅ All variations complete! Check the 'variations/' folder.
```

**🎉 Milestone 3 Complete!** You can now create image variations and apply style transfer.

---

## Step 4: Text-to-Video Generation

Now let's generate short video clips from text prompts.

Create a file called `step4_text_to_video.py`:

```python
"""step4_text_to_video.py - Text-to-video generation demo."""

import torch
from diffusers import DiffusionPipeline, TextToVideoSDPipeline
from diffusers.utils import export_to_video, export_to_gif
import imageio
import numpy as np
import os


def load_t2v_pipeline() -> TextToVideoSDPipeline:
    """
    Load a text-to-video model.

    We use 'damo-vilab/text-to-video-ms-1.7b' — a lightweight T2V model
    that can run on consumer GPUs (needs ~8GB VRAM).
    """
    model_id = "damo-vilab/text-to-video-ms-1.7b"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading text-to-video model on {device}...")
    print("(First run will download ~3.5GB)")

    pipe = TextToVideoSDPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # Memory optimization
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    print("Text-to-video model loaded!")
    return pipe


def generate_video(
    pipe: TextToVideoSDPipeline,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted",
    num_frames: int = 16,
    num_inference_steps: int = 25,
    guidance_scale: float = 9.0,
    height: int = 256,
    width: int = 256,
    seed: int = None,
) -> list:
    """
    Generate a video from a text prompt.

    Args:
        pipe: The T2V pipeline.
        prompt: Text description of the video.
        negative_prompt: What to avoid.
        num_frames: Number of frames to generate (more = longer video).
        num_inference_steps: Denoising steps per frame.
        guidance_scale: Prompt adherence strength.
        height: Frame height (lower = faster, less VRAM).
        width: Frame width (lower = faster, less VRAM).
        seed: Random seed for reproducibility.

    Returns:
        List of PIL Images (video frames).
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    print(f"Generating video: '{prompt}'")
    print(f"  Frames: {num_frames}, Steps: {num_inference_steps}")
    print(f"  Resolution: {width}x{height}")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )

    frames = result.frames[0]
    print(f"  Generated {len(frames)} frames!")
    return frames


def save_video(frames: list, output_path: str, fps: int = 8):
    """
    Save frames as a video file.

    Args:
        frames: List of PIL Images or numpy arrays.
        output_path: Path to save (supports .mp4, .gif).
        fps: Frames per second.
    """
    if output_path.endswith(".gif"):
        export_to_gif(frames, output_path)
    else:
        export_to_video(frames, output_path, fps=fps)

    print(f"  Video saved to: {output_path}")


def create_video_with_interpolation(
    pipe: TextToVideoSDPipeline,
    prompt: str,
    num_frames: int = 16,
    seed: int = 42,
) -> list:
    """
    Generate a video and apply simple frame interpolation for smoother motion.

    This is a basic technique — advanced methods use optical flow.
    """
    # Generate base frames
    frames = generate_video(pipe, prompt, num_frames=num_frames, seed=seed)

    # Simple frame blending interpolation (2x frame count)
    interpolated = []
    for i in range(len(frames) - 1):
        interpolated.append(frames[i])
        # Create a blended frame between consecutive frames
        frame1 = np.array(frames[i]).astype(float)
        frame2 = np.array(frames[i + 1]).astype(float)
        blended = ((frame1 + frame2) / 2).astype(np.uint8)
        from PIL import Image
        interpolated.append(Image.fromarray(blended))
    interpolated.append(frames[-1])

    print(f"  Interpolated: {len(frames)} → {len(interpolated)} frames")
    return interpolated


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Text-to-Video Generation")
    print("=" * 60)

    os.makedirs("videos", exist_ok=True)

    # Load model
    pipe = load_t2v_pipeline()

    # Example 1: Simple video
    print("\n🎬 Example 1: Nature scene")
    frames = generate_video(
        pipe,
        prompt="A waterfall flowing into a crystal clear pool, nature, cinematic",
        num_frames=16,
        seed=42,
    )
    save_video(frames, "videos/waterfall.mp4", fps=8)
    save_video(frames, "videos/waterfall.gif", fps=8)

    # Example 2: Action scene
    print("\n🎬 Example 2: Action scene")
    frames = generate_video(
        pipe,
        prompt="A rocket launching into space with fire and smoke, dramatic",
        num_frames=16,
        seed=123,
    )
    save_video(frames, "videos/rocket.mp4", fps=8)

    # Example 3: With interpolation
    print("\n🎬 Example 3: Smooth video with interpolation")
    smooth_frames = create_video_with_interpolation(
        pipe,
        prompt="Ocean waves crashing on a beach at sunset, peaceful",
        num_frames=16,
        seed=77,
    )
    save_video(smooth_frames, "videos/ocean_smooth.mp4", fps=16)

    print("\n✅ All videos generated! Check the 'videos/' folder.")
    print("\n💡 Tips for better videos:")
    print("  • Use descriptive prompts with motion words")
    print("  • 16 frames at 8fps = 2 second clip")
    print("  • Lower resolution (256x256) for testing, higher for quality")
    print("  • Higher guidance_scale (9-12) works well for videos")
```

**Run it:**
```bash
python step4_text_to_video.py
```

**Expected output:**
```
============================================================
Step 4: Text-to-Video Generation
============================================================

Loading text-to-video model on cuda...
Text-to-video model loaded!

🎬 Example 1: Nature scene
Generating video: 'A waterfall flowing into a crystal clear pool, nature, cinematic'
  Frames: 16, Steps: 25
  Resolution: 256x256
  Generated 16 frames!
  Video saved to: videos/waterfall.mp4
  Video saved to: videos/waterfall.gif

🎬 Example 2: Action scene
  ...

✅ All videos generated! Check the 'videos/' folder.
```

**🎉 Milestone 4 Complete!** You can now generate videos from text.

---

## Step 5: Building the Gradio UI

Let's create a web interface to interact with our multi-modal agent.

Create a file called `step5_gradio_ui.py`:

```python
"""step5_gradio_ui.py - Gradio web UI for the multi-modal generation agent."""

import torch
import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    TextToVideoSDPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
import os
import tempfile


# ==================== Model Loading ====================

# Global variables for loaded models
t2i_pipe = None
img2img_pipe = None
t2v_pipe = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SCHEDULERS = {
    "Euler": EulerDiscreteScheduler,
    "Euler Ancestral": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
}


def get_t2i_pipeline():
    """Lazy-load the text-to-image pipeline."""
    global t2i_pipe
    if t2i_pipe is None:
        print("Loading text-to-image model...")
        t2i_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            safety_checker=None,
        ).to(DEVICE)
        if DEVICE == "cuda":
            t2i_pipe.enable_attention_slicing()
        print("Text-to-image model loaded!")
    return t2i_pipe


def get_img2img_pipeline():
    """Lazy-load the image-to-image pipeline."""
    global img2img_pipe
    if img2img_pipe is None:
        print("Loading image-to-image model...")
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            safety_checker=None,
        ).to(DEVICE)
        if DEVICE == "cuda":
            img2img_pipe.enable_attention_slicing()
        print("Image-to-image model loaded!")
    return img2img_pipe


def get_t2v_pipeline():
    """Lazy-load the text-to-video pipeline."""
    global t2v_pipe
    if t2v_pipe is None:
        print("Loading text-to-video model...")
        t2v_pipe = TextToVideoSDPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        if DEVICE == "cuda":
            t2v_pipe.enable_attention_slicing()
            t2v_pipe.enable_vae_slicing()
        print("Text-to-video model loaded!")
    return t2v_pipe


# ==================== Generation Functions ====================


def generate_image_fn(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    scheduler_name: str,
    seed: int,
):
    """Generate an image from text (Gradio callback)."""
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt!"

    pipe = get_t2i_pipeline()

    # Set scheduler
    if scheduler_name in SCHEDULERS:
        pipe.scheduler = SCHEDULERS[scheduler_name].from_config(pipe.scheduler.config)

    # Handle seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )

    image = result.images[0]
    info = f"✅ Generated! Seed: {seed} | Steps: {steps} | CFG: {guidance_scale} | Scheduler: {scheduler_name}"
    return image, info


def generate_variation_fn(
    input_image: Image.Image,
    prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    seed: int,
):
    """Generate an image variation (Gradio callback)."""
    if input_image is None:
        return None, "⚠️ Please upload an input image!"
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt!"

    pipe = get_img2img_pipeline()

    # Resize input
    input_image = input_image.resize((512, 512))

    # Handle seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=input_image,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps),
        generator=generator,
    )

    image = result.images[0]
    info = f"✅ Variation created! Seed: {seed} | Strength: {strength}"
    return image, info


def generate_video_fn(
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    """Generate a video from text (Gradio callback)."""
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt!"

    pipe = get_t2v_pipeline()

    # Handle seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        num_frames=int(num_frames),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        height=256,
        width=256,
        generator=generator,
    )

    frames = result.frames[0]

    # Save to a temporary file
    os.makedirs("outputs", exist_ok=True)
    video_path = f"outputs/video_{seed}.mp4"
    export_to_video(frames, video_path, fps=8)

    info = f"✅ Video generated! Seed: {seed} | Frames: {num_frames} | Steps: {steps}"
    return video_path, info


# ==================== Build UI ====================


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Multi-modal Generation Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎨 Multi-modal Generation Agent
            Generate images and videos from text using AI diffusion models.
            """
        )

        with gr.Tabs():
            # ---- Tab 1: Text to Image ----
            with gr.Tab("🖼️ Text to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful sunset over mountains...",
                            lines=3,
                        )
                        t2i_negative = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, deformed",
                            lines=2,
                        )
                        with gr.Row():
                            t2i_width = gr.Slider(256, 768, value=512, step=64, label="Width")
                            t2i_height = gr.Slider(256, 768, value=512, step=64, label="Height")
                        t2i_steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                        t2i_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                        t2i_scheduler = gr.Dropdown(
                            choices=list(SCHEDULERS.keys()),
                            value="Euler",
                            label="Scheduler",
                        )
                        t2i_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        t2i_btn = gr.Button("🎨 Generate Image", variant="primary")

                    with gr.Column(scale=1):
                        t2i_output = gr.Image(label="Generated Image", type="pil")
                        t2i_info = gr.Textbox(label="Info", interactive=False)

                t2i_btn.click(
                    fn=generate_image_fn,
                    inputs=[t2i_prompt, t2i_negative, t2i_width, t2i_height,
                            t2i_steps, t2i_cfg, t2i_scheduler, t2i_seed],
                    outputs=[t2i_output, t2i_info],
                )

            # ---- Tab 2: Image Variations ----
            with gr.Tab("🔄 Image Variations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        var_input = gr.Image(label="Input Image", type="pil")
                        var_prompt = gr.Textbox(
                            label="Prompt (describe desired output)",
                            placeholder="Same scene but in watercolor style...",
                            lines=2,
                        )
                        var_strength = gr.Slider(
                            0.1, 1.0, value=0.6, step=0.05,
                            label="Strength (higher = more change)",
                        )
                        var_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                        var_steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                        var_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        var_btn = gr.Button("🔄 Generate Variation", variant="primary")

                    with gr.Column(scale=1):
                        var_output = gr.Image(label="Output Image", type="pil")
                        var_info = gr.Textbox(label="Info", interactive=False)

                var_btn.click(
                    fn=generate_variation_fn,
                    inputs=[var_input, var_prompt, var_strength, var_cfg, var_steps, var_seed],
                    outputs=[var_output, var_info],
                )

            # ---- Tab 3: Text to Video ----
            with gr.Tab("🎬 Text to Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A rocket launching into space...",
                            lines=3,
                        )
                        t2v_negative = gr.Textbox(
                            label="Negative Prompt",
                            value="low quality, blurry, distorted",
                            lines=2,
                        )
                        t2v_frames = gr.Slider(8, 32, value=16, step=4, label="Number of Frames")
                        t2v_steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                        t2v_cfg = gr.Slider(1.0, 20.0, value=9.0, step=0.5, label="Guidance Scale")
                        t2v_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        t2v_btn = gr.Button("🎬 Generate Video", variant="primary")

                    with gr.Column(scale=1):
                        t2v_output = gr.Video(label="Generated Video")
                        t2v_info = gr.Textbox(label="Info", interactive=False)

                t2v_btn.click(
                    fn=generate_video_fn,
                    inputs=[t2v_prompt, t2v_negative, t2v_frames, t2v_steps, t2v_cfg, t2v_seed],
                    outputs=[t2v_output, t2v_info],
                )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Use detailed prompts for better results
            - Start with low steps (15-20) for quick previews
            - Increase steps (30-50) for final quality
            - Set a seed for reproducible results
            - Negative prompts help avoid common artifacts
            """
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-modal Generation Agent - Web UI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Models will be loaded on first use (may take a moment).")
    print()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for a public link
    )
```

**Run it:**
```bash
python step5_gradio_ui.py
```

**Expected output:**
```
============================================================
Multi-modal Generation Agent - Web UI
============================================================
Device: cuda
Models will be loaded on first use (may take a moment).

Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

Open http://localhost:7860 in your browser to use the UI!

**🎉 Milestone 5 Complete!** You have a full web UI for multi-modal generation.

---

## Step 6: Complete Multi-modal Agent

Finally, let's wrap everything into a clean, reusable agent class.

Create a file called `step6_multimodal_agent.py`:

```python
"""step6_multimodal_agent.py - Complete Multi-modal Generation Agent."""

import torch
import os
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    TextToVideoSDPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from diffusers.utils import export_to_video, export_to_gif


# ==================== Configuration ====================


@dataclass
class AgentConfig:
    """Configuration for the Multi-modal Agent."""

    # Model IDs
    t2i_model: str = "runwayml/stable-diffusion-v1-5"
    t2v_model: str = "damo-vilab/text-to-video-ms-1.7b"

    # Default generation parameters
    default_steps: int = 25
    default_guidance_scale: float = 7.5
    default_width: int = 512
    default_height: int = 512
    default_negative_prompt: str = "blurry, low quality, distorted, deformed"

    # Video defaults
    default_num_frames: int = 16
    default_video_guidance: float = 9.0
    default_video_height: int = 256
    default_video_width: int = 256
    default_fps: int = 8

    # System
    output_dir: str = "agent_outputs"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    enable_memory_optimization: bool = True


# ==================== Agent ====================


class MultimodalAgent:
    """
    A multi-modal generation agent that can create images and videos from text.

    Features:
    - Text-to-Image generation with parameter tuning
    - Image-to-Image variations and style transfer
    - Text-to-Video generation
    - Multiple scheduler support
    - Reproducible generation with seeds
    """

    SCHEDULERS = {
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm++_2m": DPMSolverMultistepScheduler,
        "ddim": DDIMScheduler,
    }

    def __init__(self, config: AgentConfig = None):
        """Initialize the agent with configuration."""
        self.config = config or AgentConfig()

        # Determine device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Lazy-loaded pipelines
        self._t2i_pipe = None
        self._img2img_pipe = None
        self._t2v_pipe = None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"🤖 Multi-modal Agent initialized")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.config.output_dir}/")

    # ---- Pipeline Management ----

    @property
    def t2i_pipe(self) -> StableDiffusionPipeline:
        """Get or load text-to-image pipeline."""
        if self._t2i_pipe is None:
            print("📦 Loading text-to-image model...")
            self._t2i_pipe = StableDiffusionPipeline.from_pretrained(
                self.config.t2i_model,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
            if self.config.enable_memory_optimization and self.device == "cuda":
                self._t2i_pipe.enable_attention_slicing()
            print("   ✓ Text-to-image ready!")
        return self._t2i_pipe

    @property
    def img2img_pipe(self) -> StableDiffusionImg2ImgPipeline:
        """Get or load image-to-image pipeline."""
        if self._img2img_pipe is None:
            print("📦 Loading image-to-image model...")
            self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.t2i_model,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
            if self.config.enable_memory_optimization and self.device == "cuda":
                self._img2img_pipe.enable_attention_slicing()
            print("   ✓ Image-to-image ready!")
        return self._img2img_pipe

    @property
    def t2v_pipe(self) -> TextToVideoSDPipeline:
        """Get or load text-to-video pipeline."""
        if self._t2v_pipe is None:
            print("📦 Loading text-to-video model...")
            self._t2v_pipe = TextToVideoSDPipeline.from_pretrained(
                self.config.t2v_model,
                torch_dtype=self.dtype,
            ).to(self.device)
            if self.config.enable_memory_optimization and self.device == "cuda":
                self._t2v_pipe.enable_attention_slicing()
                self._t2v_pipe.enable_vae_slicing()
            print("   ✓ Text-to-video ready!")
        return self._t2v_pipe

    def set_scheduler(self, pipeline, scheduler_name: str):
        """Set the scheduler for a pipeline."""
        if scheduler_name not in self.SCHEDULERS:
            available = ", ".join(self.SCHEDULERS.keys())
            raise ValueError(f"Unknown scheduler '{scheduler_name}'. Available: {available}")
        pipeline.scheduler = self.SCHEDULERS[scheduler_name].from_config(
            pipeline.scheduler.config
        )

    # ---- Text-to-Image ----

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        guidance_scale: float = None,
        scheduler: str = "euler",
        seed: int = None,
        save: bool = True,
        filename: str = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of desired image.
            negative_prompt: What to avoid (default from config).
            width: Image width (default from config).
            height: Image height (default from config).
            steps: Denoising steps (default from config).
            guidance_scale: CFG scale (default from config).
            scheduler: Scheduler name ('euler', 'euler_a', 'dpm++_2m', 'ddim').
            seed: Random seed (None = random).
            save: Whether to save the image to disk.
            filename: Custom filename (auto-generated if None).

        Returns:
            Generated PIL Image.
        """
        # Apply defaults
        negative_prompt = negative_prompt or self.config.default_negative_prompt
        width = width or self.config.default_width
        height = height or self.config.default_height
        steps = steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_guidance_scale

        # Set scheduler
        self.set_scheduler(self.t2i_pipe, scheduler)

        # Handle seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n🎨 Generating image...")
        print(f"   Prompt: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"   Size: {width}x{height} | Steps: {steps} | CFG: {guidance_scale} | Seed: {seed}")

        # Generate
        result = self.t2i_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]

        # Save
        if save:
            if filename is None:
                filename = f"img_{seed}.png"
            path = os.path.join(self.config.output_dir, filename)
            image.save(path)
            print(f"   💾 Saved: {path}")

        print("   ✅ Done!")
        return image

    # ---- Image Variations ----

    def create_variation(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.6,
        guidance_scale: float = None,
        steps: int = None,
        seed: int = None,
        save: bool = True,
        filename: str = None,
    ) -> Image.Image:
        """
        Create a variation of an existing image.

        Args:
            image: Input PIL Image.
            prompt: Description of desired output.
            strength: How much to change (0.0-1.0).
            guidance_scale: CFG scale.
            steps: Denoising steps.
            seed: Random seed.
            save: Whether to save.
            filename: Custom filename.

        Returns:
            Modified PIL Image.
        """
        guidance_scale = guidance_scale or self.config.default_guidance_scale
        steps = steps or self.config.default_steps

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Resize input
        image = image.resize((self.config.default_width, self.config.default_height))

        print(f"\n🔄 Creating variation...")
        print(f"   Prompt: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"   Strength: {strength} | Seed: {seed}")

        result = self.img2img_pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        output = result.images[0]

        if save:
            if filename is None:
                filename = f"var_{seed}.png"
            path = os.path.join(self.config.output_dir, filename)
            output.save(path)
            print(f"   💾 Saved: {path}")

        print("   ✅ Done!")
        return output

    def style_transfer(
        self,
        image: Image.Image,
        style: str,
        strength: float = 0.65,
        seed: int = None,
        save: bool = True,
    ) -> Image.Image:
        """
        Apply a style to an image.

        Args:
            image: Input PIL Image.
            style: Style description (e.g., "watercolor", "anime", "oil painting").
            strength: How strongly to apply style.
            seed: Random seed.
            save: Whether to save.

        Returns:
            Styled PIL Image.
        """
        prompt = f"({style} style), high quality, detailed, masterpiece"
        filename = f"style_{style.replace(' ', '_')[:15]}_{seed or 'rand'}.png"
        return self.create_variation(
            image, prompt, strength=strength, seed=seed, save=save, filename=filename
        )

    # ---- Text-to-Video ----

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_frames: int = None,
        steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        save: bool = True,
        filename: str = None,
        output_format: str = "mp4",
    ) -> List[Image.Image]:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the video.
            negative_prompt: What to avoid.
            num_frames: Number of frames (default from config).
            steps: Denoising steps.
            guidance_scale: CFG scale.
            seed: Random seed.
            save: Whether to save.
            filename: Custom filename.
            output_format: 'mp4' or 'gif'.

        Returns:
            List of PIL Images (video frames).
        """
        negative_prompt = negative_prompt or self.config.default_negative_prompt
        num_frames = num_frames or self.config.default_num_frames
        steps = steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_video_guidance

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n🎬 Generating video...")
        print(f"   Prompt: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"   Frames: {num_frames} | Steps: {steps} | CFG: {guidance_scale} | Seed: {seed}")

        result = self.t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=self.config.default_video_height,
            width=self.config.default_video_width,
            generator=generator,
        )

        frames = result.frames[0]

        if save:
            if filename is None:
                filename = f"vid_{seed}.{output_format}"
            path = os.path.join(self.config.output_dir, filename)

            if output_format == "gif":
                export_to_gif(frames, path)
            else:
                export_to_video(frames, path, fps=self.config.default_fps)
            print(f"   💾 Saved: {path}")

        print(f"   ✅ Done! ({len(frames)} frames)")
        return frames

    # ---- Batch Operations ----

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate multiple images from a list of prompts.

        Args:
            prompts: List of text prompts.
            **kwargs: Additional arguments passed to generate_image().

        Returns:
            List of generated PIL Images.
        """
        print(f"\n📦 Batch generation: {len(prompts)} images")
        images = []
        for i, prompt in enumerate(prompts):
            print(f"\n--- Image {i+1}/{len(prompts)} ---")
            img = self.generate_image(prompt, filename=f"batch_{i}.png", **kwargs)
            images.append(img)
        print(f"\n✅ Batch complete! {len(images)} images generated.")
        return images

    # ---- Utility ----

    def unload_models(self):
        """Free GPU memory by unloading all models."""
        self._t2i_pipe = None
        self._img2img_pipe = None
        self._t2v_pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("🗑️  All models unloaded, memory freed.")

    def status(self):
        """Print agent status."""
        print(f"\n{'='*50}")
        print(f"🤖 Multi-modal Agent Status")
        print(f"{'='*50}")
        print(f"  Device: {self.device}")
        print(f"  T2I loaded: {'✓' if self._t2i_pipe else '✗'}")
        print(f"  Img2Img loaded: {'✓' if self._img2img_pipe else '✗'}")
        print(f"  T2V loaded: {'✓' if self._t2v_pipe else '✗'}")
        if self.device == "cuda":
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU Memory: {mem_used:.1f} / {mem_total:.1f} GB")
        print(f"  Output dir: {self.config.output_dir}/")
        print(f"{'='*50}\n")


# ==================== Main Demo ====================


if __name__ == "__main__":
    print("=" * 60)
    print("   Multi-modal Generation Agent - Full Demo")
    print("=" * 60)

    # Create agent with default config
    agent = MultimodalAgent()
    agent.status()

    # --- Demo 1: Generate Images ---
    print("\n" + "─" * 40)
    print("Demo 1: Text-to-Image Generation")
    print("─" * 40)

    image1 = agent.generate_image(
        prompt="A cozy coffee shop interior with warm lighting, digital art style",
        steps=25,
        seed=42,
    )

    image2 = agent.generate_image(
        prompt="A futuristic city skyline at night with neon lights, cyberpunk",
        steps=25,
        guidance_scale=8.0,
        scheduler="dpm++_2m",
        seed=123,
    )

    # --- Demo 2: Image Variations ---
    print("\n" + "─" * 40)
    print("Demo 2: Image Variations")
    print("─" * 40)

    variation = agent.create_variation(
        image=image1,
        prompt="A cozy coffee shop interior, watercolor painting style",
        strength=0.6,
        seed=42,
    )

    styled = agent.style_transfer(
        image=image2,
        style="studio ghibli anime",
        strength=0.7,
        seed=42,
    )

    # --- Demo 3: Text-to-Video ---
    print("\n" + "─" * 40)
    print("Demo 3: Text-to-Video Generation")
    print("─" * 40)

    frames = agent.generate_video(
        prompt="A beautiful aurora borealis dancing in the night sky, timelapse",
        num_frames=16,
        steps=25,
        seed=42,
    )

    # --- Demo 4: Batch Generation ---
    print("\n" + "─" * 40)
    print("Demo 4: Batch Generation")
    print("─" * 40)

    prompts = [
        "A red panda eating bamboo, wildlife photography",
        "An ancient temple overgrown with vines, fantasy art",
        "A steampunk airship flying over clouds, illustration",
    ]
    batch_images = agent.generate_batch(prompts, steps=20, seed=42)

    # Final status
    agent.status()

    print("\n🎉 All demos complete!")
    print(f"   Check '{agent.config.output_dir}/' for generated outputs.")
```

**Run it:**
```bash
python step6_multimodal_agent.py
```

**Expected output:**
```
============================================================
   Multi-modal Generation Agent - Full Demo
============================================================
🤖 Multi-modal Agent initialized
   Device: cuda
   Output: agent_outputs/

==================================================
🤖 Multi-modal Agent Status
==================================================
  Device: cuda
  T2I loaded: ✗
  Img2Img loaded: ✗
  T2V loaded: ✗
  GPU Memory: 0.0 / 10.0 GB
  Output dir: agent_outputs/
==================================================

──────────────────────────────────
Demo 1: Text-to-Image Generation
──────────────────────────────────
📦 Loading text-to-image model...
   ✓ Text-to-image ready!

🎨 Generating image...
   Prompt: 'A cozy coffee shop interior with warm lighting, digital art style'
   Size: 512x512 | Steps: 25 | CFG: 7.5 | Seed: 42
   💾 Saved: agent_outputs/img_42.png
   ✅ Done!
...
```

**🎉 Milestone 6 Complete!** You have a full, production-ready Multi-modal Generation Agent!

---

## Common Pitfalls & Troubleshooting

### 1. Out of Memory (OOM) Errors

**Problem:** `CUDA out of memory`

**Solutions:**
```python
# Solution 1: Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()  # Moves layers to CPU when not in use

# Solution 2: Use smaller resolution
image = pipe(prompt, height=384, width=384).images[0]

# Solution 3: Use float16
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Solution 4: Clear cache between generations
torch.cuda.empty_cache()
```

### 2. Black or Blank Images

**Problem:** Generated images are completely black or blank.

**Solutions:**
```python
# Often caused by float16 overflow on certain GPUs
# Solution 1: Use float32 for VAE
pipe.vae = pipe.vae.to(dtype=torch.float32)

# Solution 2: Enable safe dtype autocast
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",  # Use fp16 variant if available
)
```

### 3. Slow Generation on CPU

**Problem:** Generation takes 10+ minutes.

**Solutions:**
```python
# Solution 1: Use fewer steps
result = pipe(prompt, num_inference_steps=10)  # Fast but lower quality

# Solution 2: Use a fast scheduler
from diffusers import LCMScheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
result = pipe(prompt, num_inference_steps=4)  # Very fast!

# Solution 3: Use smaller resolution
result = pipe(prompt, height=256, width=256)

# Solution 4: Use Google Colab (free GPU)
```

### 4. Model Download Fails

**Problem:** Connection timeout or download errors.

**Solutions:**
```bash
# Solution 1: Set a longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Solution 2: Use huggingface-cli to download manually
pip install huggingface_hub
huggingface-cli download runwayml/stable-diffusion-v1-5

# Solution 3: Check disk space (models are 2-7GB each)
# On Windows:
# Get-PSDrive C
# On Linux:
# df -h
```

### 5. Import Errors

**Problem:** `ModuleNotFoundError`

**Solutions:**
```bash
# Make sure you're in the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Reinstall everything
pip install --upgrade diffusers transformers accelerate torch
```

### 6. Poor Image Quality

**Problem:** Images look bad, blurry, or distorted.

**Solutions:**
```python
# Tip 1: Use negative prompts
negative = "blurry, bad quality, distorted, deformed, ugly, low resolution"

# Tip 2: Increase steps
result = pipe(prompt, num_inference_steps=40)

# Tip 3: Use appropriate guidance scale (7-9 is usually best)
result = pipe(prompt, guidance_scale=7.5)

# Tip 4: Add quality boosters to prompt
prompt = "A beautiful landscape, high quality, detailed, 4k, professional photography"

# Tip 5: Try a different scheduler
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

### 7. Gradio UI Won't Start

**Problem:** Port already in use or connection refused.

**Solutions:**
```python
# Solution 1: Use a different port
demo.launch(server_port=7861)

# Solution 2: Kill process using the port
# Windows: netstat -ano | findstr :7860, then taskkill /PID <pid> /F
# Linux: lsof -i :7860, then kill <pid>

# Solution 3: Enable sharing for remote access
demo.launch(share=True)
```

### 8. Video Generation Issues

**Problem:** Video is too short, choppy, or has artifacts.

**Solutions:**
```python
# Tip 1: Use more frames for longer videos
frames = pipe(prompt, num_frames=24).frames[0]  # 3 seconds at 8fps

# Tip 2: Higher guidance for videos
frames = pipe(prompt, guidance_scale=12.0).frames[0]

# Tip 3: Add motion keywords to prompt
prompt = "A river flowing through a forest, smooth motion, cinematic, steady camera"

# Tip 4: Post-process with frame interpolation (see Step 4)
```

---

## Project Structure

After completing all steps, your project should look like this:

```
multimodal-agent/
├── venv/                      # Virtual environment
├── verify_setup.py            # Step 0: Environment verification
├── step1_basic_t2i.py         # Step 1: Basic text-to-image
├── step2_parameter_tuning.py  # Step 2: Parameter exploration
├── step3_image_variations.py  # Step 3: Image variations
├── step4_text_to_video.py     # Step 4: Text-to-video
├── step5_gradio_ui.py         # Step 5: Web UI
├── step6_multimodal_agent.py  # Step 6: Complete agent
├── output_step1.png           # Generated outputs
├── comparisons/               # Parameter comparison images
├── variations/                # Image variations
├── videos/                    # Generated videos
└── agent_outputs/             # Final agent outputs
```

---

## Next Steps

Once you've completed this project, consider:

1. **Try different models:** Replace `stable-diffusion-v1-5` with `stabilityai/stable-diffusion-xl-base-1.0` for higher quality
2. **Add ControlNet:** Guide generation with edge maps, depth maps, or poses
3. **Implement LoRA:** Fine-tune models on specific styles with minimal compute
4. **Add inpainting:** Edit specific regions of an image
5. **Improve video:** Experiment with AnimateDiff for better temporal consistency
6. **Deploy:** Package your Gradio app as a Docker container for deployment

---

*Congratulations! You've built a complete Multi-modal Generation Agent! 🎉*
