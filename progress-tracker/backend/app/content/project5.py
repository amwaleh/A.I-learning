PROJECT_5 = {
    "number": 5,
    "title": "Build a Multi-modal Generation Agent",
    "description": "Build a Multi-modal Generation Agent",
    "topics": [
        {
            "title": "Text-to-Image",
            "content": """\
## Text-to-Image Generation

Text-to-image generation has undergone a dramatic evolution over the past several years. Early approaches relied on **Generative Adversarial Networks (GANs)**, where a generator and discriminator competed to produce realistic images. Models like StyleGAN achieved impressive quality for specific domains (faces, landscapes), but GANs suffered from **mode collapse**, training instability, and struggled with diverse, open-domain generation from arbitrary text prompts.

The breakthrough came with **diffusion models**, which learn to reverse a gradual noising process. Unlike GANs, diffusion models offer stable training, excellent mode coverage, and naturally support conditioning on text embeddings. This made them the dominant paradigm by 2022-2023.

### Key Models in the Evolution

| Model | Year | Key Innovation |
|-------|------|----------------|
| DALL-E (OpenAI) | 2021 | Discrete VAE + autoregressive transformer |
| DALL-E 2 | 2022 | CLIP embeddings + diffusion prior + decoder |
| Stable Diffusion | 2022 | Latent diffusion, open-source, community-driven |
| Midjourney | 2022+ | Aesthetic fine-tuning, artistic quality |
| SDXL | 2023 | Dual U-Net, refiner model, higher resolution |
| SD3 / Flux | 2024 | DiT backbone (Diffusion Transformer), flow matching |

### Why Diffusion Models Won

```
GANs:        Generator ←→ Discriminator (adversarial, unstable)
Diffusion:   x_T → x_{T-1} → ... → x_0  (iterative denoising, stable)
```

1. **Training stability** — no adversarial min-max game; just predict noise
2. **Mode coverage** — the stochastic process naturally explores the full distribution
3. **Conditioning flexibility** — text, sketches, depth maps can all guide generation via cross-attention
4. **Scalability** — larger models consistently improve quality (unlike GANs which plateau)

Today, state-of-the-art systems combine large language model text encoders (T5, CLIP) with powerful diffusion backbones to generate photorealistic images from natural language descriptions. Understanding this pipeline is essential for building multi-modal generation agents.
""",
            "children": [
                {
                    "title": "Data preparation",
                    "content": """\
## Data Preparation for Text-to-Image Models

Training text-to-image models requires massive, high-quality datasets of **image-text pairs**. The quality and scale of your data directly determines model capability.

### LAION-5B: The Foundation Dataset

**LAION-5B** is the largest publicly available image-text dataset, containing approximately 5.85 billion image-text pairs scraped from the web. It was used to train Stable Diffusion and many other open-source models. The pipeline for building such a dataset follows this flow:

```
Web Crawl (Common Crawl)
    → Extract <img> tags + alt text
    → Download images
    → Filter by resolution (≥ 256×256)
    → Compute CLIP similarity scores
    → Filter low-quality pairs (CLIP score < 0.28)
    → NSFW detection & filtering
    → Deduplicate
    → Final dataset
```

### CLIP Score Filtering

Not every image-alt-text pair is meaningful. **CLIP score** measures the cosine similarity between the CLIP image embedding and text embedding. Pairs with low CLIP scores indicate poor alignment (e.g., stock photo watermarks, irrelevant alt text). A typical threshold is **0.28** for English pairs.

### Caption Quality

Raw alt-text from the web is often noisy — short, generic, or keyword-stuffed. Modern pipelines use **re-captioning** with vision-language models (e.g., CogVLM, LLaVA) to generate detailed, descriptive captions:

```
Original alt text: "IMG_2047.jpg"
Re-captioned:      "A golden retriever running through a sunlit meadow
                    with wildflowers, photographed from a low angle"
```

### Resolution Bucketing

Rather than resizing all images to a fixed square, **resolution bucketing** groups images by their native aspect ratio into predefined buckets (e.g., 512×768, 768×512, 640×640). This preserves composition and avoids distortion. During training, each batch is drawn from a single bucket for uniform tensor shapes.

### NSFW and Safety Filtering

Production datasets require multi-layered safety filtering: NSFW classifiers (nudity, violence), watermark detectors, personally identifiable information (PII) removal, and content policy enforcement. This is critical for responsible deployment.
""",
                },
                {
                    "title": "Diffusion architectures",
                    "content": """\
## Diffusion Model Architectures

The architecture of a diffusion model defines how it processes noisy inputs and text conditioning to predict clean outputs. The field has evolved from U-Net to Transformer-based designs.

### Latent Diffusion (LDM)

Instead of diffusing in pixel space (expensive at high resolutions), **Latent Diffusion Models** first encode images into a compressed latent space using a pre-trained VAE:

```mermaid
graph LR
    A[Text Prompt] --> B[Text Encoder<br>CLIP / T5]
    B --> C[Cross-Attn<br>Conditioning]
    D[Image or z_T] --> E[VAE Encode]
    E --> F[U-Net / DiT Backbone<br>Denoising Network]
    C --> F
    F --> G[VAE Decode]
    G --> H[Output Image]
```

The latent space is typically **8× downsampled** (a 512×512 image becomes 64×64×4 latents), making diffusion computationally tractable.

### U-Net Backbone

The classic U-Net architecture features an **encoder-decoder** with skip connections. At each resolution level, **ResNet blocks** process spatial features, **self-attention blocks** capture global context, and **cross-attention blocks** inject text conditioning:

```python
# Simplified cross-attention for text conditioning
Q = proj_q(latent_features)     # from image latents
K = proj_k(text_embeddings)     # from CLIP/T5
V = proj_v(text_embeddings)
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

### SDXL Improvements

SDXL introduced a **dual U-Net** pipeline (base + refiner), larger cross-attention dimensions (2048), **two text encoders** (OpenCLIP ViT-bigG + CLIP ViT-L), and micro-conditioning (original resolution, crop coordinates) to improve quality.

### SD3 / Flux: The DiT Revolution

**Stable Diffusion 3** and **Flux** replaced the U-Net with a **Diffusion Transformer (DiT)**. Instead of convolutions, the backbone uses pure transformer blocks with **MMDiT (Multi-Modal DiT)** layers that jointly attend over image and text tokens. This architecture scales more predictably and leverages **flow matching** objectives for improved training dynamics.
""",
                },
                {
                    "title": "Diffusion training",
                    "content": """\
## Diffusion Model Training

Training a diffusion model involves teaching a neural network to **reverse a noising process**. The math is elegant and the training objective is surprisingly simple.

### Forward Process (Adding Noise)

Given a clean image `x₀`, the forward process gradually adds Gaussian noise over `T` timesteps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

With the reparameterization trick, we can jump directly to any timestep:

```
x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε,    where ε ~ N(0, I)
```

Here `ᾱ_t = ∏(1 - β_s)` is the cumulative noise schedule. At `t = T`, the image is pure noise.

### Noise Schedules

The schedule `β_t` controls how fast noise is added:

| Schedule | Description | Use Case |
|----------|-------------|----------|
| Linear | β increases linearly from 1e-4 to 0.02 | Original DDPM |
| Cosine | Smoother progression, preserves signal longer | Improved DDPM |
| Shifted cosine | Adjusted for latent diffusion | SD, SDXL |

### Training Objective (Simplified)

The model `ε_θ` predicts the noise added at timestep `t`. The loss is a simple **MSE**:

```python
# Training loop (simplified)
x_0 = load_image()                          # clean image
t = randint(0, T)                           # random timestep
epsilon = torch.randn_like(x_0)             # sample noise
x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon
epsilon_pred = model(x_t, t, text_emb)      # predict noise
loss = F.mse_loss(epsilon_pred, epsilon)     # simple MSE loss
```

### Classifier-Free Guidance (CFG)

During training, the text condition is randomly **dropped** (replaced with null/empty embedding) with probability ~10%. At inference, we combine conditional and unconditional predictions:

```
ε_guided = ε_uncond + w · (ε_cond - ε_uncond)
```

where `w` is the **guidance scale** (typically 7-15). Higher `w` increases prompt adherence at the cost of diversity. CFG is crucial — without it, generated images often ignore the text prompt.

### Prediction Targets

Models can predict noise (`ε`), the clean image (`x₀`), or velocity (`v = √ᾱ·ε - √(1-ᾱ)·x₀`). Velocity prediction is preferred in modern models as it provides more balanced gradients across timesteps.
""",
                },
                {
                    "title": "Diffusion sampling",
                    "content": """\
## Diffusion Sampling (Inference)

Sampling is the reverse process: starting from pure noise `x_T ~ N(0, I)` and iteratively denoising to produce a clean image `x_0`. The choice of **sampler** dramatically affects quality, speed, and determinism.

### DDPM Sampler (Original)

The original **Denoising Diffusion Probabilistic Model** sampler follows the learned reverse process step by step. It requires **1000 steps** and adds stochastic noise at each step, producing diverse but slow results:

```
x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z
```

### DDIM Sampler (Deterministic)

**Denoising Diffusion Implicit Models** reformulate the reverse process as a deterministic mapping (when η=0). Key advantage: the same initial noise always produces the same image, and it works with **as few as 20-50 steps**:

```
DDPM: 1000 steps, stochastic  → high quality, slow, varied
DDIM:   50 steps, deterministic → good quality, fast, reproducible
```

### DPM-Solver (ODE-Based)

Viewing diffusion as solving an **ordinary differential equation (ODE)**, DPM-Solver applies higher-order numerical methods. DPM-Solver++ achieves excellent quality in **15-25 steps**, making it a popular default:

Step comparison (512×512 image):

| Sampler | Steps | Time (s) | Quality |
|---------|-------|----------|---------|
| DDPM | 1000 | ~60 | ★★★★★ |
| DDIM | 50 | ~3 | ★★★★ |
| DPM-Solver++ | 20 | ~1.2 | ★★★★★ |
| Euler | 25 | ~1.5 | ★★★★ |
| Euler a | 25 | ~1.5 | ★★★★ |

### Euler and Karras Schedulers

**Euler** is a simple first-order ODE solver that is fast and effective. The **Karras** noise schedule modification (from the EDM paper) adjusts the sigma spacing to concentrate steps where they matter most, improving quality without extra compute.

### CFG Scale Effect

The **classifier-free guidance scale** `w` controls the quality-diversity tradeoff at inference:

- `w = 1.0` — no guidance, diverse but off-prompt
- `w = 7.5` — balanced (common default)
- `w = 15+` — strong adherence, but oversaturated, artifacts appear

Modern approaches like **dynamic CFG** (reducing guidance at early/late steps) and **PAG (Perturbed Attention Guidance)** aim to improve this tradeoff further.
""",
                },
                {
                    "title": "Evaluation",
                    "content": """\
## Evaluation of Text-to-Image Models

Evaluating generative image models is notoriously difficult because quality is subjective and multi-dimensional. The field uses a combination of **automated metrics** and **human preference studies**.

### FID (Fréchet Inception Distance)

FID is the most widely used automated metric. It compares the **distribution** of generated images to real images in Inception-v3 feature space:

```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2·(Σ_real · Σ_gen)^{1/2})
```

- Lower FID = generated distribution is closer to real distribution
- Typical values: FID < 10 is excellent, FID < 30 is good
- **Limitations**: FID measures distributional similarity, not per-image quality. It can miss mode collapse and doesn't assess text alignment.

### CLIP Score

**CLIP score** measures how well a generated image matches its text prompt by computing cosine similarity in CLIP embedding space:

```python
clip_score = cosine_similarity(
    clip.encode_image(generated_image),
    clip.encode_text(prompt)
)
```

Higher CLIP score indicates better text-image alignment. However, CLIP has biases — it may favor images with embedded text or stereotypical representations.

### Aesthetic Score Predictors

Models like **LAION Aesthetic Predictor** are trained on human aesthetic ratings to predict how visually appealing an image is, independent of prompt fidelity. These scores (typically 1-10 scale) are used both for evaluation and for filtering training data.

### Human Preference Studies

Automated metrics correlate imperfectly with human judgment. **ImageReward** trains a reward model on human preference data (pairwise comparisons of images for the same prompt). It captures nuanced quality dimensions that FID and CLIP miss:

```mermaid
graph TD
    A[Fidelity<br>FID] --> D[Overall Model Score]
    B[Alignment<br>CLIP Score] --> D
    C[Aesthetics<br>Human Pref] --> D
```

### GenAI-Bench

**GenAI-Bench** is a comprehensive benchmark that tests compositional text-to-image generation with complex prompts involving spatial relations, attribute binding, counting, and more. It provides a standardized leaderboard for comparing models on challenging, realistic prompts rather than simple descriptions.
""",
                },
            ],
        },
        {
            "title": "Text-to-Video",
            "content": """\
## Text-to-Video Generation

Text-to-video (T2V) generation extends text-to-image by adding the **temporal dimension** — producing coherent sequences of frames that tell a visual story from a text description. This is one of the most challenging problems in generative AI.

### Core Challenges

Video generation is fundamentally harder than image generation for several reasons:

```
Image: Generate 1 frame     → ~1M pixels (512×512×3)
Video: Generate 120 frames  → ~120M pixels (512×512×3×120)
```

**Challenges:**

```mermaid
graph TD
    A[Video Generation Challenges]
    A --> B[1. Temporal consistency - no flickering]
    A --> C[2. Motion coherence - physics-aware movement]
    A --> D[3. Compute: ~100-1000× more than images]
    A --> E[4. Training data: high-quality video+text]
    A --> F[5. Evaluation: no standard metrics yet]
```

A single inconsistent frame is immediately noticeable in video, making temporal coherence the primary technical challenge. Objects must maintain identity, lighting must stay consistent, and motion must follow plausible physics.

### Key Models

| Model | Organization | Key Features |
|-------|-------------|--------------|
| Sora | OpenAI | DiT backbone, long-duration, world simulation |
| Runway Gen-3 Alpha | Runway | High temporal consistency, fine control |
| Kling | Kuaishou | 3D spatiotemporal VAE, physics understanding |
| CogVideoX | Tsinghua/Zhipu | Open-source, 3D causal VAE, expert transformer |
| Veo 2 | Google DeepMind | High fidelity, cinematic quality |
| HunyuanVideo | Tencent | Open-source, dual-stream DiT |

### Architecture Paradigm Shift

Early T2V models adapted image U-Nets by inserting temporal attention layers. Modern approaches use **Diffusion Transformers (DiT)** that natively process spatiotemporal token sequences, treating video as a collection of spacetime patches:

```
Video → Patchify (space + time) → DiT Blocks → Unpatchify → Video
```

This mirrors the shift from CNNs to Transformers in NLP and vision, enabling better scaling and temporal modeling. The Sora technical report demonstrated that scaling DiT models produces emergent capabilities like consistent 3D scenes and realistic physics.
""",
            "children": [
                {
                    "title": "LDM and compression",
                    "content": """\
## Video Latent Diffusion and Compression

Running diffusion directly in pixel space for video is computationally prohibitive. A 5-second 720p video at 24fps contains over **150 million pixels**. **Video Variational Autoencoders (Video VAEs)** compress video into a manageable latent representation before diffusion.

### Video VAE Architecture

A Video VAE extends the image VAE with **temporal compression**. It encodes a video clip into a 3D latent tensor that is smaller in all dimensions:

```
Input Video:   [B, C, T, H, W]  = [1, 3, 64, 512, 512]
                                     ↓ Video VAE Encoder
Latent:        [B, c, t, h, w]  = [1, 16, 8, 64, 64]
                                     ↓ Diffusion in latent space
Denoised Lat:  [B, c, t, h, w]  = [1, 16, 8, 64, 64]
                                     ↓ Video VAE Decoder
Output Video:  [B, C, T, H, W]  = [1, 3, 64, 512, 512]

Compression: 8× temporal, 8× spatial height, 8× spatial width
Total compression: ~512× fewer elements
```

### Spatial vs. Temporal Compression

The VAE performs two types of downsampling:

- **Spatial compression** (height and width): Similar to image VAEs, uses strided convolutions or attention pooling — typically **8×** per spatial dimension
- **Temporal compression** (frames): Reduces the number of frames — typically **4-8×**, achieved via 3D convolutions with temporal stride

The balance matters: too much temporal compression loses motion detail; too little wastes compute during diffusion.

### Causal 3D VAE

A **causal 3D VAE** (used in CogVideoX, HunyuanVideo) processes frames in temporal order, where each frame's encoding depends only on current and past frames — never future frames. This enables:

```
Causal processing: frame_t depends on [frame_0, ..., frame_t]
                   (not frame_{t+1}, frame_{t+2}, ...)
```

```mermaid
graph TD
    A[Causal 3D VAE Benefits]
    A --> B[Streaming / autoregressive generation]
    A --> C[Variable-length video support]
    A --> D[Consistent with how video is consumed]
```

### Training the Video VAE

The VAE is trained separately from the diffusion model, using a combination of **reconstruction loss** (L1/L2 on decoded video), **perceptual loss** (LPIPS), **adversarial loss** (temporal discriminator), and **KL regularization** on the latent distribution. Getting the VAE right is critical — artifacts in the VAE propagate through the entire generation pipeline.
""",
                },
                {
                    "title": "Data preparation",
                    "content": """\
## Data Preparation for Text-to-Video Models

Video data preparation is significantly more complex than image data preparation. The temporal dimension introduces challenges around **motion quality**, **scene consistency**, and **caption granularity**.

### Video-Text Datasets

Large-scale video-text datasets are harder to curate than image-text pairs. Common sources include:

- **WebVid-10M** — 10M video-text pairs from stock footage sites
- **HD-VILA-100M** — 100M clips from YouTube with ASR captions
- **Panda-70M** — 70M high-quality clips with generated captions
- **InternVid** — 234M clips with detailed multi-modal captions

### Caption Quality Is Critical

Raw video captions (from metadata, ASR, or alt-text) are typically poor quality. Modern pipelines use **video captioning models** to generate detailed descriptions:

```mermaid
graph LR
    A[Raw Video<br>+ weak text] --> B[Video Captioning Model<br>e.g. InternVL, GPT-4V] --> C[Detailed Temporal<br>Description]
```

```
Example output:
"A woman in a red dress walks along a sunlit beach,
 the camera slowly panning right as waves gently
 crash on the shore. She pauses to look at the
 horizon before continuing."
```

Good captions describe **actions over time**, camera movements, and scene transitions — not just static scene descriptions.

### Scene Detection and Segmentation

Raw videos often contain multiple scenes, cuts, and transitions. Before training, videos must be segmented into **single-scene clips** using scene detection algorithms (PySceneDetect, TransNetV2):

```python
# Scene detection splits a video at cut boundaries
full_video = "interview_30min.mp4"
scenes = detect_scenes(full_video)
# → [0:00-0:15, 0:15-0:22, 0:22-0:45, ...]
# Each scene = one training clip
```

### Frame Sampling Strategies

Training doesn't always use every frame. Common strategies include:

- **Uniform sampling**: Select N frames evenly spaced (e.g., 16 frames from a 4-second clip)
- **Stride-based**: Every k-th frame (k=2,4,8) to cover longer temporal spans
- **FPS normalization**: Resample all videos to a standard FPS (e.g., 8 or 24 FPS)

### Temporal Annotation and Filtering

Videos are filtered for **motion quality** (removing static shots, excessive camera shake), **aesthetic quality** (resolution, lighting, compression artifacts), and **content safety**. Optical flow magnitude is often used to quantify motion and remove near-static clips.
""",
                },
                {
                    "title": "DiT architecture",
                    "content": """\
## Diffusion Transformer (DiT) Architecture for Video

The **Diffusion Transformer (DiT)** replaces the traditional U-Net backbone with a pure transformer architecture. Originally proposed for image generation, DiT has become the dominant architecture for video generation due to its superior scaling properties and natural handling of sequential data.

### Why Replace U-Net with Transformer?

```mermaid
graph TD
    subgraph UNet["U-Net Limitations"]
        U1[Fixed spatial structure<br>encoder-decoder]
        U2[Skip connections create<br>architecture rigidity]
        U3[Scaling is ad-hoc<br>wider? deeper? more attention?]
        U4[Temporal extension feels<br>bolted-on]
    end
    subgraph DiT["DiT Advantages"]
        D1[Uniform architecture<br>just transformer blocks]
        D2[Proven scaling laws<br>loss decreases predictably]
        D3[Native sequence modeling<br>space AND time as tokens]
        D4[Flexible input<br>variable resolution and duration]
    end
```

### Patchification: Video to Tokens

The first step in DiT is converting video latents into a sequence of tokens via **3D patchification**:

```python
# Video latent: [B, C, T, H, W] = [1, 16, 8, 64, 64]
# Patch size: (2, 2, 2) in (T, H, W)
# Number of tokens = (8/2) × (64/2) × (64/2) = 4 × 32 × 32 = 4096 tokens
# Each token: patch_dim = 16 × 2 × 2 × 2 = 128 → projected to hidden_dim
```

### Spatial-Temporal Attention Blocks

Each DiT block processes the token sequence with attention. Two common designs:

1. **Full 3D attention** — all tokens attend to all others (O(n²) where n = spatial × temporal). Maximum quality but expensive.
2. **Factored attention** — alternate between spatial-only and temporal-only attention within blocks. Reduces cost significantly:

```
DiT Block (Factored):
    Input tokens → LayerNorm → Spatial Self-Attention (within each frame)
                 → LayerNorm → Temporal Self-Attention (across frames at same position)
                 → LayerNorm → Cross-Attention (with text embeddings)
                 → LayerNorm → FFN (MLP)
                 → Output tokens
```

### RoPE for Positional Encoding

**Rotary Position Embeddings (RoPE)** encode 3D positions (t, h, w) by applying rotation matrices to query and key vectors. RoPE enables **resolution and duration extrapolation** — a model trained at 256×256 can generate at 512×512 because the positional encoding generalizes, unlike absolute learned embeddings.

### Scaling Advantages

DiT follows predictable scaling laws: doubling parameters yields consistent quality improvements. Sora demonstrated that scaling DiT to massive sizes produces emergent capabilities like 3D consistency, object permanence, and physically plausible motion — properties not explicitly trained for.
""",
                },
                {
                    "title": "Large-scale training",
                    "content": """\
## Large-Scale Training for Video Generation

Training a state-of-the-art video generation model requires enormous compute resources and sophisticated distributed training strategies. This is one of the most resource-intensive tasks in all of AI.

### Compute Requirements

Approximate GPU-hours for training video generation models:

| Model Scale | Parameters | GPU-Hours |
|-------------|-----------|-----------|
| Research prototype | 500M - 1B | ~10K |
| Production quality | 2B - 5B | ~100K |
| Frontier (Sora-class) | 5B - 30B+ | ~500K-1M+ |

*(GPU-hours measured on A100/H100 equivalents)*

### Distributed Training Strategies

No single GPU can hold a large video model. Key parallelism strategies include:

- **FSDP (Fully Sharded Data Parallel)**: PyTorch-native approach that shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only a fraction of the model at rest, gathering parameters on-demand for computation.
- **DeepSpeed ZeRO**: Similar sharding with three stages (ZeRO-1/2/3) offering increasing memory savings. ZeRO-3 achieves near-linear memory scaling with GPU count.
- **Tensor Parallelism**: Splits individual layers across GPUs (for very large layers). Requires high-bandwidth interconnect (NVLink).
- **Sequence Parallelism**: Distributes the long video token sequence across GPUs, essential when token counts exceed single-GPU memory.

```python
# Example: FSDP setup for DiT training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = DiT(params=5e9)
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16))
```

### Progressive Resolution Training

Training directly at high resolution is wasteful — early training iterations learn coarse structure. **Progressive training** starts at low resolution and gradually increases:

```
Stage 1: 256×256, 16 frames   (100K steps, fast iteration)
Stage 2: 512×512, 32 frames   (50K steps)
Stage 3: 720p,    64 frames   (30K steps, fine-tuning)
Stage 4: 1080p,  120 frames   (10K steps, final polish)
```

Each stage initializes from the previous checkpoint. This dramatically reduces total compute while achieving high final quality.

### Mixed Precision Training

Using **BFloat16** (bf16) for forward/backward passes while keeping master weights in FP32 halves memory usage and doubles throughput on modern GPUs (A100, H100). Gradient scaling and loss scaling ensure training stability despite reduced precision.
""",
                },
                {
                    "title": "T2V overall system",
                    "content": """\
## Text-to-Video Overall System

A production text-to-video system is far more than just a diffusion model. It is a **multi-stage pipeline** where each component plays a critical role in producing high-quality, high-resolution video from a user's text prompt.

### Full Pipeline Diagram

```mermaid
graph TD
    A[User Prompt] --> B[1. Prompt Enhancement<br>LLM Rewrite]
    B --> C[2. Text Encoding<br>CLIP + T5]
    C --> D[3. T2V Base Generation<br>DiT Diffusion, e.g. 480p 24fps]
    D --> E[4. Video VAE Decode<br>3D VAE Decoder]
    E --> F[5. Spatial Super-Res<br>Upscaling Model, 480p to 1080p/4K]
    F --> G[6. Frame Interpolation<br>RIFE / FILM, 8fps to 24fps]
    G --> H[7. Post-Processing<br>& Safety Filter]
    H --> I[Final Video Output]
```

### Component Details

**Prompt Enhancement**: Raw user prompts are often vague. An LLM (e.g., GPT-4, LLaMA) rewrites them into detailed scene descriptions with camera angles, lighting, and style keywords. This dramatically improves generation quality.

**Super-Resolution**: The base model generates at moderate resolution (e.g., 480p) for computational efficiency. A separate upscaling diffusion model adds high-frequency detail to reach 1080p or 4K.

**Frame Interpolation**: Generating every frame is expensive. Models like **RIFE** or **FILM** synthesize intermediate frames, allowing the base model to generate at low FPS (8-12) while delivering smooth output at 24+ FPS.

### Latency and Serving

End-to-end generation of a 5-second 1080p video typically takes **1-5 minutes** on a single H100 GPU. Production systems use batching, model parallelism, and caching to serve multiple requests efficiently. Distilled models and consistency models are actively being developed to reduce generation time to seconds.
""",
                },
            ],
        },
    ],
}
