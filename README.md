# Hybrid Linear DIT

Hybrid Linear Diffusion Transformer is a text-to-image diffusion project built around a custom Flow-Matching Diffusion Transformer trained in DC-AE latent space. The final project combines a `148.5M` parameter hybrid linear/full-attention DiT, frozen T5-base text conditioning, frozen DC-AE latents, and a WebDataset-based latent cache pipeline for fast large-scale iteration.

## Key Contributions

- Hybrid linear + full attention diffusion transformer
- Anchor block design for global reasoning
- Latent cache WebDataset training pipeline
- Flow-matching training in latent space

## Highlights

- Custom `LinearDiT v2` architecture with token-funnel conditioning, per-block QKV, per-block adaLN, and full-attention anchor blocks.
- Flow-matching training objective in `32 x 16 x 16` DC-AE latent space.
- Support for raw-image training and much faster latent-cache training.
-  FP8 training with `torchao`, `torch.compile`, EMA, and classifier-free guidance.
- Inference script for Euler or Heun sampling from saved checkpoints.
- Final best checkpoint selected visually from training: `step_028000` EMA.
<img width="1620" height="2670" alt="gridlatest" src="https://github.com/user-attachments/assets/8dca8f1f-c231-4314-bc16-443d77d322cc" />

<img width="1536" height="1024" alt="modelarchituecture" src="https://github.com/user-attachments/assets/1469dd51-371b-4672-a25e-a27b9bf52316" />




## Final Model Summary

The final architecture kept the project’s empirically stronger hybrid design rather than the later dual-stream refactor:

- final model size: `148.5M` parameters
- training resolution: `512 x 512`
- latent representation: `32 x 16 x 16`
- final transformer config: `dim=768`, `heads=12`, `depth=12`
- Image and text tokens are processed together through most of the stack.
- Most blocks use linear attention for efficiency.
- A few evenly spaced full-attention blocks provide stronger global reasoning.
- Only full-attention blocks receive explicit image-to-text cross-attention.
- Mix-FFN applies 3x3 depthwise convolution to image tokens for spatial bias without positional embeddings.
- Each block owns its own self-attention QKV projection.

This model reliably learned scene layout, prompt intent, and object-level structure. The main remaining weakness was facial sharpness, which looked more like a model/data ceiling than a hidden training bug.

## Architecture Details

The final released model is best described as a custom hybrid latent DiT:

- Input representation:
  - `512 x 512` images are encoded into `32 x 16 x 16` DC-AE latents.
  - prompts are encoded with frozen T5-base into `384 x 768` hidden states.
- Conditioning:
  - continuous timesteps are embedded with a sinusoidal timestep encoder
  - pooled text plus timestep conditioning is passed through a shared conditioning trunk
  - each transformer block has its own adaLN modulation head
- Transformer stack:
  - image and text tokens are concatenated and processed together through most of the network
  - most blocks use linear attention for efficiency
  - a small number of anchor blocks use full softmax attention
  - those anchor blocks also receive explicit image-to-text cross-attention
  - each block owns its own self-attention QKV projection
- Feedforward path:
  - Mix-FFN with depthwise `3x3` convolution on image tokens
  - no explicit positional embeddings
- Training objective:
  - flow matching
  - model predicts latent velocity rather than epsilon directly in a DDPM schedule

This is not a direct reproduction of DiT, Sana, PixArt, or Flux. It is a project-specific architecture that combines ideas from several systems and then keeps the variant that worked best empirically in this codebase.

## Credits And What Changed

### What this project uses from prior models

- **T5-base**
  - frozen text encoder from `google-t5/t5-base`
  - masked token handling and long-sequence text conditioning
- **DC-AE / Sana stack**
  - frozen `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` latent autoencoder
  - latent-space training at `32 x 16 x 16`
  - overall motivation for efficient latent diffusion and linear-attention-friendly image modeling
- **DiT-style design**
  - transformer backbone for latent diffusion
  - adaLN-style conditioning
  - diffusion-in-latent-space framing rather than pixel-space modeling
- **Flow-matching / rectified-flow style training**
  - continuous timestep training in `[0, 1]`
  - velocity target prediction
  - Euler/Heun ODE-style sampling at inference
- **Classifier-free guidance**
  - null-text condition and conditional/unconditional batching during inference

### What changed in this project

- Instead of a pure dual-stream Sana-style stack, the final model uses a **hybrid token-funnel design** where image and text tokens are processed together through most of the network.
- Instead of all blocks using the same attention style, the final model uses **mostly linear attention plus a few full-attention anchor blocks** for stronger global reasoning.
- Instead of only a shared self-attention projection, the final model uses **per-block QKV**.
- Instead of broadcasting identical conditioning to every block, the final model uses a **shared conditioning trunk with per-block adaLN heads**.
- The final training pipeline relies on a **custom latent-cache WebDataset path** and a raw `webdataset` backend for faster and lower-overhead large-batch training.
- Checkpoint selection was done **visually as well as numerically**, with `step_028000` EMA chosen as the best final release checkpoint.

### Attribution Note

This repo should be understood as:

- **using pretrained components** from T5-base and DC-AE
- **borrowing high-level training and architecture ideas** from latent DiT, Sana, and flow-matching style systems
- **implementing a custom hybrid architecture and training stack** rather than reproducing any single prior model one-to-one

## Repository Layout

```text
scratchdiffusion/
├── dataloader.py
├── inference.py
├── model.py
├── precompute_wds_cache.py
├── train.py
├── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want FP8 training, make sure `torchao` is installed and supported in your environment.

## Data Pipeline

The project supports two paths:

1. Raw-image streaming
   - source: WebDataset image shards
   - frozen encoders run online during training

2. Latent cache training
   - source: cached `latents.npy`, `text.npy`, and `text_mask.npy` tar shards
   - much faster iteration and lower host overhead
   - final successful runs used the raw `webdataset` backend rather than Hugging Face streaming

## Training Data

The final model was trained on the `synthetic_enhanced_prompt_random_resolution` subset used throughout this repo. In practice, the project used:

- source image/text corpus streamed from the Hugging Face dataset `ma-xu/fine-t2i`
- a precomputed latent cache version of that same corpus for the fastest training runs
- frozen T5-base text embeddings and frozen DC-AE latents during the final training path

The strongest runs used the latent-cache repo path referenced in the code:

- source dataset repo: `ma-xu/fine-t2i`
- source subset: `synthetic_enhanced_prompt_random_resolution`
- latent cache repo id: `akrao9/512t2ilatent`
- effective training set used in the final run: about `1.6 million` samples

This means the final model quality is tied not only to architecture and optimization, but also to the quality, diversity, and prompt-image alignment of that cached dataset.

## Why The Latent Cache Helped So Much

The latent-cache dataset was one of the most important practical optimizations in the project.

Instead of doing this on every training step:

- load raw image bytes
- decode and resize images
- run the frozen DC-AE encoder
- run the frozen T5 encoder

the cached pipeline stored these outputs ahead of time:

- `latents.npy`
- `text.npy`
- `text_mask.npy`
- metadata and captions

That changed training in two important ways:

1. **Lower step cost**
   - the expensive frozen encoder work was moved out of the training loop
   - the GPU spent more time training the DiT and less time re-encoding the same data

2. **Higher throughput**
   - batches could be delivered as already-prepared latents and text features
   - this enabled larger batch sizes, faster iteration, and more stable long runs

In this project, the raw `webdataset` backend over the latent cache also reduced host RAM pressure compared with the Hugging Face streaming wrapper, which made large-batch training much more practical.

## Training Process

The end-to-end training process was:

1. Stream or precompute the dataset into DC-AE latent space.
2. Encode prompts with frozen T5-base.
3. Sample a continuous timestep `t` in `[0, 1]` using logit-normal timestep sampling.
4. Form noisy latent inputs with the flow-matching interpolation:
   - `x_t = (1 - t) * x0 + t * noise`
5. Train the DiT to predict velocity:
   - `v = noise - x0`
6. Use EMA weights for evaluation and final checkpoint selection.
7. Compare checkpoints both quantitatively and visually, then keep the best visual checkpoint.

The final best checkpoint was selected visually as `step_028000` EMA, even though later checkpoints sometimes had slightly lower validation loss.

## Training

`train.py` is built to be notebook-friendly and script-friendly.

Typical flow:

1. Set `CFG` in `train.py` or from an imported notebook/session.
2. Choose `data_mode = "latent_cache"` for the fast path.
3. Train with:

```bash
python train.py
```

Or resume:

```bash
python train.py --resume checkpoints/step_020000.pt
```

Project-proven settings from the final strong run:

- `train_batch = 256`
- `timestep_sampling = "logit_normal"`
- `use_fp8 = True`
- `compile_dit = True`
- `gradient_checkpointing = False`
- `latent_cache_loader_backend = "raw_webdataset"`

## Latent Cache Precompute

To precompute a latent cache:

```bash
python precompute_wds_cache.py \
  --split train \
  --output-dir /path/to/cache \
  --max-samples 2000000 \
  --cache-text \
  --batch-size 64
```

This writes sharded WebDataset tar files plus a manifest and cached null condition.

## Inference

Generate samples from a checkpoint:

```bash
python inference.py \
  --ckpt checkpoints/step_028000.pt \
  --prompt "a glass bridge in the mountains with a volcano" \
  --steps 20 \
  --cfg 3.5 \
  --sampler heun \
  --out outputs
```

## Best Checkpoint

The final best checkpoint for this project was chosen by visual comparison, not just lowest loss:

- Best checkpoint: `step_028000.pt`
- Best weight source: EMA

Later checkpoints continued to improve loss slightly, but sample quality became less reliable, so `step_028000` was kept as the final result.

## Evaluation

The final checkpoint was evaluated on a held-out raw-image subset from `ma-xu/fine-t2i`.

Evaluation setup:

- best checkpoint: `step_028000` EMA
- evaluation set size: `512` held-out raw images
- prompt-alignment metric: CLIP score
- distribution metrics: raw-image FID and KID

Results:

- CLIP score (generated): `33.85 ± 3.70`
- CLIP score (real images): `34.25 ± 3.63`
- Raw-image FID: `91.46`
- Raw-image KID: `0.00321 ± 0.00292`

Interpretation:

- the generated CLIP score is very close to the real-image CLIP score on the same held-out subset, which is a strong signal for prompt alignment
- FID/KID remain limited by overall model size, data quality, facial-detail weakness, and the latent-autoencoder bottleneck

Caveat:

These FID/KID numbers should be treated as project evaluation metrics rather than large-scale benchmark claims, since they were computed on a relatively small held-out subset and include the full latent autoencoding pipeline in the final image quality.

## Current Limits

The final model is clearly working, but it is not saturating the problem yet.

What it does well:

- coherent scene layout
- understandable prompts
- strong atmosphere, lighting, and environment composition
- decent object-level image generation

What remains limited:

- faces are softer and blurrier than the rest of the image
- fine local detail is less reliable than global composition
- some prompts remain better at scene understanding than precise facial fidelity

The project conclusion is that these limits look more like a **capacity/data ceiling** than a hidden architecture bug. The overfit test, training curves, and visually coherent outputs all indicate that the training pipeline and core architecture are functioning correctly.

## Scaling Outlook

The final project result suggests that this architecture should scale reasonably with:

- a larger model
- better and more diverse training data
- more face-heavy and higher-quality prompt/image pairs
- stronger latent resolution or a stronger autoencoder

Why that is a reasonable expectation:

- the current model already learns composition and prompt intent
- the training stack is stable
- the architecture passes deterministic overfit tests
- the main failure mode is not "the model cannot learn anything", but "the model learns the big picture better than the fine details"

So the likely next gains are:

- more channels / heads / depth
- better data quality and broader coverage
- possibly higher-resolution latent training

This is an inference from the project results, not a claim that larger-scale training has already been completed in this repo.

## Publishing Notes

### GitHub

Before pushing:

- add your checkpoint download instructions or exclude weights entirely
- keep large checkpoint files out of Git
- include generated sample grids in a `assets/` or `docs/` folder if desired

### Hugging Face

Use [HF_MODEL_CARD.md](HF_MODEL_CARD.md) as the starting point for the model repo README/model card. Pair it with:

- your chosen checkpoint weights
- a small sample gallery
- clear usage instructions for inference

## Documentation

- Full project writeup: [docs/final_project_report.md](docs/final_project_report.md)
- Resume bullets: [docs/resume_points.md](docs/resume_points.md)
