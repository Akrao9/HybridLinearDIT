"""
inference.py — LinearDiT
─────────────────────────────────────────────────────────────────────────────
Generate images from a trained LinearDiT checkpoint.

Features:
  - Classifier-Free Guidance (CFG)
  - Euler sampler (flow matching t=1 → t=0)
  - Heun sampler (2nd-order, better quality at same step count)
  - Batch generation
  - Saves images as PNG grid + individual files

Usage:
    # Single prompt
    python inference.py --ckpt checkpoints/final.pt \
                        --prompt "a cinematic photo of a mountain at sunset" \
                        --steps 20 --cfg 7.5

    # Multiple prompts from a file (one per line)
    python inference.py --ckpt checkpoints/final.pt \
                        --prompt_file prompts.txt \
                        --steps 20 --cfg 7.5 --out outputs/

    # Quick test with no CFG
    python inference.py --ckpt checkpoints/final.pt \
                        --prompt "a cat" --cfg 1.0 --steps 10
"""

import os
import argparse
import json
import torch
from torch.amp import autocast
from torchvision.utils import save_image
from PIL import Image

# ── Imports from your project ─────────────────────────────────────────────────
from model import DiT
from dataloader import LATENT_SIZE, load_encoders, make_encode_text, make_decode_latents
from train import FlowMatching


# ─────────────────────────────────────────────────────────────────────────────
# Config defaults  (override via CLI args)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT = dict(
    # Model dims — must match your training CFG exactly
    latent_ch   = 32,
    latent_size = LATENT_SIZE,
    text_dim    = 768,
    text_seq    = 384,
    dit_dim     = 768,
    dit_heads   = 12,
    dit_depth   = 12,

    # Sampling
    n_steps     = 20,        # Euler steps (20 is good quality, 50 is best)
    cfg_scale   = 7.5,       # classifier-free guidance strength
                             #   1.0 = no guidance (pure unconditional)
                             #   3-5 = mild guidance
                             #   7-9 = strong prompt adherence (recommended)
                             #  >12  = over-saturated, avoid
    sampler     = "euler",   # "euler" or "heun"
    seed        = 42,

    # Output
    out_dir     = "outputs",
    device      = "cuda",
)


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip torch.compile() prefixes so raw DiT instances can load checkpoints."""
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def checkpoint_model_cfg(ckpt: dict) -> dict:
    """Prefer architecture metadata saved by train.py, with legacy fallback."""
    model_cfg = ckpt.get("model_cfg", {})
    dit_depth = model_cfg.get("dit_depth", DEFAULT["dit_depth"])
    arch_version = model_cfg.get("arch_version", "legacy")
    legacy_text_drop_block = min(8, max(0, dit_depth - 1))
    legacy_full_attn_blocks = [
        max(0, legacy_text_drop_block - 1),
        max(0, dit_depth - 1),
    ]
    if arch_version in {
        "v2_full_cross",
        "v2_full_cross_unshared_qkv",
        "v3_sana_dual_stream",
    }:
        return {
            "latent_ch":   model_cfg.get("latent_ch", DEFAULT["latent_ch"]),
            "latent_size": model_cfg.get("latent_size", DEFAULT["latent_size"]),
            "text_dim":    model_cfg.get("text_dim", DEFAULT["text_dim"]),
            "text_seq":    model_cfg.get("text_seq", DEFAULT["text_seq"]),
            "dit_dim":     model_cfg.get("dit_dim", DEFAULT["dit_dim"]),
            "dit_heads":   model_cfg.get("dit_heads", DEFAULT["dit_heads"]),
            "dit_depth":   dit_depth,
            "text_drop_block": model_cfg.get("text_drop_block", None),
            "full_attn_blocks": model_cfg.get("full_attn_blocks", []),
        }
    return {
        "latent_ch":   model_cfg.get("latent_ch", DEFAULT["latent_ch"]),
        "latent_size": model_cfg.get("latent_size", DEFAULT["latent_size"]),
        "text_dim":    model_cfg.get("text_dim", DEFAULT["text_dim"]),
        "text_seq":    model_cfg.get("text_seq", DEFAULT["text_seq"]),
        "dit_dim":     model_cfg.get("dit_dim", DEFAULT["dit_dim"]),
        "dit_heads":   model_cfg.get("dit_heads", DEFAULT["dit_heads"]),
        "dit_depth":   dit_depth,
        "text_drop_block": model_cfg.get("text_drop_block", legacy_text_drop_block),
        "full_attn_blocks": model_cfg.get("full_attn_blocks", legacy_full_attn_blocks),
    }


def checkpoint_dit_state_dict(ckpt: dict) -> tuple[dict[str, torch.Tensor], str]:
    """Prefer EMA weights when a checkpoint includes them."""
    ema_state = ckpt.get("ema_dit")
    if ema_state is not None:
        return ema_state, "ema"
    return ckpt["dit"], "train"


def load_model_bundle(
    ckpt_path: str,
    config_path: str | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor], str, dict | None]:
    """
    Load either a training checkpoint (.pt) or an inference export (.safetensors).

    Returns:
        model_cfg, state_dict, weight_source, raw_checkpoint
    """
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors requires `safetensors`. "
                "Install it with `pip install safetensors`."
            ) from exc

        resolved_config = config_path
        if resolved_config is None:
            resolved_config = os.path.join(os.path.dirname(ckpt_path), "model_config.json")
        if not os.path.exists(resolved_config):
            raise FileNotFoundError(
                "Could not find model_config.json for the safetensors checkpoint. "
                "Pass it explicitly with `--config /path/to/model_config.json`."
            )

        with open(resolved_config, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        state_dict = load_file(ckpt_path)
        return model_cfg, state_dict, "safetensors", None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = checkpoint_model_cfg(ckpt)
    state_dict, weight_source = checkpoint_dit_state_dict(ckpt)
    return model_cfg, state_dict, weight_source, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Samplers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def euler_sample(
    dit:         torch.nn.Module,
    x:           torch.Tensor,          # [B, 32, 16, 16]  initial noise
    text_hidden: torch.Tensor,          # [B, 384, 768]
    text_mask:   torch.Tensor,          # [B, 384]
    null_hidden: torch.Tensor,          # [B, 384, 768]   empty caption
    null_mask:   torch.Tensor,          # [B, 384]
    cfg_scale:   float,
    n_steps:     int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Euler sampler with Classifier-Free Guidance.

    Steps from t=1 (noise) → t=0 (clean image) using:
        x_{t+dt} = x_t  +  v_cfg * dt       dt < 0

    CFG combines conditional and unconditional velocity:
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    """
    # Linearly spaced timesteps from 1.0 down to 0.0
    timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t_now = timesteps[i]
        dt    = timesteps[i + 1] - timesteps[i]          # negative
        t_vec = t_now.expand(x.shape[0])                  # [B]

        with autocast("cuda", dtype=torch.bfloat16):
            if cfg_scale != 1.0:
                # Single forward pass with doubled batch: [cond, uncond]
                x_in      = torch.cat([x, x], dim=0)
                t_in      = t_vec.repeat(2)
                text_in   = torch.cat([text_hidden, null_hidden], dim=0)
                mask_in   = torch.cat([text_mask,   null_mask],   dim=0)
                v_both    = dit(x_in, t_in, text_in, mask_in)
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                v_cfg     = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # No guidance — single forward pass
                v_cfg = dit(x, t_vec, text_hidden, text_mask)

        x = FlowMatching.euler_step(x.float(), v_cfg.float(),
                                    t_now.item(), dt.item())

    return x


@torch.no_grad()
def heun_sample(
    dit:         torch.nn.Module,
    x:           torch.Tensor,
    text_hidden: torch.Tensor,
    text_mask:   torch.Tensor,
    null_hidden: torch.Tensor,
    null_mask:   torch.Tensor,
    cfg_scale:   float,
    n_steps:     int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Heun sampler (2nd-order Runge-Kutta) with CFG.

    Better quality than Euler at the same step count.
    Costs 2× model evaluations per step.

    Algorithm:
      1. Euler predictor:  x̃ = x_t + v(x_t, t) * dt
      2. Heun corrector:   x = x_t + 0.5 * (v(x_t,t) + v(x̃, t+dt)) * dt
    """
    timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    def get_v(xt, t_vec):
        with autocast("cuda", dtype=torch.bfloat16):
            if cfg_scale != 1.0:
                x_in    = torch.cat([xt, xt], dim=0)
                t_in    = t_vec.repeat(2)
                text_in = torch.cat([text_hidden, null_hidden], dim=0)
                mask_in = torch.cat([text_mask,   null_mask],   dim=0)
                v_both  = dit(x_in, t_in, text_in, mask_in)
                v_c, v_u = v_both.chunk(2, dim=0)
                return (v_u + cfg_scale * (v_c - v_u)).float()
            else:
                return dit(xt, t_vec, text_hidden, text_mask).float()

    for i in range(n_steps):
        t_now  = timesteps[i]
        t_next = timesteps[i + 1]
        dt     = (t_next - t_now).item()
        t_vec  = t_now.expand(x.shape[0])

        # Euler predictor
        v1   = get_v(x, t_vec)
        x_p  = x.float() + v1 * dt

        # Heun corrector (skip on last step to avoid t=0 eval)
        if i < n_steps - 1:
            t_vec_next = t_next.expand(x.shape[0])
            v2 = get_v(x_p, t_vec_next)
            x  = x.float() + 0.5 * (v1 + v2) * dt
        else:
            x = x_p

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main generate function
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    prompts:    list,                   # list of strings
    ckpt_path:  str,                    # path to .pt checkpoint
    config_path: str | None = None,     # optional config for .safetensors exports
    n_steps:    int   = DEFAULT["n_steps"],
    cfg_scale:  float = DEFAULT["cfg_scale"],
    sampler:    str   = DEFAULT["sampler"],
    seed:       int   = DEFAULT["seed"],
    out_dir:    str   = DEFAULT["out_dir"],
    device:     str   = DEFAULT["device"],
) -> list:
    """
    Generate one image per prompt. Returns list of PIL Images.

    Args:
        prompts   : list of text prompts
        ckpt_path : path to trained checkpoint (from train.py)
        n_steps   : number of denoising steps (20 = fast, 50 = best)
        cfg_scale : guidance strength (7.5 recommended)
        sampler   : "euler" or "heun"
        seed      : random seed for reproducibility
        out_dir   : directory to save output images

    Returns:
        list of PIL.Image objects
    """
    device_ = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── 1. Load encoders ──────────────────────────────────────────────────────
    print("Loading encoders...")
    tokenizer, text_encoder, dc_ae, scaling_factor = load_encoders(device_)
    encode_text    = make_encode_text(text_encoder, tokenizer, device_)
    decode_latents = make_decode_latents(dc_ae, scaling_factor, device_)

    # ── 2. Load DiT ───────────────────────────────────────────────────────────
    print(f"Loading DiT from {ckpt_path}...")
    model_cfg, state_dict, weight_source, ckpt = load_model_bundle(
        ckpt_path,
        config_path=config_path,
    )
    dit = DiT(
        latent_ch   = model_cfg["latent_ch"],
        latent_size = model_cfg["latent_size"],
        text_dim    = model_cfg["text_dim"],
        text_seq    = model_cfg["text_seq"],
        dim         = model_cfg["dit_dim"],
        n_heads     = model_cfg["dit_heads"],
        n_blocks    = model_cfg["dit_depth"],
        text_drop_block = model_cfg["text_drop_block"],
        full_attn_blocks = model_cfg["full_attn_blocks"],
    ).to(device_)
    dit.load_state_dict(normalize_state_dict_keys(state_dict))
    dit.eval()
    step = ckpt.get("step", "?") if ckpt is not None else "export"
    print(f"Loaded checkpoint from step {step} using {weight_source} weights")

    # ── 3. Encode prompts ─────────────────────────────────────────────────────
    B = len(prompts)
    print(f"Encoding {B} prompt(s)...")

    # Conditional: real prompts
    text_hidden, text_mask = encode_text(prompts)        # [B, 384, 768]

    # Unconditional: empty string for each prompt (CFG null condition)
    null_hidden, null_mask = encode_text([""] * B)       # [B, 384, 768]

    # ── 4. Sample ─────────────────────────────────────────────────────────────
    print(f"Sampling with {sampler} ({n_steps} steps, CFG={cfg_scale})...")

    # Start from pure noise at t=1
    x = torch.randn(B, model_cfg["latent_ch"],
                    model_cfg["latent_size"], model_cfg["latent_size"],
                    device=device_)

    sampler_fn = euler_sample if sampler == "euler" else heun_sample

    x = sampler_fn(
        dit         = dit,
        x           = x,
        text_hidden = text_hidden,
        text_mask   = text_mask,
        null_hidden = null_hidden,
        null_mask   = null_mask,
        cfg_scale   = cfg_scale,
        n_steps     = n_steps,
        device      = device_,
    )

    # ── 5. Decode latents → images ────────────────────────────────────────────
    images = decode_latents(x)                           # [B, 3, 512, 512]

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    # Save grid of all images
    grid_path = os.path.join(out_dir, "grid.png")
    save_image(images * 0.5 + 0.5, grid_path, nrow=min(B, 4))
    print(f"Saved grid → {grid_path}")

    # Save individual images + return PIL list
    pil_images = []
    for i, (img_t, prompt) in enumerate(zip(images, prompts)):
        # Tensor → PIL
        img_np = (img_t.float().cpu() * 0.5 + 0.5).clamp(0, 1)
        img_np = (img_np.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil    = Image.fromarray(img_np)
        pil_images.append(pil)

        # Save with sanitised prompt as filename
        fname = prompt[:60].replace(" ", "_").replace("/", "-")
        path  = os.path.join(out_dir, f"{i:03d}_{fname}.png")
        pil.save(path)
        print(f"  [{i+1}/{B}] {path}")

    return pil_images


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LinearDiT inference")
    p.add_argument("--ckpt",         required=True,
                   help="Path to checkpoint (.pt or .safetensors)")
    p.add_argument("--config",       type=str, default=None,
                   help="Optional model_config.json path for .safetensors checkpoints")
    p.add_argument("--prompt",       type=str, default=None,
                   help="Single text prompt")
    p.add_argument("--prompt_file",  type=str, default=None,
                   help="Text file with one prompt per line")
    p.add_argument("--steps",        type=int,   default=DEFAULT["n_steps"],
                   help=f"Euler steps (default: {DEFAULT['n_steps']})")
    p.add_argument("--cfg",          type=float, default=DEFAULT["cfg_scale"],
                   help=f"CFG scale (default: {DEFAULT['cfg_scale']})")
    p.add_argument("--sampler",      type=str,   default=DEFAULT["sampler"],
                   choices=["euler", "heun"],
                   help="Sampler: euler (fast) or heun (better quality)")
    p.add_argument("--seed",         type=int,   default=DEFAULT["seed"])
    p.add_argument("--out",          type=str,   default=DEFAULT["out_dir"],
                   help="Output directory")
    p.add_argument("--device",       type=str,   default=DEFAULT["device"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Collect prompts
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Default demo prompts if neither is given
        prompts = [
            "a cinematic photo of a mountain lake at golden hour",
            "a charming cartoon illustration of a cat reading a book",
            "an oil painting of a bustling city street at night, neon lights",
            "a realistic photo of a tiger walking through dense jungle fog",
        ]
        print("No prompt given — using demo prompts")

    images = generate(
        prompts   = prompts,
        ckpt_path = args.ckpt,
        config_path = args.config,
        n_steps   = args.steps,
        cfg_scale = args.cfg,
        sampler   = args.sampler,
        seed      = args.seed,
        out_dir   = args.out,
        device    = args.device,
    )

    print(f"\nDone — {len(images)} image(s) saved to {args.out}/")
