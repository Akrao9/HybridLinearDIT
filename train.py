"""
train.py
─────────────────────────────────────────────────────────────────────────────
Training script for the project-final LinearDiT v2 setup.

Pipeline:
  - Text encoder  : T5-base (frozen)
  - Image codec   : DC-AE f32c32 (frozen)
  - Trainable     : Hybrid LinearDiT v2
  - Objective     : Flow matching in DC-AE latent space

Run:
    python train.py

Resume:
    python train.py --resume checkpoints/step_001000.pt
"""

import os
import sys
import math
import argparse
import copy
import itertools
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm
import wandb

# ── Import your dataloader ────────────────────────────────────────────────────
from dataloader import (
    CAPTION_DROP,
    RESOLUTION,
    LATENT_SIZE,
    LATENT_CACHE_REPO_ID,
    SUBSET,
    build_cached_loaders,
    load_cached_null_condition,
    load_encoders,
    build_loaders,
    make_encode_images,
    make_encode_text,
    make_decode_latents,
)


# ─────────────────────────────────────────────────────────────────────────────
# Flow Matching utilities
# ─────────────────────────────────────────────────────────────────────────────
class FlowMatching:
    """
    Rectified Flow Matching (Liu et al. 2022, Lipman et al. 2022).

    Key equations:
      Forward  : x_t = (1 - t) * x0  +  t * noise     t ∈ [0, 1]
      Velocity : v   = noise - x0                       (what the model predicts)
      Loss     : MSE(v_pred, v_target)

    t=0 → clean image x0
    t=1 → pure noise ε ~ N(0,I)

    Inference (Euler step from t=1 → t=0):
      x_{t-dt} = x_t  -  v_pred * dt
    """

    @staticmethod
    def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample training timesteps.

        - uniform: flat coverage over [0, 1]
        - logit_normal: concentrates more mass around the mid-noise regime
        """
        mode = CFG["timestep_sampling"]
        if mode == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        if mode == "logit_normal":
            u = torch.randn(batch_size, device=device, dtype=torch.float32)
            u = u * CFG["timestep_logit_std"] + CFG["timestep_logit_mean"]
            return torch.sigmoid(u)
        raise ValueError(f"Unknown timestep_sampling={mode!r}")

    @staticmethod
    def add_noise(
        x0: torch.Tensor,          # [B, C, H, W]  clean latents
        noise: torch.Tensor,        # [B, C, H, W]  ε ~ N(0,I)
        t: torch.Tensor,            # [B]            timesteps in [0,1]
    ) -> torch.Tensor:
        """
        Interpolate between clean image and noise.
        x_t = (1 - t) * x0  +  t * noise
        """
        # Reshape t for broadcasting: [B] → [B, 1, 1, 1]
        t_ = t.view(-1, 1, 1, 1)
        return (1.0 - t_) * x0 + t_ * noise

    @staticmethod
    def get_velocity(
        x0: torch.Tensor,           # [B, C, H, W]  clean latents
        noise: torch.Tensor,         # [B, C, H, W]  ε ~ N(0,I)
    ) -> torch.Tensor:
        """
        Target velocity: v = noise - x0
        This is what the model is trained to predict.
        """
        return noise - x0

    @staticmethod
    def loss(
        v_pred: torch.Tensor,       # [B, C, H, W]  model prediction
        v_target: torch.Tensor,     # [B, C, H, W]  noise - x0
        weights: torch.Tensor | None = None,  # [B]  optional per-sample weights
    ) -> torch.Tensor:
        """
        Flow matching loss: weighted MSE between predicted and target velocity.
        """
        loss = F.mse_loss(v_pred, v_target, reduction="none")  # [B,C,H,W]
        loss = loss.mean(dim=[1, 2, 3])                         # [B]
        if weights is not None:
            loss = loss * weights
        return loss.mean()                                      # scalar

    @staticmethod
    def euler_step(
        x_t: torch.Tensor,          # [B, C, H, W]  noisy latent at time t
        v_pred: torch.Tensor,        # [B, C, H, W]  predicted velocity
        t: float,                    # current timestep (scalar)
        dt: float,                   # step size (negative: going t=1 → t=0)
    ) -> torch.Tensor:
        """
        Single Euler integration step for inference.
        x_{t+dt} = x_t  +  v_pred * dt

        For denoising, dt is negative (stepping from t=1 toward t=0).
        Example: t=1.0, dt=-0.1 → x at t=0.9
        """
        return x_t + v_pred * dt



# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Model
    latent_ch   = 32,       # DC-AE latent channels
    latent_size = LATENT_SIZE,  # DC-AE spatial size for 512px (512 / 32 = 16)
    text_dim    = 768,      # T5-base hidden size
    text_seq    = 384,      # token length
    dit_dim     = 768,      # your DiT hidden dim  → ~100M params at depth=12
    dit_heads   = 12,
    dit_depth   = 12,
    text_drop_block = None,    # None = keep text alive through all blocks; set an int to drop earlier
    full_attn_blocks = None,   # default = 4 evenly spaced softmax-attention anchor blocks
    timestep_sampling = "logit_normal",  # "uniform" or "logit_normal"
    timestep_logit_mean = 0.0,
    timestep_logit_std = 1.0,

    # Training
    num_epochs      = 2,
    train_batch     = 256,
    val_batch       = 256,
    train_samples   = 2_000_000,
    val_samples     = 2_000,
    num_workers     = 16,
    val_num_workers = 2,
    prefetch_factor = 1,
    persistent_workers = False,
    image_encode_batch = 32,   # GPU micro-batch for DC-AE encode
    text_encode_batch  = 64,   # GPU micro-batch for T5 encode
    lr              = 4e-4,       # scaled for batch 256 (base 1e-4 × 256/64)
    weight_decay    = 0.01,
    warmup_frac     = 0.05,       # 5% of steps for warmup
    grad_clip       = 1.0,
    use_ema         = True,       # keep an EMA shadow model for eval + inference checkpoints
    ema_decay       = 0.9999,     # typical diffusion EMA decay
    reset_ema_on_resume = False,  # optionally restart EMA from raw weights after loading a checkpoint
    resume_weights_only = False,  # load model/EMA weights from a checkpoint but reset optimizer + scheduler
    extend_scheduler_on_resume = True,  # continue from current LR and decay smoothly over remaining steps
    resume_to_next_epoch = False,  # treat a mid-epoch resume checkpoint as "epoch done" and start from the next epoch boundary
    use_fp8         = False,      # optional torchao float8 training for DiT Linear layers
    fp8_recipe      = "tensorwise",  # safer default for batched token projections in this model
    compile_dit     = False,      # optional torch.compile() on the trainable DiT
    gradient_checkpointing = False,  # block-level activation checkpointing to trade compute for VRAM
    wandb_watch     = False,      # avoid heavyweight per-parameter gradient hooks by default
    wandb_artifact_every = 0,     # 0 disables checkpoint artifact uploads; N uploads every Nth save
    log_every       = 100,        # print loss every N steps
    val_every       = 500,        # run validation every N steps
    save_every      = 1000,       # save checkpoint every N steps
    overfit_one_batch_steps = 0,  # if > 0, repeat a single training batch this many steps and exit
    cfg_scale       = 7.5,        # classifier-free guidance scale (inference only)
    data_mode       = "latent_cache",  # "raw" or "latent_cache"
    latent_cache_repo_id = LATENT_CACHE_REPO_ID,
    latent_cache_subset  = SUBSET,
    latent_cache_train_url = None,
    latent_cache_val_url   = None,
    latent_cache_val_shards = 4,
    latent_cache_loader_backend = "hf_streaming",  # "hf_streaming" or "raw_webdataset"
    latent_cache_shuffle_buffer = 10_000,
    pin_memory      = True,

    # Paths
    ckpt_dir    = "checkpoints",
    device      = "cuda",

    # Wandb
    wandb_project = "LinearDit",
    wandb_run_name = None,       # set to a string to name the run, else auto-named
    wandb_log_images = True,     # log validation images to wandb every val_every steps
    wandb_n_images   = 4,        # how many images to log per val step
    wandb_log_generated = True,  # also log actual generated-from-noise samples
    wandb_log_reconstructions = True,  # keep logging decoded val latents for comparison
    wandb_sample_steps = 20,     # sampler steps for generated validation images
    wandb_sample_cfg_scale = 3.5,  # CFG scale for generated validation images
    wandb_sample_sampler = "heun",  # "euler" or "heun"
    wandb_sample_seed = 1234,    # fixed seed so validation images are comparable over time
)


# ─────────────────────────────────────────────────────────────────────────────
# ── Import your DiT model ─────────────────────────────────────────────────────
# Define your model in model.py and import it here.
# Your DiT must accept:
#   forward(x, t, text_hidden, text_mask)
#     x           : [B, C, H, W]   noisy latents
#     t           : [B]             float timesteps in [0,1]  (flow matching)
#     text_hidden : [B, 384, 768]   T5 hidden states
#     text_mask   : [B, 384]        attention mask (1=real, 0=pad)
#   and return predicted velocity [B, C, H, W]
from model import DiT


total_steps_global = None
warmup_steps_global = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def unwrap_compiled_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying raw module when torch.compile() wraps it."""
    return getattr(module, "_orig_mod", module)


def build_optimizer(dit: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with standard decay / no-decay parameter grouping."""
    raw_dit = unwrap_compiled_module(dit)
    decay_params = []
    no_decay_params = []
    for _, param in raw_dit.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": CFG["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CFG["lr"],
        betas=(0.9, 0.999),
        fused=True,
    )


def assert_ema_matches_model(ema_dit: torch.nn.Module, dit: torch.nn.Module) -> None:
    raw_dit = unwrap_compiled_module(dit)
    n_ema_params = sum(1 for _ in ema_dit.parameters())
    n_dit_params = sum(1 for _ in raw_dit.parameters())
    if n_ema_params != n_dit_params:
        raise RuntimeError(
            f"EMA/DiT param count mismatch: {n_ema_params} vs {n_dit_params}"
        )
    n_ema_buffers = sum(1 for _ in ema_dit.buffers())
    n_dit_buffers = sum(1 for _ in raw_dit.buffers())
    if n_ema_buffers != n_dit_buffers:
        raise RuntimeError(
            f"EMA/DiT buffer count mismatch: {n_ema_buffers} vs {n_dit_buffers}"
        )


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip torch.compile() prefixes so raw DiT instances can load checkpoints."""
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def current_model_cfg(dit) -> dict[str, object]:
    """Architecture fields needed to recreate the DiT for resume/inference."""
    raw_dit = unwrap_compiled_module(dit)
    return {
        "arch_version": "v2_full_cross_unshared_qkv",
        "latent_ch":   CFG["latent_ch"],
        "latent_size": CFG["latent_size"],
        "text_dim":    CFG["text_dim"],
        "text_seq":    CFG["text_seq"],
        "dit_dim":     CFG["dit_dim"],
        "dit_heads":   CFG["dit_heads"],
        "dit_depth":   CFG["dit_depth"],
        "gradient_checkpointing": raw_dit.gradient_checkpointing,
        "text_drop_block": raw_dit.text_drop_block,
        "full_attn_blocks": sorted(raw_dit.full_attn_blocks),
    }


def maybe_enable_fp8_training(dit, enabled: bool, recipe: str):
    """Optionally convert compatible DiT Linear layers to torchao float8 training."""
    if not enabled:
        return dit

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise ImportError(
            "FP8 training requested, but torchao is not installed. "
            "Install it with `pip install -U torchao`."
        ) from exc

    config = Float8LinearConfig.from_recipe_name(recipe)

    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        del fqn
        return (
            isinstance(mod, torch.nn.Linear)
            and mod.in_features % 16 == 0
            and mod.out_features % 16 == 0
        )

    convert_to_float8_training(dit, config=config, module_filter_fn=module_filter_fn)
    print(f"[fp8] Enabled torchao float8 training with recipe='{recipe}'")
    return dit


def make_ema_model(dit: torch.nn.Module) -> torch.nn.Module:
    """Create a frozen EMA copy of the raw training model."""
    ema_dit = copy.deepcopy(unwrap_compiled_module(dit)).eval()
    for param in ema_dit.parameters():
        param.requires_grad_(False)
    return ema_dit


@torch.no_grad()
def copy_model_state(target: torch.nn.Module, source: torch.nn.Module):
    """Hard-copy weights from source -> target."""
    source = unwrap_compiled_module(source)
    target.load_state_dict(normalize_state_dict_keys(source.state_dict()))
    target.eval()
    for param in target.parameters():
        param.requires_grad_(False)


@torch.no_grad()
def update_ema_model(ema_dit: torch.nn.Module, dit: torch.nn.Module, decay: float):
    """EMA update on parameters and floating buffers."""
    raw_dit = unwrap_compiled_module(dit)

    for ema_param, model_param in zip(ema_dit.parameters(), raw_dit.parameters()):
        model_data = model_param.detach().to(
            device=ema_param.device,
            dtype=ema_param.dtype,
        )
        ema_param.lerp_(model_data, 1.0 - decay)

    for ema_buffer, model_buffer in zip(ema_dit.buffers(), raw_dit.buffers()):
        model_data = model_buffer.detach().to(device=ema_buffer.device)
        if torch.is_floating_point(ema_buffer):
            model_data = model_data.to(dtype=ema_buffer.dtype)
            ema_buffer.lerp_(model_data, 1.0 - decay)
        else:
            ema_buffer.copy_(model_data)


def save_checkpoint(dit, ema_dit, optimizer, scheduler_lr, step, loss, path):
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    raw_dit = unwrap_compiled_module(dit)
    raw_ema = unwrap_compiled_module(ema_dit) if ema_dit is not None else None
    torch.save({
        "step":          step,
        "loss":          loss,
        "dit":           raw_dit.state_dict(),
        "ema_dit":       raw_ema.state_dict() if raw_ema is not None else None,
        "model_cfg":     current_model_cfg(raw_dit),
        "use_ema":       CFG["use_ema"],
        "ema_decay":     CFG["ema_decay"],
        "scheduler_total_steps": total_steps_global,
        "scheduler_warmup_steps": warmup_steps_global,
        "use_fp8":       CFG["use_fp8"],
        "fp8_recipe":    CFG["fp8_recipe"],
        "compile_dit":   CFG["compile_dit"],
        "optimizer":     optimizer.state_dict(),
        "scheduler_lr":  scheduler_lr.state_dict(),
    }, path)
    ema_note = " + EMA" if raw_ema is not None else ""
    print(f"[ckpt] saved{ema_note} → {path}")


def load_checkpoint(
    path,
    dit,
    optimizer,
    scheduler_lr,
    ema_dit=None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
):
    ckpt = torch.load(path, map_location="cpu")
    dit.load_state_dict(normalize_state_dict_keys(ckpt["dit"]))
    if ema_dit is not None:
        ema_state = ckpt.get("ema_dit")
        if ema_state is not None:
            ema_dit.load_state_dict(normalize_state_dict_keys(ema_state))
        else:
            copy_model_state(ema_dit, dit)
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if load_optimizer and load_scheduler and scheduler_lr is not None:
        scheduler_lr.load_state_dict(ckpt["scheduler_lr"])
    mode = "resumed" if load_optimizer else "loaded weights"
    print(f"[ckpt] {mode} from step {ckpt['step']}  loss={ckpt['loss']:.4f}")
    return ckpt


def get_lr_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)             # linear warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))   # cosine decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_resume_scheduler(
    optimizer,
    start_step: int,
    total_steps: int,
):
    """
    Resume-aware cosine tail:
      - keeps the currently loaded optimizer LR at `start_step`
      - decays smoothly to zero by `total_steps`
      - never re-enters warmup or spikes upward after resume
    """
    base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
    start_scales = [
        group["lr"] / max(base_lr, 1e-12)
        for group, base_lr in zip(optimizer.param_groups, base_lrs)
    ]

    def build_lambda(start_scale: float):
        def lr_lambda(step: int):
            if step <= start_step:
                return start_scale
            if total_steps <= start_step:
                return start_scale
            progress = (step - start_step) / max(1, total_steps - start_step)
            progress = min(max(progress, 0.0), 1.0)
            return start_scale * 0.5 * (1 + math.cos(math.pi * progress))
        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[build_lambda(scale) for scale in start_scales],
    )
    scheduler.last_epoch = start_step
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return scheduler


def materialize_training_batch(
    batch: dict,
    device: torch.device,
    encode_images=None,
    encode_text=None,
    null_condition: tuple[torch.Tensor, torch.Tensor] | None = None,
    caption_drop: float = 0.0,
) -> dict:
    """Move a raw-image batch or cached-latent batch onto the GPU for training."""
    aesthetic = batch["aesthetic_score"].to(device, non_blocking=True)
    captions = list(batch.get("captions", []))

    if "latents" in batch:
        latents = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
        text_hidden = batch["text_hidden"].to(device, dtype=torch.bfloat16, non_blocking=True)
        text_mask = batch["text_mask"].to(device, dtype=torch.bool, non_blocking=True)
        if caption_drop > 0.0 and null_condition is not None:
            null_hidden, null_mask = null_condition
            drop_mask = torch.rand(latents.shape[0], device=device) < caption_drop
            if drop_mask.any():
                text_hidden = text_hidden.clone()
                text_mask = text_mask.clone()
                n_drop = int(drop_mask.sum().item())
                text_hidden[drop_mask] = null_hidden.unsqueeze(0).expand(n_drop, -1, -1)
                text_mask[drop_mask] = null_mask.unsqueeze(0).expand(n_drop, -1)
                for idx in drop_mask.nonzero(as_tuple=False).flatten().tolist():
                    captions[idx] = ""
        return {
            "latents": latents,
            "text_hidden": text_hidden,
            "text_mask": text_mask,
            "aesthetic": aesthetic,
            "captions": captions,
            "pixel_values": None,
        }

    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    latents = encode_images(pixel_values)
    text_hidden, text_mask = encode_text(captions)
    return {
        "latents": latents,
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "aesthetic": aesthetic,
        "captions": captions,
        "pixel_values": pixel_values,
    }


@torch.no_grad()
def run_validation(
    dit,
    val_loader,
    device,
    encode_images=None,
    encode_text=None,
    max_batches=10,
):
    """Run flow matching velocity loss on a small fixed val set."""
    was_training = dit.training
    dit.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        prepared = materialize_training_batch(
            batch,
            device,
            encode_images=encode_images,
            encode_text=encode_text,
        )
        latents = prepared["latents"]
        text_hidden = prepared["text_hidden"]
        mask = prepared["text_mask"]
        with autocast("cuda", dtype=torch.bfloat16):
            noise    = torch.randn_like(latents)
            t        = FlowMatching.sample_timesteps(latents.shape[0], device)
            t_bf     = t.to(latents.dtype)
            x_t      = FlowMatching.add_noise(latents, noise, t_bf)
            v_target = FlowMatching.get_velocity(latents, noise)
            v_pred   = dit(x_t, t, text_hidden, mask)
            loss     = FlowMatching.loss(v_pred, v_target)
        losses.append(loss.item())
    if not losses:
        raise RuntimeError(
            "Validation loader produced no batches. "
            "Check val_samples / val_url_pattern for the current data mode."
        )
    if was_training:
        dit.train()
    else:
        dit.eval()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_for_wandb(
    dit,
    text_hidden: torch.Tensor,
    text_mask: torch.Tensor,
    null_hidden: torch.Tensor,
    null_mask: torch.Tensor,
    *,
    cfg_scale: float,
    n_steps: int,
    sampler: str,
    latent_ch: int,
    latent_size: int,
    device: torch.device,
    seed: int,
):
    """Generate latent samples from noise for validation logging."""
    was_training = dit.training
    dit.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(
        text_hidden.shape[0],
        latent_ch,
        latent_size,
        latent_size,
        device=device,
        generator=gen,
    )

    timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    def get_v(xt: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", dtype=torch.bfloat16):
            if cfg_scale != 1.0:
                x_in = torch.cat([xt, xt], dim=0)
                t_in = t_vec.repeat(2)
                text_in = torch.cat([text_hidden, null_hidden], dim=0)
                mask_in = torch.cat([text_mask, null_mask], dim=0)
                v_both = dit(x_in, t_in, text_in, mask_in)
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                return (v_uncond + cfg_scale * (v_cond - v_uncond)).float()
            return dit(xt, t_vec, text_hidden, text_mask).float()

    if sampler == "heun":
        for i in range(n_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            dt = (t_next - t_now).item()
            t_vec = t_now.expand(x.shape[0])

            v1 = get_v(x, t_vec)
            x_pred = x.float() + v1 * dt

            if i < n_steps - 1:
                t_vec_next = t_next.expand(x.shape[0])
                v2 = get_v(x_pred, t_vec_next)
                x = x.float() + 0.5 * (v1 + v2) * dt
            else:
                x = x_pred
    else:
        for i in range(n_steps):
            t_now = timesteps[i]
            dt = (timesteps[i + 1] - timesteps[i]).item()
            t_vec = t_now.expand(x.shape[0])
            v_cfg = get_v(x, t_vec)
            x = FlowMatching.euler_step(x.float(), v_cfg.float(), t_now.item(), dt)

    if was_training:
        dit.train()
    else:
        dit.eval()

    return x


def run_overfit_one_batch(
    *,
    dit,
    ema_dit,
    optimizer,
    scheduler_lr,
    train_loader,
    device,
    data_mode,
    encode_images=None,
    encode_text=None,
    null_condition=None,
):
    """Repeat one training batch to quickly sanity-check optimization."""
    steps = int(CFG["overfit_one_batch_steps"])
    if steps <= 0:
        return

    print(f"[overfit] running one-batch sanity test for {steps} steps")
    overfit_batch = next(iter(train_loader))
    running_loss = 0.0
    running_count = 0
    last_grad_norm = 0.0

    prepared = materialize_training_batch(
        overfit_batch,
        device,
        encode_images=encode_images,
        encode_text=encode_text,
        null_condition=null_condition,
        caption_drop=0.0,
    )
    latents = prepared["latents"]
    text_hidden = prepared["text_hidden"]
    mask = prepared["text_mask"]
    aesthetic = prepared["aesthetic"]
    fixed_noise = torch.randn_like(latents)
    fixed_t = FlowMatching.sample_timesteps(latents.shape[0], device)
    fixed_t_bf = fixed_t.to(latents.dtype)
    fixed_x_t = FlowMatching.add_noise(latents, fixed_noise, fixed_t_bf)
    fixed_v_target = FlowMatching.get_velocity(latents, fixed_noise)

    # The main scheduler may leave the optimizer at the step-0 warmup LR (often 0).
    # For a one-batch sanity check we want a fixed, nonzero base LR throughout.
    for group in optimizer.param_groups:
        group["lr"] = group.get("initial_lr", CFG["lr"])

    dit.train()
    pbar = tqdm(
        range(steps),
        desc="Overfit 1 batch",
        total=steps,
        dynamic_ncols=True,
        leave=True,
        mininterval=1.0,
        file=sys.stdout,
    )

    for step_idx in pbar:

        with autocast("cuda", dtype=torch.bfloat16):
            v_pred = dit(fixed_x_t, fixed_t, text_hidden, mask)
            weights = (aesthetic / 10.0).clamp(0.5, 1.0)
            loss = FlowMatching.loss(v_pred, fixed_v_target, weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), CFG["grad_clip"])
        optimizer.step()
        if ema_dit is not None:
            update_ema_model(ema_dit, dit, CFG["ema_decay"])

        running_loss += loss.item()
        running_count += 1
        last_grad_norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

        if (step_idx + 1) % max(1, CFG["log_every"]) == 0 or step_idx == steps - 1:
            avg_loss = running_loss / max(1, running_count)
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}", grad=f"{last_grad_norm:.2f}")
            wandb.log(
                {
                    "overfit/loss": avg_loss,
                    "overfit/lr": lr_now,
                    "overfit/grad_norm": last_grad_norm,
                    "overfit/step": step_idx + 1,
                },
                step=step_idx + 1,
            )
            running_loss = 0.0
            running_count = 0

    print("[overfit] complete")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(resume_path=None):
    device = torch.device(CFG["device"])
    data_mode = CFG["data_mode"]
    encode_images = None
    encode_text = None
    decode_latents = None
    null_condition = None

    if data_mode == "raw":
        # ── 1. Load frozen encoders ───────────────────────────────────────────
        print("Loading encoders...")
        tokenizer, text_encoder, dc_ae, scaling_factor = load_encoders(device)
        encode_images  = make_encode_images(
            dc_ae,
            scaling_factor,
            device,
            microbatch_size=CFG["image_encode_batch"],
        )
        encode_text    = make_encode_text(
            text_encoder,
            tokenizer,
            device,
            microbatch_size=CFG["text_encode_batch"],
        )
        decode_latents = make_decode_latents(dc_ae, scaling_factor, device)

        # ── 2. Build raw data loaders ─────────────────────────────────────────
        print("Building raw-image loaders...")
        train_loader, val_loader = build_loaders(
            train_batch   = CFG["train_batch"],
            val_batch     = CFG["val_batch"],
            num_workers   = CFG["num_workers"],
            val_num_workers = CFG["val_num_workers"],
            prefetch_factor = CFG["prefetch_factor"],
            persistent_workers = CFG["persistent_workers"],
            train_samples = CFG["train_samples"],
            val_samples   = CFG["val_samples"],
            pin_memory    = CFG["pin_memory"],
        )
    else:
        # ── 1. Build cached latent loaders ────────────────────────────────────
        print("Building latent-cache loaders...")
        train_loader, val_loader = build_cached_loaders(
            train_batch   = CFG["train_batch"],
            val_batch     = CFG["val_batch"],
            num_workers   = CFG["num_workers"],
            val_num_workers = CFG["val_num_workers"],
            prefetch_factor = CFG["prefetch_factor"],
            persistent_workers = CFG["persistent_workers"],
            train_samples = CFG["train_samples"],
            val_samples   = CFG["val_samples"],
            repo_id       = CFG["latent_cache_repo_id"],
            subset        = CFG["latent_cache_subset"],
            train_url_pattern = CFG["latent_cache_train_url"],
            val_url_pattern   = CFG["latent_cache_val_url"],
            val_shards        = CFG["latent_cache_val_shards"],
            loader_backend    = CFG["latent_cache_loader_backend"],
            shuffle_buffer    = CFG["latent_cache_shuffle_buffer"],
            pin_memory        = CFG["pin_memory"],
        )
        print("Loading cached null condition...")
        null_condition = load_cached_null_condition(
            repo_id=CFG["latent_cache_repo_id"],
            subset=CFG["latent_cache_subset"],
            device=device,
        )
        if CFG["wandb_log_images"]:
            print("Loading encoders for reconstruction logging...")
            tokenizer, text_encoder, dc_ae, scaling_factor = load_encoders(device)
            decode_latents = make_decode_latents(dc_ae, scaling_factor, device)

    # ── 3. Build DiT ─────────────────────────────────────────────────────────
    dit = DiT(
        latent_ch   = CFG["latent_ch"],
        latent_size = CFG["latent_size"],
        text_dim    = CFG["text_dim"],
        text_seq    = CFG["text_seq"],
        dim         = CFG["dit_dim"],
        n_heads     = CFG["dit_heads"],
        n_blocks    = CFG["dit_depth"],
        text_drop_block = CFG["text_drop_block"],
        full_attn_blocks = CFG["full_attn_blocks"],
        gradient_checkpointing = CFG["gradient_checkpointing"],
    ).to(device)
    dit = maybe_enable_fp8_training(
        dit,
        enabled=CFG["use_fp8"],
        recipe=CFG["fp8_recipe"],
    )
    ema_dit = make_ema_model(dit) if CFG["use_ema"] else None
    if ema_dit is not None:
        assert_ema_matches_model(ema_dit, dit)

    print(f"DiT parameters: {count_params(dit):.1f}M")
    print(f"VRAM after DiT: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    if ema_dit is not None:
        print(f"EMA enabled: decay={CFG['ema_decay']}")
    raw_dit = unwrap_compiled_module(dit)
    print(
        f"[arch] n_blocks={raw_dit.n_blocks}, "
        f"full_attn={sorted(raw_dit.full_attn_blocks)}, "
        f"text_drop={raw_dit.text_drop_block}"
    )
    assert raw_dit.n_blocks == CFG["dit_depth"]

    # ── 4. Optimizer + LR scheduler ──────────────────────────────────────────
    optimizer = build_optimizer(dit)

    steps_per_epoch = CFG["train_samples"] // CFG["train_batch"]
    total_steps     = steps_per_epoch * CFG["num_epochs"]
    warmup_steps    = int(total_steps * CFG["warmup_frac"])

    scheduler_lr = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    global total_steps_global, warmup_steps_global
    total_steps_global = total_steps
    warmup_steps_global = warmup_steps

    # ── 6. Resume from checkpoint ─────────────────────────────────────────────
    start_step = 0
    scheduler_resume_step = 0
    ckpt = None
    loaded_step = None
    if resume_path:
        resume_weights_only = CFG["resume_weights_only"]
        ckpt = load_checkpoint(
            resume_path,
            dit,
            optimizer,
            scheduler_lr,
            ema_dit=ema_dit,
            load_optimizer=not resume_weights_only,
            load_scheduler=(
                (not resume_weights_only)
                and (not CFG["extend_scheduler_on_resume"])
            ),
        )
        loaded_step = ckpt["step"]
        start_step = 0 if resume_weights_only else ckpt["step"]
        scheduler_resume_step = start_step
        if (
            CFG["resume_to_next_epoch"]
            and steps_per_epoch > 0
            and start_step % steps_per_epoch != 0
        ):
            scheduler_resume_step = min(
                total_steps,
                ((start_step // steps_per_epoch) + 1) * steps_per_epoch,
            )
        if ema_dit is not None and CFG["reset_ema_on_resume"]:
            copy_model_state(ema_dit, dit)
            print("[ema] reset EMA weights from resumed raw model")
        if CFG["resume_weights_only"]:
            print("[resume] weights-only mode: optimizer and scheduler reset")
        elif CFG["extend_scheduler_on_resume"]:
            scheduler_lr = get_resume_scheduler(
                optimizer,
                start_step=scheduler_resume_step,
                total_steps=total_steps,
            )
            print(
                "[lr] resume scheduler tail enabled "
                f"(step={scheduler_resume_step}, lr={scheduler_lr.get_last_lr()[0]:.2e}, total={total_steps})"
            )

    if CFG["compile_dit"]:
        dit = torch.compile(dit)
        print("[compile] DiT compiled")

    # ── 7. Init wandb ─────────────────────────────────────────────────────────
    wandb.init(
        project = CFG["wandb_project"],
        name    = CFG["wandb_run_name"],
        resume  = "allow" if (resume_path and not CFG["resume_weights_only"]) else None,
        config  = {
            # model
            "arch_version":  current_model_cfg(dit)["arch_version"],
            "dit_dim":       CFG["dit_dim"],
            "dit_heads":     CFG["dit_heads"],
            "dit_depth":     CFG["dit_depth"],
            "gradient_checkpointing": unwrap_compiled_module(dit).gradient_checkpointing,
            "text_drop_block": unwrap_compiled_module(dit).text_drop_block,
            "full_attn_blocks": sorted(unwrap_compiled_module(dit).full_attn_blocks),
            "timestep_sampling": CFG["timestep_sampling"],
            "timestep_logit_mean": CFG["timestep_logit_mean"],
            "timestep_logit_std": CFG["timestep_logit_std"],
            "dit_params_M":  round(count_params(dit), 1),
            "use_ema":       CFG["use_ema"],
            "ema_decay":     CFG["ema_decay"],
            "reset_ema_on_resume": CFG["reset_ema_on_resume"],
            "resume_weights_only": CFG["resume_weights_only"],
            "extend_scheduler_on_resume": CFG["extend_scheduler_on_resume"],
            "use_fp8":       CFG["use_fp8"],
            "fp8_recipe":    CFG["fp8_recipe"],
            "compile_dit":   CFG["compile_dit"],
            "wandb_watch":   CFG["wandb_watch"],
            "wandb_artifact_every": CFG["wandb_artifact_every"],
            "latent_ch":     CFG["latent_ch"],
            "latent_size":   CFG["latent_size"],
            "text_dim":      CFG["text_dim"],
            "text_seq":      CFG["text_seq"],
            "data_mode":     CFG["data_mode"],
            "latent_cache_repo_id": CFG["latent_cache_repo_id"],
            "latent_cache_subset": CFG["latent_cache_subset"],
            "latent_cache_val_shards": CFG["latent_cache_val_shards"],
            "latent_cache_loader_backend": CFG["latent_cache_loader_backend"],
            "latent_cache_shuffle_buffer": CFG["latent_cache_shuffle_buffer"],
            "pin_memory": CFG["pin_memory"],
            # training
            "train_batch":   CFG["train_batch"],
            "train_samples": CFG["train_samples"],
            "num_epochs":    CFG["num_epochs"],
            "lr":            CFG["lr"],
            "weight_decay":  CFG["weight_decay"],
            "warmup_frac":   CFG["warmup_frac"],
            "grad_clip":     CFG["grad_clip"],
            "overfit_one_batch_steps": CFG["overfit_one_batch_steps"],
            "total_steps":   total_steps,
            "warmup_steps":  warmup_steps,
            # encoders
            "text_encoder":  "google-t5/t5-base",
            "image_codec":   "dc-ae-f32c32-sana-1.1",
            "resolution":    RESOLUTION,
        },
    )
    if CFG["wandb_watch"]:
        wandb.watch(dit, log=None)

    val_image_batch_cached = None
    if CFG["wandb_log_images"] and decode_latents is not None:
        val_image_batch_cached = next(iter(val_loader))

    if CFG["overfit_one_batch_steps"] > 0:
        run_overfit_one_batch(
            dit=dit,
            ema_dit=ema_dit,
            optimizer=optimizer,
            scheduler_lr=scheduler_lr,
            train_loader=train_loader,
            device=device,
            data_mode=data_mode,
            encode_images=encode_images,
            encode_text=encode_text,
            null_condition=null_condition,
        )
        return

    # ── 8. Training loop ──────────────────────────────────────────────────────
    dit.train()
    effective_start_step = start_step
    if (
        resume_path
        and CFG["resume_to_next_epoch"]
        and steps_per_epoch > 0
        and start_step % steps_per_epoch != 0
    ):
        effective_start_step = min(
            total_steps,
            ((start_step // steps_per_epoch) + 1) * steps_per_epoch,
        )
        print(
            "[resume] advancing to next epoch boundary "
            f"({start_step} -> {effective_start_step})"
        )

    global_step  = effective_start_step
    running_loss = 0.0
    last_grad_norm = 0.0

    start_epoch = effective_start_step // max(1, steps_per_epoch)
    resume_batch_skip = 0 if CFG["resume_to_next_epoch"] else (start_step % max(1, steps_per_epoch))

    print(f"\nStarting training")
    print(f"  Total steps     : {total_steps:,}")
    print(f"  Warmup steps    : {warmup_steps:,}")
    print(f"  Steps per epoch : {steps_per_epoch:,}")
    print(f"  Resume step     : {start_step:,}")
    print(f"  Effective step  : {effective_start_step:,}")
    if loaded_step is not None and CFG["resume_weights_only"]:
        print(f"  Loaded weights  : step {loaded_step:,} (fresh optimizer/scheduler)")
    if resume_batch_skip:
        print(f"  Resume skip     : {resume_batch_skip:,} batches in epoch {start_epoch + 1}")

    for epoch in range(start_epoch, CFG["num_epochs"]):
        epoch_total = steps_per_epoch
        if epoch == start_epoch and resume_batch_skip:
            epoch_total = max(0, steps_per_epoch - resume_batch_skip)
            epoch_loader = itertools.islice(
                train_loader,
                resume_batch_skip,
                resume_batch_skip + epoch_total,
            )
        else:
            epoch_loader = itertools.islice(train_loader, epoch_total)

        pbar = tqdm(
            epoch_loader,
            desc=f"Epoch {epoch+1}/{CFG['num_epochs']}",
            total=epoch_total,
            dynamic_ncols=True,
            leave=True,
            mininterval=1.0,
            file=sys.stdout,
        )

        for batch in pbar:
            if global_step >= total_steps:
                break

            prepared = materialize_training_batch(
                batch,
                device,
                encode_images=encode_images,
                encode_text=encode_text,
                null_condition=null_condition,
                caption_drop=CAPTION_DROP if data_mode == "latent_cache" else 0.0,
            )
            pixel_values = prepared["pixel_values"]
            aesthetic = prepared["aesthetic"]
            latents = prepared["latents"]
            text_hidden = prepared["text_hidden"]
            mask = prepared["text_mask"]

            with autocast("cuda", dtype=torch.bfloat16):
                # ── Flow Matching forward process ─────────────────────────────
                noise    = torch.randn_like(latents)                # ε ~ N(0,I)  bfloat16
                t        = FlowMatching.sample_timesteps(           # t ~ U(0,1)  float32
                    latents.shape[0], device
                )
                # Cast t to match latents dtype for broadcasting inside autocast
                t_bf     = t.to(latents.dtype)
                # x_t = (1-t)*x0 + t*ε   — interpolate clean ↔ noise
                x_t      = FlowMatching.add_noise(latents, noise, t_bf)
                # v = ε - x0             — target velocity
                v_target = FlowMatching.get_velocity(latents, noise)

                # ── DiT forward pass (only thing with gradients) ──────────────
                # Model receives t as float32 [0,1] — TimestepEmbedding handles scaling
                v_pred = dit(x_t, t, text_hidden, mask)

                # ── Flow Matching loss ────────────────────────────────────────
                # Weight by aesthetic score: higher quality → matters more
                # Clamp to [0.5, 1.0] so low-quality images still contribute
                weights = (aesthetic / 10.0).clamp(0.5, 1.0)
                loss = FlowMatching.loss(v_pred, v_target, weights)

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), CFG["grad_clip"])
            optimizer.step()
            if ema_dit is not None:
                update_ema_model(ema_dit, dit, CFG["ema_decay"])
            scheduler_lr.step()

            running_loss += loss.item()
            global_step  += 1
            last_grad_norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                avg_loss = running_loss / CFG["log_every"]
                lr_now   = scheduler_lr.get_last_lr()[0]
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9
                vram_peak_reserved = torch.cuda.max_memory_reserved() / 1e9
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr_now:.2e}",
                    alloc=f"{vram_alloc:.1f}G",
                    reserv=f"{vram_reserved:.1f}G",
                )
                wandb.log({
                    "train/loss":       avg_loss,
                    "train/lr":         lr_now,
                    "train/vram_alloc_gb":    vram_alloc,
                    "train/vram_reserved_gb": vram_reserved,
                    "train/vram_peak_reserved_gb": vram_peak_reserved,
                    "train/grad_norm": last_grad_norm,
                    "train/epoch":      epoch,
                    "train/step":       global_step,
                }, step=global_step)
                running_loss = 0.0

            # ── Validation ────────────────────────────────────────────────────
            if global_step % CFG["val_every"] == 0:
                eval_model = ema_dit if ema_dit is not None else dit
                val_loss = run_validation(
                    eval_model,
                    val_loader,
                    device,
                    encode_images=encode_images,
                    encode_text=encode_text,
                )
                model_label = "ema" if ema_dit is not None else "train"
                print(f"\n[step {global_step:>6}] val_loss ({model_label}) = {val_loss:.4f}")

                log_dict = {
                    "val/loss": val_loss,
                    "val/using_ema": 1.0 if ema_dit is not None else 0.0,
                    "train/step": global_step,
                }

                # Log original vs decoded images to wandb
                if CFG["wandb_log_images"] and decode_latents is not None:
                    val_batch = val_image_batch_cached
                    n = min(CFG["wandb_n_images"], len(val_batch["aesthetic_score"]))
                    prepared_val = materialize_training_batch(
                        {
                            key: value[:n] if torch.is_tensor(value) else value[:n]
                            for key, value in val_batch.items()
                        },
                        device,
                        encode_images=encode_images,
                        encode_text=encode_text,
                    )
                    pv = prepared_val["pixel_values"]
                    lats = prepared_val["latents"]
                    text_hidden_val = prepared_val["text_hidden"]
                    text_mask_val = prepared_val["text_mask"]

                    def to_wandb(t):
                        """[-1,1] CHW tensor → wandb.Image"""
                        img = (t.float().cpu() * 0.5 + 0.5).clamp(0, 1)
                        img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
                        return wandb.Image(img)

                    if pv is not None:
                        log_dict["val/originals"] = [
                            to_wandb(pv[i].cpu()) for i in range(n)
                        ]

                    if CFG["wandb_log_reconstructions"]:
                        recon = decode_latents(lats)  # [n,3,512,512] in [-1,1]
                        log_dict["val/dc_ae_reconstructions"] = [
                            to_wandb(recon[i]) for i in range(n)
                        ]

                    if CFG["wandb_log_generated"]:
                        if null_condition is not None:
                            null_hidden, null_mask = null_condition
                            null_hidden = null_hidden.unsqueeze(0).expand(n, -1, -1)
                            null_mask = null_mask.unsqueeze(0).expand(n, -1)
                        elif encode_text is not None:
                            null_hidden, null_mask = encode_text([""] * n)
                        else:
                            null_hidden = None
                            null_mask = None

                        if null_hidden is not None and null_mask is not None:
                            sampled = sample_for_wandb(
                                eval_model,
                                text_hidden_val,
                                text_mask_val,
                                null_hidden,
                                null_mask,
                                cfg_scale=CFG["wandb_sample_cfg_scale"],
                                n_steps=CFG["wandb_sample_steps"],
                                sampler=CFG["wandb_sample_sampler"],
                                latent_ch=CFG["latent_ch"],
                                latent_size=CFG["latent_size"],
                                device=device,
                                seed=CFG["wandb_sample_seed"],
                            )
                            generated = decode_latents(sampled)
                            log_dict["val/generated_samples"] = [
                                to_wandb(generated[i]) for i in range(n)
                            ]

                    log_dict["val/captions"] = wandb.Table(
                        columns=["step", "caption", "aesthetic_score"],
                        data=[
                            [global_step,
                             prepared_val["captions"][i],
                             val_batch["aesthetic_score"][i].item()]
                            for i in range(n)
                        ]
                    )

                wandb.log(log_dict, step=global_step)
                dit.train()

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % CFG["save_every"] == 0:
                ckpt_path = f"{CFG['ckpt_dir']}/step_{global_step:06d}.pt"
                save_checkpoint(
                    dit, ema_dit, optimizer, scheduler_lr,
                    step=global_step,
                    loss=loss.item(),
                    path=ckpt_path,
                )
                artifact_every = int(CFG["wandb_artifact_every"])
                if (
                    artifact_every > 0
                    and global_step % (CFG["save_every"] * artifact_every) == 0
                ):
                    artifact = wandb.Artifact(
                        name=f"dit-checkpoint-{global_step}",
                        type="model",
                        description=f"DiT checkpoint at step {global_step}",
                        metadata={"step": global_step, "loss": loss.item()},
                    )
                    artifact.add_file(ckpt_path)
                    wandb.log_artifact(artifact)

            # Stop at the configured absolute total step count.
            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(
        dit, ema_dit, optimizer, scheduler_lr,
        step=global_step,
        loss=loss.item(),
        path=f"{CFG['ckpt_dir']}/final.pt",
    )
    print(f"\nTraining complete. Final model saved → {CFG['ckpt_dir']}/final.pt")
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_path=args.resume)
