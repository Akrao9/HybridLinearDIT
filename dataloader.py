"""
dataloader.py
─────────────────────────────────────────────────────────────────────────────
Data loading utilities for ScratchDiffusion.

Supports two training paths:
  - raw image streaming from the source WebDataset shards
  - cached latent/text WebDataset shards for faster iteration

Frozen encoder stack:
  - Text encoder : google-t5/t5-base
  - Image codec  : mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers

Latent shape : [B, 32, 16, 16]   (512px input, 32× spatial compression)
Text shape   : [B, 384, 768]     (384 tokens, 768 hidden dim)
"""

import io
import json
import random
import urllib.request

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

try:
    import webdataset as wds
except ImportError:
    wds = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
LATENT_COMPRESSION = 32
RESOLUTION     = 512
LATENT_SIZE    = RESOLUTION // LATENT_COMPRESSION
TRAIN_SAMPLES  = 2_000_000
VAL_SAMPLES    = 2_000
TRAIN_BATCH    = 256
VAL_BATCH      = 256
NUM_WORKERS    = 16
CAPTION_DROP   = 0.1       # drop caption with this prob → enables CFG
MAX_SEQ_LEN    = 384
SEED           = 42
DEVICE         = torch.device("cuda")

# ── Dataset — wildcard pattern (the ONLY way that works for this dataset) ─────
# Explicit URL lists cause datasets to resolve them as hf:// paths and fail.
# Wildcard patterns are resolved correctly as https:// streams.
SUBSET = "synthetic_enhanced_prompt_random_resolution"

# Single wildcard URL — streams all shards, no shard count needed
TRAIN_URL_PATTERN = (
    f"https://huggingface.co/datasets/ma-xu/fine-t2i"
    f"/resolve/main/{SUBSET}/train-*.tar"
)

# For val we use a fixed small set of shards via explicit https:// URLs.
# These bypass hf:// resolution and work correctly.
# Adjust range if you know the exact shard count.
VAL_URL_PATTERN = (
    f"https://huggingface.co/datasets/ma-xu/fine-t2i"
    f"/resolve/main/{SUBSET}/train-*.tar"
)

TEXT_ENCODER_ID = "google-t5/t5-base"
DC_AE_ID        = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
LATENT_CACHE_REPO_ID = "akrao9/512t2ilatent"
LATENT_CACHE_LOADER_BACKEND = "hf_streaming"
LATENT_CACHE_SHUFFLE_BUFFER = 10_000
PIN_MEMORY     = True


# ─────────────────────────────────────────────────────────────────────────────
# Load frozen encoders
# ─────────────────────────────────────────────────────────────────────────────
def load_encoders(device=DEVICE):
    """
    Load and freeze both encoders. Call once at startup.
    Returns (tokenizer, text_encoder, dc_ae, scaling_factor).
    """
    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import AutoencoderDC

    # ── T5-base ───────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    text_encoder = T5EncoderModel.from_pretrained(
        TEXT_ENCODER_ID,
        torch_dtype=torch.bfloat16,
    )
    # Keep tokenizer/model padding aligned for masked text pooling.
    text_encoder.config.pad_token_id = tokenizer.pad_token_id
    text_encoder = text_encoder.to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # ── DC-AE f32c32 sana-1.1 ────────────────────────────────────────────────
    dc_ae = AutoencoderDC.from_pretrained(
        DC_AE_ID,
        torch_dtype=torch.bfloat16,
    )
    dc_ae = dc_ae.to(device).eval()
    for p in dc_ae.parameters():
        p.requires_grad = False

    scaling_factor = dc_ae.config.scaling_factor

    text_hidden_dim = getattr(text_encoder.config, "d_model", None)
    if text_hidden_dim is None:
        text_hidden_dim = text_encoder.config.hidden_size
    print(f"[encoders] text hidden dim    : {text_hidden_dim}")
    print(f"[encoders] DC-AE scaling      : {scaling_factor:.5f}")
    print(f"[encoders] latent shape ({RESOLUTION}) : [B, 32, {LATENT_SIZE}, {LATENT_SIZE}]")
    print(f"[encoders] VRAM after load    : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return tokenizer, text_encoder, dc_ae, scaling_factor


# ─────────────────────────────────────────────────────────────────────────────
# GPU encoding helpers  (call inside training loop, NOT in DataLoader workers)
# ─────────────────────────────────────────────────────────────────────────────
def make_encode_images(
    dc_ae,
    scaling_factor,
    device=DEVICE,
    microbatch_size: int | None = None,
):
    @torch.no_grad()
    def encode_images(pixel_values: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float32 at the configured resolution → [B,32,h,w] bfloat16."""
        chunks = (
            pixel_values.split(microbatch_size)
            if microbatch_size and microbatch_size < pixel_values.shape[0]
            else (pixel_values,)
        )
        latents = []
        for chunk in chunks:
            x = chunk.to(device, dtype=torch.bfloat16, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                latents.append(dc_ae.encode(x).latent * scaling_factor)
        return torch.cat(latents, dim=0)
    return encode_images


def make_encode_text(
    text_encoder,
    tokenizer,
    device=DEVICE,
    microbatch_size: int | None = None,
):
    @torch.no_grad()
    def encode_text(captions: list) -> tuple:
        """list[str] → (hidden [B,384,768] bfloat16, mask [B,384])"""
        chunks = (
            [captions[i:i + microbatch_size] for i in range(0, len(captions), microbatch_size)]
            if microbatch_size and microbatch_size < len(captions)
            else [captions]
        )
        hidden_chunks = []
        mask_chunks = []
        for caption_chunk in chunks:
            tokens = tokenizer(
                caption_chunk,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                hidden = text_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).last_hidden_state.to(torch.bfloat16)
            hidden_chunks.append(hidden)
            mask_chunks.append(tokens["attention_mask"])
        return torch.cat(hidden_chunks, dim=0), torch.cat(mask_chunks, dim=0)
    return encode_text


def make_decode_latents(dc_ae, scaling_factor, device=DEVICE):
    @torch.no_grad()
    def decode_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        [B,32,h,w] bfloat16 -> [B,3,H,W] float32 clamped to [-1,1]
        DC-AE output can slightly exceed [-1,1] so clamp is required
        (official DC-AE recommendation).
        """
        with autocast("cuda", dtype=torch.bfloat16):
            images = dc_ae.decode(latents / scaling_factor).sample
        images = torch.clamp(images.float(), -1.0, 1.0)
        return images
    return decode_latents


def make_latent_cache_urls(
    repo_id: str = LATENT_CACHE_REPO_ID,
    subset: str = SUBSET,
) -> dict[str, str]:
    """Construct direct HTTPS URLs for a latent-cache dataset repo."""
    base = (
        f"https://huggingface.co/datasets/{repo_id}"
        f"/resolve/main/{subset}"
    )
    return {
        "train": f"{base}/train-*.tar",
        "null_text": f"{base}/null_text.npy",
        "null_mask": f"{base}/null_mask.npy",
        "manifest": f"{base}/manifest.json",
    }


def list_latent_cache_shard_urls(
    repo_id: str = LATENT_CACHE_REPO_ID,
    subset: str = SUBSET,
) -> list[str]:
    """List explicit tar shard URLs for a latent-cache dataset repo."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shard_paths = sorted(
        path for path in files
        if path.startswith(f"{subset}/train-") and path.endswith(".tar")
    )
    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
        for path in shard_paths
    ]


def load_cached_null_condition(
    repo_id: str = LATENT_CACHE_REPO_ID,
    subset: str = SUBSET,
    device=DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached unconditional text embedding + mask from the latent-cache repo."""
    urls = make_latent_cache_urls(repo_id=repo_id, subset=subset)

    def load_remote_npy(url: str) -> np.ndarray:
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        return np.load(io.BytesIO(payload), allow_pickle=False)

    null_hidden = torch.from_numpy(load_remote_npy(urls["null_text"])).to(
        device=device, dtype=torch.bfloat16
    )
    null_mask = torch.from_numpy(load_remote_npy(urls["null_mask"])).to(
        device=device, dtype=torch.bool
    )
    return null_hidden, null_mask


# ─────────────────────────────────────────────────────────────────────────────
# Image transform  (CPU, runs in DataLoader workers)
# ─────────────────────────────────────────────────────────────────────────────
_image_transform = transforms.Compose([
    transforms.Resize(
        RESOLUTION,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.CenterCrop(RESOLUTION),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess  (CPU, batched, runs in DataLoader workers)
# ─────────────────────────────────────────────────────────────────────────────
def _make_preprocess(is_train: bool):
    def preprocess(batch: dict) -> dict:
        batch["pixel_values"] = [
            _image_transform(img.convert("RGB")) for img in batch["jpg"]
        ]
        captions = []
        for c in batch["txt"]:
            if is_train and random.random() < CAPTION_DROP:
                c = ""
            captions.append(c)
        batch["captions"] = captions
        batch["aesthetic_score"] = [
            m["aesthetic_predictor_v_2_5_score"] for m in batch["json"]
        ]
        return batch
    return preprocess


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def _collate_fn(batch: list) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values":    pixel_values,                                         # [B,3,512,512]
        "captions":        [b["captions"] for b in batch],                       # list[str]
        "aesthetic_score": torch.tensor([b["aesthetic_score"] for b in batch]),  # [B]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────
def _make_dataset(
    url_pattern: str,
    is_train: bool,
    max_samples: int | None,
    skip_samples: int = 0,
    shuffle: bool = False,
):
    ds = load_dataset(
        "webdataset",
        data_files={"train": url_pattern},   # wildcard string, NOT a list
        split="train",
        streaming=True,
    )
    # ⚠️  shuffle BEFORE skip/take — take() locks shard order
    if shuffle:
        ds = ds.shuffle(seed=SEED, buffer_size=10_000)
    if skip_samples:
        ds = ds.skip(skip_samples)
    if max_samples is not None:
        ds = ds.take(max_samples)
    ds = ds.map(
        _make_preprocess(is_train),
        batched=True,
        batch_size=32,
        remove_columns=["jpg", "txt", "json"],
    )
    ds = ds.with_format("torch")
    return ds


def _make_cached_dataset(
    url_pattern: str | list[str],
    max_samples: int | None,
    skip_samples: int = 0,
    shuffle: bool = False,
    shuffle_buffer: int = LATENT_CACHE_SHUFFLE_BUFFER,
):
    ds = load_dataset(
        "webdataset",
        data_files={"train": url_pattern},
        split="train",
        streaming=True,
    )
    if shuffle:
        ds = ds.shuffle(seed=SEED, buffer_size=shuffle_buffer)
    if skip_samples:
        ds = ds.skip(skip_samples)
    if max_samples is not None:
        ds = ds.take(max_samples)
    return ds


def _match_cached_sample_field(sample: dict, *names: str):
    for name in names:
        if name in sample:
            return sample[name]
    for key, value in sample.items():
        for name in names:
            if key.endswith(name):
                return value
    raise KeyError(
        f"Missing cached field {names!r}; available keys: {sorted(sample.keys())}"
    )


def _normalize_cached_sample(sample: dict) -> dict:
    return {
        "latents.npy": _match_cached_sample_field(sample, "latents.npy", "latents"),
        "text.npy": _match_cached_sample_field(sample, "text.npy", "text"),
        "text_mask.npy": _match_cached_sample_field(
            sample, "text_mask.npy", "text_mask"
        ),
        "caption.txt": _match_cached_sample_field(sample, "caption.txt", "caption"),
        "meta.json": _match_cached_sample_field(sample, "meta.json", "meta"),
    }


def _make_cached_dataset_wds(
    url_pattern: str | list[str],
    max_samples: int | None,
    skip_samples: int = 0,
    shuffle: bool = False,
    shuffle_buffer: int = LATENT_CACHE_SHUFFLE_BUFFER,
):
    if wds is None:
        raise ImportError(
            "raw_webdataset backend requested, but `webdataset` is not installed. "
            "Install it with `%pip install webdataset`."
        )
    if skip_samples:
        raise ValueError(
            "raw_webdataset backend does not support skip_samples yet."
        )

    handler = getattr(wds, "warn_and_continue", None)
    shardshuffle = shuffle_buffer if shuffle else 0

    ds = wds.WebDataset(
        url_pattern,
        shardshuffle=shardshuffle,
        handler=handler,
    )
    if shuffle and shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, rng=random.Random(SEED))
    ds = ds.map(_normalize_cached_sample, handler=handler)
    if max_samples is not None:
        if hasattr(ds, "with_epoch"):
            ds = ds.with_epoch(max_samples)
    return ds


def _decode_npy_field(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray)):
        return np.load(io.BytesIO(value), allow_pickle=False)
    return np.asarray(value)


def _decode_text_field(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_json_field(value) -> dict:
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _collate_cached_fn(batch: list) -> dict:
    latents = torch.from_numpy(
        np.stack([_decode_npy_field(b["latents.npy"]) for b in batch])
    ).contiguous()
    text_hidden = torch.from_numpy(
        np.stack([_decode_npy_field(b["text.npy"]) for b in batch])
    ).contiguous()
    text_mask = torch.from_numpy(
        np.stack([_decode_npy_field(b["text_mask.npy"]) for b in batch])
    ).to(torch.bool)
    captions = [_decode_text_field(b["caption.txt"]) for b in batch]
    meta = [_decode_json_field(b["meta.json"]) for b in batch]
    aesthetic = torch.tensor(
        [m.get("aesthetic_score", 1.0) for m in meta],
        dtype=torch.float32,
    )
    return {
        "latents": latents,                 # [B,32,16,16] float16
        "text_hidden": text_hidden,         # [B,384,768] float16
        "text_mask": text_mask,             # [B,384] bool
        "captions": captions,               # list[str]
        "aesthetic_score": aesthetic,       # [B] float32
        "meta": meta,                       # list[dict]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(
    train_batch:        int = TRAIN_BATCH,
    val_batch:          int = VAL_BATCH,
    num_workers:        int = NUM_WORKERS,
    val_num_workers:    int | None = None,
    prefetch_factor:    int = 2,
    persistent_workers: bool = True,
    train_samples:      int = TRAIN_SAMPLES,
    val_samples:        int = VAL_SAMPLES,
    train_url_pattern:  str = TRAIN_URL_PATTERN,
    val_url_pattern:    str = VAL_URL_PATTERN,
    pin_memory:         bool = PIN_MEMORY,
) -> tuple:
    """
    Returns (train_loader, val_loader).

    Each batch contains:
        pixel_values    : [B, 3, 512, 512]  float32  in [-1, 1]
        captions        : list[str]  len=B  (10% empty for CFG)
        aesthetic_score : [B]        float

    Note: when train and val share the same URL pattern, both are derived
    from the same deterministic shuffled stream so skip+take stays disjoint.
    """
    if val_num_workers is None:
        val_num_workers = min(4, num_workers)

    train_loader_kwargs = dict(
        batch_size=train_batch,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader_kwargs = dict(
        batch_size=val_batch,
        num_workers=val_num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["persistent_workers"] = persistent_workers
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["persistent_workers"] = persistent_workers

    shared_stream = train_url_pattern == val_url_pattern

    train_ds = _make_dataset(
        train_url_pattern,
        is_train=True,
        max_samples=train_samples,
        shuffle=True,
    )
    val_ds = _make_dataset(
        val_url_pattern,
        is_train=False,
        max_samples=val_samples,
        skip_samples=train_samples if shared_stream else 0,
        shuffle=shared_stream,
    )

    train_loader = DataLoader(
        train_ds,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        **val_loader_kwargs,
    )

    steps = train_samples // train_batch
    print(f"[loader] subset         : {SUBSET}")
    print(f"[loader] train samples  : {train_samples:,}")
    print(f"[loader] val samples    : {val_samples:,}")
    print(f"[loader] batch size     : {train_batch}")
    print(f"[loader] steps / epoch  : {steps:,}")

    return train_loader, val_loader


def build_cached_loaders(
    train_batch:        int = TRAIN_BATCH,
    val_batch:          int = VAL_BATCH,
    num_workers:        int = NUM_WORKERS,
    val_num_workers:    int | None = None,
    prefetch_factor:    int = 2,
    persistent_workers: bool = True,
    train_samples:      int = TRAIN_SAMPLES,
    val_samples:        int = VAL_SAMPLES,
    repo_id:            str = LATENT_CACHE_REPO_ID,
    subset:             str = SUBSET,
    train_url_pattern:  str | None = None,
    val_url_pattern:    str | None = None,
    val_shards:         int = 4,
    loader_backend:     str = LATENT_CACHE_LOADER_BACKEND,
    shuffle_buffer:     int = LATENT_CACHE_SHUFFLE_BUFFER,
    pin_memory:         bool = PIN_MEMORY,
) -> tuple:
    """
    Returns (train_loader, val_loader) for precomputed latent-cache shards.

    Each batch contains:
        latents         : [B, 32, 16, 16]  float16
        text_hidden     : [B, 384, 768]    float16
        text_mask       : [B, 384]         bool
        captions        : list[str]
        aesthetic_score : [B]              float32
    """
    if val_num_workers is None:
        val_num_workers = min(4, num_workers)

    urls = make_latent_cache_urls(repo_id=repo_id, subset=subset)
    reserved_val_shards = None
    if train_url_pattern is None and val_url_pattern is None:
        try:
            shard_urls = list_latent_cache_shard_urls(repo_id=repo_id, subset=subset)
        except Exception as exc:
            print(
                "[cached-loader] shard listing failed; "
                f"falling back to wildcard stream ({exc})"
            )
            train_url_pattern = urls["train"]
            val_url_pattern = train_url_pattern
        else:
            if len(shard_urls) >= 2:
                reserved_val_shards = min(max(1, val_shards), len(shard_urls) - 1)
                train_url_pattern = shard_urls[:-reserved_val_shards]
                val_url_pattern = shard_urls[-reserved_val_shards:]
            else:
                train_url_pattern = urls["train"]
                val_url_pattern = train_url_pattern
    else:
        train_url_pattern = train_url_pattern or urls["train"]
        val_url_pattern = val_url_pattern or train_url_pattern

    train_loader_kwargs = dict(
        batch_size=train_batch,
        num_workers=num_workers,
        collate_fn=_collate_cached_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader_kwargs = dict(
        batch_size=val_batch,
        num_workers=val_num_workers,
        collate_fn=_collate_cached_fn,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["persistent_workers"] = persistent_workers
    if val_num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["persistent_workers"] = persistent_workers

    shared_stream = train_url_pattern == val_url_pattern

    if loader_backend == "hf_streaming":
        dataset_builder = _make_cached_dataset
    elif loader_backend == "raw_webdataset":
        dataset_builder = _make_cached_dataset_wds
    else:
        raise ValueError(
            f"Unknown latent-cache loader backend: {loader_backend!r}"
        )

    train_ds = dataset_builder(
        train_url_pattern,
        max_samples=train_samples,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
    )
    # When train/val share one streamed shard set, skipping train_samples makes the
    # first validation pass walk past the entire training split over HTTP. That can
    # look like a hang around val_every steps. Use the head of the stream as a small
    # fixed validation slice unless the caller provides a distinct val_url_pattern.
    val_ds = dataset_builder(
        val_url_pattern,
        max_samples=val_samples,
        skip_samples=0,
        shuffle=False,
        shuffle_buffer=shuffle_buffer,
    )

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    steps = train_samples // train_batch
    print(f"[cached-loader] repo        : {repo_id}")
    print(f"[cached-loader] subset      : {subset}")
    print(f"[cached-loader] backend     : {loader_backend}")
    print(f"[cached-loader] train samples : {train_samples:,}")
    print(f"[cached-loader] val samples   : {val_samples:,}")
    print(f"[cached-loader] batch size    : {train_batch}")
    print(f"[cached-loader] steps / epoch : {steps:,}")
    print(f"[cached-loader] shuffle buf   : {shuffle_buffer:,}")
    print(f"[cached-loader] pin_memory    : {pin_memory}")
    if reserved_val_shards is not None:
        print(f"[cached-loader] val shards    : {reserved_val_shards}")
        print("[cached-loader] val source    : held-out tail shards")
    elif shared_stream:
        print("[cached-loader] val source    : shared stream head (no skip)")
    else:
        print("[cached-loader] val source    : dedicated val stream")

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check  —  python dataloader.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load encoders ─────────────────────────────────────────────────────────
    tokenizer, text_encoder, dc_ae, scaling_factor = load_encoders()
    encode_images  = make_encode_images(dc_ae, scaling_factor)
    encode_text    = make_encode_text(text_encoder, tokenizer)
    decode_latents = make_decode_latents(dc_ae, scaling_factor)

    # ── Tiny loaders for quick check ─────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        train_batch=4,
        val_batch=4,
        num_workers=2,
        train_samples=20,
        val_samples=8,
    )

    # ── Inspect one train batch ───────────────────────────────────────────────
    batch = next(iter(train_loader))
    print("\n── Train batch ──────────────────────────────")
    print(f"pixel_values    : {batch['pixel_values'].shape}")    # [4,3,512,512]
    print(f"captions[0]     : {batch['captions'][0][:80]}...")
    print(f"aesthetic_score : {batch['aesthetic_score']}")

    # ── GPU encode ────────────────────────────────────────────────────────────
    latents        = encode_images(batch["pixel_values"])
    text_hid, mask = encode_text(batch["captions"])
    print(f"\nlatents         : {latents.shape}")                # [4,32,16,16]
    print(f"text_hidden     : {text_hid.shape}")                # [4,384,768]
    print(f"attention_mask  : {mask.shape}")                    # [4,384]

    # ── Decode back ───────────────────────────────────────────────────────────
    recon = decode_latents(latents)
    print(f"decoded images  : {recon.shape}")                   # [4,3,512,512]
    print(f"\nFinal VRAM      : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Display original vs decoded ───────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from torchvision.utils import save_image

    def tensor_to_numpy(t):
        """Convert a single [-1,1] CHW tensor → HWC uint8 numpy for matplotlib."""
        t = t.float().cpu()
        t = (t * 0.5 + 0.5).clamp(0, 1)     # [-1,1] → [0,1]
        return (t.permute(1, 2, 0).numpy() * 255).astype("uint8")

    B = batch["pixel_values"].shape[0]

    # ── save_image — official DC-AE style (image * 0.5 + 0.5 normalises to [0,1]) ──
    orig_grid  = batch["pixel_values"].float().cpu()
    recon_grid = recon.float().cpu()

    save_image(orig_grid  * 0.5 + 0.5, "originals.png",      nrow=B)
    save_image(recon_grid * 0.5 + 0.5, "dc_ae_decoded.png",  nrow=B)
    print("Saved → originals.png")
    print("Saved → dc_ae_decoded.png")

    # ── matplotlib — side-by-side original vs decoded with captions ──────────
    fig = plt.figure(figsize=(5 * B, 12))
    gs  = gridspec.GridSpec(
        3, B,
        hspace=0.4, wspace=0.05,
        height_ratios=[1, 1, 0.08],   # original | decoded | caption row
    )

    for i in range(B):
        caption   = batch["captions"][i]
        score     = batch["aesthetic_score"][i].item()

        orig_np   = tensor_to_numpy(batch["pixel_values"][i])
        recon_np  = tensor_to_numpy(recon[i])

        # ── Original ──────────────────────────────────────────────────────────
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(orig_np)
        ax_orig.axis("off")
        if i == 0:
            ax_orig.set_ylabel("Original", fontsize=11, labelpad=6)
        ax_orig.set_title(f"Score: {score:.2f}", fontsize=9)

        # ── DC-AE encode → decode ─────────────────────────────────────────────
        ax_recon = fig.add_subplot(gs[1, i])
        ax_recon.imshow(recon_np)
        ax_recon.axis("off")
        if i == 0:
            ax_recon.set_ylabel("DC-AE Decoded", fontsize=11, labelpad=6)

        # ── Caption below each column ─────────────────────────────────────────
        ax_cap = fig.add_subplot(gs[2, i])
        ax_cap.axis("off")
        ax_cap.text(
            0.5, 0.5,
            caption[:120] + ("..." if len(caption) > 120 else ""),
            ha="center", va="center",
            fontsize=7, wrap=True,
            transform=ax_cap.transAxes,
        )

    fig.suptitle(
        f"Original  vs  DC-AE Encode→Decode  "
        f"(latent: {latents.shape[1]}×{latents.shape[2]}×{latents.shape[3]})",
        fontsize=12, y=1.01,
    )

    plt.savefig("encode_decode_check.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → encode_decode_check.png")

    # ── Latent heatmap (visualise what DC-AE compressed to) ───────────────────
    fig2, axes = plt.subplots(B, 4, figsize=(12, 3 * B))
    if B == 1:
        axes = axes[None]   # keep 2D indexing

    for i in range(B):
        lat = latents[i].float().cpu()   # [32, 16, 16]
        for j, ch in enumerate([0, 8, 16, 24]):
            ax = axes[i, j]
            ax.imshow(lat[ch], cmap="RdBu_r", interpolation="nearest")
            ax.axis("off")
            ax.set_title(f"img {i} · ch {ch}", fontsize=8)

    fig2.suptitle(
        "Latent channels (4 of 32 shown)  —  "
        f"shape {list(latents.shape[1:])}  scaling={scaling_factor:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("latent_channels.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → latent_channels.png")

    # ── Inspect one val batch ─────────────────────────────────────────────────
    val_batch = next(iter(val_loader))
    print(f"\n── Val batch ────────────────────────────────")
    print(f"pixel_values    : {val_batch['pixel_values'].shape}")
    print(f"captions[0]     : {val_batch['captions'][0][:80]}...")
