"""
precompute_wds_cache.py
─────────────────────────────────────────────────────────────────────────────
Precompute DC-AE latents and optional T5-base text embeddings into sharded
WebDataset tar files for faster future training.

Recommended shard schema:
  - latents.npy      : float16 [32, 16, 16]   (512px default)
  - text.npy         : float16 [384, 768]      (optional)
  - text_mask.npy    : uint8   [384]           (optional)
  - caption.txt      : UTF-8 prompt text
  - meta.json        : lightweight metadata (aesthetic score, subset, index)

Files written next to the shards:
  - manifest.json    : dataset/cache metadata
  - null_text.npy    : float16 [384, 768]  unconditional embedding for ""
  - null_mask.npy    : uint8   [384]

Notes:
  - This script intentionally stores arrays as float16 `.npy` files for broad
    compatibility with NumPy/WebDataset tooling. If you truly want bfloat16
    storage, you will usually need a custom loader path.
  - We preserve full captions during precompute. Classifier-free guidance
    dropout can be reintroduced later in training by swapping some cached text
    embeddings with the saved null embedding.

Example:
    python precompute_wds_cache.py \
        --split train \
        --output-dir /content/cache/train \
        --max-samples 2000000 \
        --cache-text \
        --batch-size 64 \
        --num-workers 8 \
        --image-encode-batch 16 \
        --text-encode-batch 32 \
        --shard-size-gb 1.0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import io
import json
import os
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import (
    LATENT_SIZE,
    SUBSET,
    TRAIN_URL_PATTERN,
    VAL_URL_PATTERN,
    TEXT_ENCODER_ID,
    DC_AE_ID,
    _collate_fn,
    _make_dataset,
    load_encoders,
    make_encode_images,
    make_encode_text,
)


def parse_args():
    p = argparse.ArgumentParser(description="Precompute latent/text WebDataset cache")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--url-pattern", type=str, default=None,
                   help="Override the source webdataset URL pattern")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory where tar shards + manifest will be written")
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="Number of samples to cache")
    p.add_argument("--skip-samples", type=int, default=0,
                   help="Skip this many samples from the source stream before caching")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle source stream before skip/take")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-encode-batch", type=int, default=16,
                   help="GPU microbatch for DC-AE encode")
    p.add_argument("--text-encode-batch", type=int, default=32,
                   help="GPU microbatch for T5 encode")
    p.add_argument("--cache-text", action="store_true",
                   help="Also cache T5 hidden states + masks")
    p.add_argument("--shard-size-gb", type=float, default=1.0,
                   help="Rotate to a new tar shard at about this size in GB")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--hf-repo-id", type=str, default=None,
                   help="Optional HF dataset repo id to upload shards as they close")
    p.add_argument("--hf-path-prefix", type=str, default=None,
                   help="Optional repo subfolder. Defaults to the split name.")
    p.add_argument("--hf-upload-workers", type=int, default=1,
                   help="Background upload worker count")
    p.add_argument("--shard-start-index", type=int, default=0,
                   help="Starting tar shard index when resuming an upload stream")
    p.add_argument("--delete-uploaded", action="store_true",
                   help="Delete local shard files after successful upload")
    p.add_argument("--poll-interval", type=float, default=10.0,
                   help="Seconds between scans for newly closed shard files")
    return p.parse_args()


def npy_bytes(array: np.ndarray) -> bytes:
    """Serialize a NumPy array into .npy bytes for TarWriter."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def tensor_to_fp16_npy(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().to(torch.float16).contiguous().numpy()
    return npy_bytes(arr)


def tensor_to_u8_npy(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().to(torch.uint8).contiguous().numpy()
    return npy_bytes(arr)


def save_manifest(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class AsyncHubUploader:
    """Upload completed shard files to a HF dataset repo while writing continues."""

    def __init__(
        self,
        output_dir: str,
        split: str,
        repo_id: str | None,
        path_prefix: str | None,
        workers: int,
        delete_uploaded: bool,
        poll_interval: float,
    ):
        self.output_dir = output_dir
        self.split = split
        self.repo_id = repo_id
        self.path_prefix = path_prefix or split
        self.delete_uploaded = delete_uploaded
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._finalize_event = threading.Event()
        self._uploaded: set[str] = set()
        self._lock = threading.Lock()
        self._thread = None
        self._executor = None
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._api = None

        if repo_id is not None:
            from huggingface_hub import HfApi

            self._api = HfApi()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, workers)
            )

    @property
    def enabled(self) -> bool:
        return self.repo_id is not None

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def _upload_path(self, local_path: str):
        assert self._api is not None
        repo_path = os.path.join(self.path_prefix, os.path.basename(local_path)).replace("\\", "/")
        self._api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        if self.delete_uploaded:
            try:
                os.remove(local_path)
            except FileNotFoundError:
                pass

    def _completed_shards(self) -> list[str]:
        pattern = os.path.join(self.output_dir, f"{self.split}-*.tar")
        paths = sorted(glob.glob(pattern))
        if not paths:
            return []
        if self._finalize_event.is_set():
            return paths
        # When a newer shard exists, all but the newest one are closed and safe to upload.
        return paths[:-1]

    def _watch_loop(self):
        while not self._stop_event.is_set():
            self._schedule_ready_shards()
            self._collect_finished()
            if self._finalize_event.is_set():
                if not self._futures and not self._schedule_ready_shards():
                    break
            time.sleep(self.poll_interval)

        self._schedule_ready_shards()
        while self._futures:
            self._collect_finished(wait=True)

    def _schedule_ready_shards(self) -> bool:
        if not self.enabled:
            return False
        scheduled_any = False
        ready = self._completed_shards()
        with self._lock:
            in_flight = set(self._futures)
            already = set(self._uploaded)
        for path in ready:
            if path in in_flight or path in already:
                continue
            assert self._executor is not None
            future = self._executor.submit(self._upload_path, path)
            with self._lock:
                self._futures[path] = future
            print(f"[upload] queued {os.path.basename(path)}")
            scheduled_any = True
        return scheduled_any

    def _collect_finished(self, wait: bool = False):
        if not self.enabled:
            return
        if wait and self._futures:
            concurrent.futures.wait(
                list(self._futures.values()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
        done_paths = []
        with self._lock:
            for path, future in list(self._futures.items()):
                if future.done():
                    done_paths.append(path)
        for path in done_paths:
            future = self._futures.pop(path)
            future.result()
            with self._lock:
                self._uploaded.add(path)
            print(f"[upload] done   {os.path.basename(path)}")

    def upload_small_file(self, local_path: str, repo_name: str | None = None):
        if not self.enabled:
            return
        assert self._api is not None
        repo_path = os.path.join(
            self.path_prefix,
            repo_name or os.path.basename(local_path),
        ).replace("\\", "/")
        self._api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        print(f"[upload] done   {repo_path}")

    def finalize(self):
        if not self.enabled:
            return
        self._finalize_event.set()

    def join(self):
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._executor is not None:
            self._executor.shutdown(wait=True)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    url_pattern = args.url_pattern or (
        TRAIN_URL_PATTERN if args.split == "train" else VAL_URL_PATTERN
    )
    uploader = AsyncHubUploader(
        output_dir=args.output_dir,
        split=args.split,
        repo_id=args.hf_repo_id,
        path_prefix=args.hf_path_prefix,
        workers=args.hf_upload_workers,
        delete_uploaded=args.delete_uploaded,
        poll_interval=args.poll_interval,
    )
    if uploader.enabled:
        print(
            f"Uploading closed shards to hf://datasets/{args.hf_repo_id}/"
            f"{uploader.path_prefix}"
        )
        uploader.start()

    print("Loading encoders...")
    tokenizer, text_encoder, dc_ae, scaling_factor = load_encoders(device)
    encode_images = make_encode_images(
        dc_ae,
        scaling_factor,
        device=device,
        microbatch_size=args.image_encode_batch,
    )
    encode_text = make_encode_text(
        text_encoder,
        tokenizer,
        device=device,
        microbatch_size=args.text_encode_batch,
    )

    print("Building source stream...")
    ds = _make_dataset(
        url_pattern=url_pattern,
        is_train=False,  # preserve full captions in the cache
        max_samples=args.max_samples,
        skip_samples=args.skip_samples,
        shuffle=args.shuffle,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    null_hidden = None
    null_mask = None
    if args.cache_text:
        print("Precomputing unconditional text embedding...")
        null_hidden_t, null_mask_t = encode_text([""])
        null_hidden = null_hidden_t[0]
        null_mask = null_mask_t[0]
        null_text_path = os.path.join(args.output_dir, "null_text.npy")
        null_mask_path = os.path.join(args.output_dir, "null_mask.npy")
        with open(null_text_path, "wb") as f:
            f.write(tensor_to_fp16_npy(null_hidden))
        with open(null_mask_path, "wb") as f:
            f.write(tensor_to_u8_npy(null_mask))
        uploader.upload_small_file(null_text_path)
        uploader.upload_small_file(null_mask_path)

    import webdataset as wds

    pattern = os.path.join(args.output_dir, f"{args.split}-%06d.tar")
    maxsize = int(args.shard_size_gb * (1024 ** 3))
    print(
        f"Writing shards to {pattern} "
        f"(start_shard={args.shard_start_index}, maxsize≈{args.shard_size_gb:.2f} GB)"
    )

    n_written = 0
    started = time.time()
    preview_printed = False
    latents_shape = None
    text_shape = None
    with wds.ShardWriter(
        pattern,
        maxsize=maxsize,
        start_shard=args.shard_start_index,
    ) as sink:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            latents = encode_images(pixel_values)
            if latents_shape is None:
                latents_shape = list(latents.shape[1:])

            text_hidden = None
            text_mask = None
            if args.cache_text:
                text_hidden, text_mask = encode_text(batch["captions"])
                if text_shape is None:
                    text_shape = list(text_hidden.shape[1:])

            if not preview_printed:
                print("\n[preview] first batch tensors before shard writing")
                print(
                    f"  pixel_values : shape={tuple(pixel_values.shape)} "
                    f"dtype={pixel_values.dtype}"
                )
                print(
                    f"  latents      : shape={tuple(latents.shape)} "
                    f"dtype={latents.dtype}"
                )
                latents_np = latents[0].detach().cpu().to(torch.float16).numpy()
                print(
                    f"  latents.npy  : shape={latents_np.shape} "
                    f"dtype={latents_np.dtype}"
                )
                if args.cache_text:
                    print(
                        f"  text_hidden  : shape={tuple(text_hidden.shape)} "
                        f"dtype={text_hidden.dtype}"
                    )
                    print(
                        f"  text_mask    : shape={tuple(text_mask.shape)} "
                        f"dtype={text_mask.dtype}"
                    )
                    text_np = text_hidden[0].detach().cpu().to(torch.float16).numpy()
                    mask_np = text_mask[0].detach().cpu().to(torch.uint8).numpy()
                    print(
                        f"  text.npy     : shape={text_np.shape} "
                        f"dtype={text_np.dtype}"
                    )
                    print(
                        f"  text_mask.npy: shape={mask_np.shape} "
                        f"dtype={mask_np.dtype}"
                    )
                print(
                    f"  caption.txt  : type={type(batch['captions'][0]).__name__} "
                    f"preview={batch['captions'][0][:120]!r}"
                )
                print(
                    "  meta.json    : "
                    + json.dumps({
                        "index": args.skip_samples + n_written,
                        "split": args.split,
                        "subset": SUBSET,
                        "aesthetic_score": float(batch["aesthetic_score"][0].item()),
                    })
                )
                preview_printed = True

            batch_size = latents.shape[0]
            for i in range(batch_size):
                global_idx = args.skip_samples + n_written
                sample = {
                    "__key__": f"{global_idx:09d}",
                    "latents.npy": tensor_to_fp16_npy(latents[i]),
                    "caption.txt": batch["captions"][i],
                    "meta.json": json.dumps({
                        "index": global_idx,
                        "split": args.split,
                        "subset": SUBSET,
                        "aesthetic_score": float(batch["aesthetic_score"][i].item()),
                    }).encode("utf-8"),
                }
                if args.cache_text:
                    sample["text.npy"] = tensor_to_fp16_npy(text_hidden[i])
                    sample["text_mask.npy"] = tensor_to_u8_npy(text_mask[i])

                sink.write(sample)
                n_written += 1

                if n_written % args.log_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = n_written / elapsed
                    print(
                        f"[cache] wrote {n_written:,}/{args.max_samples:,} samples "
                        f"({rate:.2f} samples/s)"
                    )

                if n_written >= args.max_samples:
                    break

            if n_written >= args.max_samples:
                break

    elapsed = time.time() - started
    manifest = {
        "format_version": 1,
        "split": args.split,
        "subset": SUBSET,
        "source_url_pattern": url_pattern,
        "max_samples": args.max_samples,
        "written_samples": n_written,
        "skip_samples": args.skip_samples,
        "shuffle": args.shuffle,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_encode_batch": args.image_encode_batch,
        "text_encode_batch": args.text_encode_batch,
        "cache_text": args.cache_text,
        "shard_start_index": args.shard_start_index,
        "latents_dtype": "float16",
        "latents_shape": latents_shape or [32, LATENT_SIZE, LATENT_SIZE],
        "text_dtype": "float16" if args.cache_text else None,
        "text_shape": text_shape if args.cache_text else None,
        "text_mask_dtype": "uint8" if args.cache_text else None,
        "caption_drop_during_precompute": False,
        "text_encoder_id": TEXT_ENCODER_ID,
        "image_codec_id": DC_AE_ID,
        "null_text_path": "null_text.npy" if args.cache_text else None,
        "null_mask_path": "null_mask.npy" if args.cache_text else None,
        "elapsed_sec": elapsed,
        "samples_per_sec": n_written / max(elapsed, 1e-6),
        "recommended_schema": [
            "latents.npy",
            "text.npy" if args.cache_text else None,
            "text_mask.npy" if args.cache_text else None,
            "caption.txt",
            "meta.json",
        ],
    }
    manifest["recommended_schema"] = [
        key for key in manifest["recommended_schema"] if key is not None
    ]
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    save_manifest(manifest_path, manifest)
    uploader.finalize()
    uploader.join()
    uploader.upload_small_file(manifest_path)

    print("\nCache complete.")
    print(f"  Samples written : {n_written:,}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Elapsed         : {elapsed/60:.1f} min")
    print(f"  Throughput      : {manifest['samples_per_sec']:.2f} samples/s")
    if uploader.enabled:
        print(f"  Uploaded repo   : {args.hf_repo_id}")
        print("  Next step       : point your training loader at the HF dataset repo")
    else:
        print("  Next step       : upload the shard directory as a HF dataset repo")


if __name__ == "__main__":
    main()
