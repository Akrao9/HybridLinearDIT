"""
model.py — LinearDiT v2
─────────────────────────────────────────────────────────────────────────────
Hybrid-attention Flow-Matching Diffusion Transformer.

Key design choices:
  - Token funnel     : image + text tokens are processed together through most
                       of the stack; text can optionally be dropped near the end
  - Hybrid attention : mostly linear attention (O(N)), with several evenly
                       spaced full softmax anchor blocks for global reasoning
  - Text cross-attn  : only full-attn blocks get image→text cross-attention
  - Per-block adaLN  : shared conditioning trunk + block-specific adaLN heads
  - Mix-FFN + NoPE   : 3×3 depthwise conv inside FFN gives spatial awareness
  - Per-block QKV    : each block has its own self-attention QKV projection
  - Flow Matching    : output is velocity v = ε − x₀

Input / output contract (matches train.py):
    forward(x, t, text_hidden, text_mask)
      x           : [B, 32, 16, 16]   noisy DC-AE latents
      t           : [B]               float timesteps in [0, 1]
      text_hidden : [B, 384, 768]     T5-base encoder hidden state
      text_mask   : [B, 384]          attention mask (1=real token, 0=pad)
    returns       : [B, 32, 16, 16]   predicted velocity
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


NUM_BLOCKS = 16
DEFAULT_NUM_FULL_ATTN = 4


def default_text_drop_block(n_blocks: int) -> int:
    """Keep text tokens alive through the full stack by default."""
    return max(0, n_blocks)


def default_full_attn_blocks(
    n_blocks: int,
    count: int = DEFAULT_NUM_FULL_ATTN,
) -> tuple[int, ...]:
    """Evenly space a few softmax-attention anchor blocks through the stack."""
    count = max(1, min(count, n_blocks))
    anchors = {
        max(0, round((i + 1) * n_blocks / count) - 1)
        for i in range(count)
    }
    return tuple(sorted(anchors))


class TimestepEmbedding(nn.Module):
    """Continuous timestep embedding for flow matching."""

    def __init__(self, dim: int, freq_dim: int = 256, t_scale: float = 1000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.t_scale = t_scale
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype
        t = t.to(compute_dtype) * self.t_scale
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(1000.0)
            * torch.arange(half, device=t.device, dtype=compute_dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        sinusoidal = self._sinusoidal(t)
        return self.proj(sinusoidal.to(self.proj[0].weight.dtype))


class CondTrunk(nn.Module):
    """
    Shared conditioning trunk: [time + pooled text] → trunk hidden state.
    Each block has its own adaLN head that reads this shared trunk output.
    """

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.trunk_dim = dim // 2
        self.trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, self.trunk_dim),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.trunk(cond)


class AdaLNHead(nn.Module):
    """Per-block head: trunk_hidden → 6 modulation params."""

    def __init__(self, trunk_dim: int, dim: int):
        super().__init__()
        self.linear = nn.Linear(trunk_dim, dim * 6)

    def reset_zero(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.linear(h).chunk(6, dim=-1)


class LinearAttention(nn.Module):
    """Linear attention using ReLU kernel features."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def forward(
        self,
        x: torch.Tensor,
        qkv: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, H, D).transpose(1, 2)
        k = k.reshape(B, N, H, D).transpose(1, 2)
        v = v.reshape(B, N, H, D).transpose(1, 2)

        q = self._kernel(q)
        k = self._kernel(k)

        if key_mask is not None:
            mask = key_mask[:, None, :, None].to(k.dtype)
            k = k * mask
            v = v * mask

        kv = torch.einsum("bhnd,bhnf->bhdf", k, v)
        qkv_ = torch.einsum("bhnd,bhdf->bhnf", q, kv)

        k_sum = k.sum(dim=2)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1) + 1e-6
        out = qkv_ / denom

        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.out_proj(out)


class FullAttention(nn.Module):
    """Standard scaled dot-product self-attention."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        qkv: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, H, D).transpose(1, 2)
        k = k.reshape(B, N, H, D).transpose(1, 2)
        v = v.reshape(B, N, H, D).transpose(1, 2)

        attn_bias = None
        if key_mask is not None:
            attn_bias = torch.zeros((B, 1, 1, N), device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(~key_mask[:, None, None, :], -1e4)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.out_proj(out)


class TextCrossAttention(nn.Module):
    """Image tokens query the full text sequence."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def reset_zero(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N_img, _ = x.shape
        N_txt = text_kv.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, N_img, H, D).transpose(1, 2)
        k = self.k_proj(text_kv).reshape(B, N_txt, H, D).transpose(1, 2)
        v = self.v_proj(text_kv).reshape(B, N_txt, H, D).transpose(1, 2)

        attn_bias = None
        if text_mask is not None:
            attn_bias = torch.zeros((B, 1, 1, N_txt), device=x.device, dtype=x.dtype)
            attn_bias.masked_fill_(~text_mask[:, None, None, :].bool(), -1e4)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, N_img, H * D)
        return self.out_proj(out)


class MixFFN(nn.Module):
    """Mix-FFN with 3×3 depthwise convolution on image tokens only."""

    def __init__(self, dim: int, img_seq: int, expand: int = 4):
        super().__init__()
        self.img_seq = img_seq
        self.h = self.w = int(img_seq ** 0.5)

        inner = dim * expand
        self.fc1 = nn.Linear(dim, inner * 2)
        self.dw_conv = nn.Conv2d(
            inner,
            inner,
            kernel_size=3,
            padding=1,
            groups=inner,
        )
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, n_img: int) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.fc1(x)
        gate, val = h.chunk(2, dim=-1)

        img_val = val[:, :n_img, :]
        text_val = val[:, n_img:, :]

        inner = img_val.shape[-1]
        img_2d = img_val.transpose(1, 2).reshape(B, inner, self.h, self.w)
        img_2d = self.dw_conv(img_2d)
        img_val = img_2d.reshape(B, inner, n_img).transpose(1, 2)

        val = torch.cat([img_val, text_val], dim=1)
        out = F.silu(gate) * val
        return self.fc2(out)


class LinearDiTBlock(nn.Module):
    """
    One transformer block.
      - self-attention: linear or full softmax
      - text cross-attention: only in full-attn blocks
      - per-block adaLN head from shared conditioning trunk
      - Mix-FFN with NoPE spatial bias
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        img_seq: int,
        trunk_dim: int,
        use_full_attn: bool,
    ):
        super().__init__()
        self.use_full_attn = use_full_attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.adaln = AdaLNHead(trunk_dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if use_full_attn:
            self.attn = FullAttention(dim, n_heads)
            self.cross_attn = TextCrossAttention(dim, n_heads)
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.attn = LinearAttention(dim, n_heads)

        self.ffn = MixFFN(dim, img_seq)

    def forward(
        self,
        x: torch.Tensor,
        n_img: int,
        trunk_h: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        text_hidden: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln(trunk_h)

        token_mask_f = None
        if token_mask is not None:
            token_mask_f = token_mask.unsqueeze(-1).to(x.dtype)

        h = self.norm1(x)
        h = h * (1 + scale_pre.unsqueeze(1)) + shift_pre.unsqueeze(1)
        qkv = self.qkv(h)
        h = self.attn(h, qkv, key_mask=token_mask)
        x = x + gate_attn.unsqueeze(1) * h
        if token_mask_f is not None:
            x = x * token_mask_f

        if self.use_full_attn and text_hidden is not None:
            x_img = x[:, :n_img, :]
            h_cross = self.cross_attn(self.norm_cross(x_img), text_hidden, text_mask)
            x_img = x_img + h_cross
            x = torch.cat([x_img, x[:, n_img:, :]], dim=1)

        h = self.norm2(x)
        h = h * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        h = self.ffn(h, n_img)
        x = x + gate_ffn.unsqueeze(1) * h
        if token_mask_f is not None:
            x = x * token_mask_f
        return x


class DiT(nn.Module):
    """
    LinearDiT v2 — Flow-Matching Diffusion Transformer.

    Keeps the earlier hybrid/token-funnel design that empirically worked better
    in this project than the later dual-stream refactor.
    """

    def __init__(
        self,
        latent_ch: int = 32,
        latent_size: int = 16,
        text_dim: int = 768,
        text_seq: int = 384,
        dim: int = 768,
        n_heads: int = 12,
        n_blocks: int = NUM_BLOCKS,
        text_drop_block: int | None = None,
        full_attn_blocks: tuple[int, ...] | list[int] | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.latent_size = latent_size
        self.gradient_checkpointing = gradient_checkpointing

        resolved_drop = default_text_drop_block(n_blocks) if text_drop_block is None else text_drop_block
        self.text_drop_block = min(max(0, int(resolved_drop)), max(0, n_blocks))

        resolved_full = (
            default_full_attn_blocks(n_blocks)
            if full_attn_blocks is None
            else tuple(int(idx) for idx in full_attn_blocks)
        )
        self.full_attn_blocks = {
            min(max(0, idx), max(0, n_blocks - 1))
            for idx in resolved_full
        }
        if not self.full_attn_blocks:
            self.full_attn_blocks = {max(0, n_blocks - 1)}

        self.img_seq = latent_size * latent_size
        self.text_seq = text_seq

        self.patch_embed = nn.Conv2d(latent_ch, dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, dim)

        self.time_embed = TimestepEmbedding(dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.cond_trunk = CondTrunk(cond_dim=dim, dim=dim)
        trunk_dim = self.cond_trunk.trunk_dim

        self.blocks = nn.ModuleList(
            [
                LinearDiTBlock(
                    dim=dim,
                    n_heads=n_heads,
                    img_seq=self.img_seq,
                    trunk_dim=trunk_dim,
                    use_full_attn=(i in self.full_attn_blocks),
                )
                for i in range(n_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Linear(trunk_dim, dim * 2)
        self.out_proj = nn.Linear(dim, latent_ch)

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)

        for block in self.blocks:
            block.adaln.reset_zero()
            if block.use_full_attn:
                block.cross_attn.reset_zero()

        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        img = self.patch_embed(x).flatten(2).transpose(1, 2)
        txt = self.text_proj(text_hidden)
        txt_for_cross = txt

        t_emb = self.time_embed(t)
        if text_mask is not None:
            mask_f = text_mask.to(txt.dtype).unsqueeze(-1)
            txt_pool = (txt * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            txt_pool = txt.mean(dim=1)

        cond = self.cond_proj(torch.cat([t_emb, txt_pool], dim=-1))
        trunk_h = self.cond_trunk(cond)

        x_seq = torch.cat([img, txt], dim=1)
        n_img = self.img_seq
        seq_mask = None
        if text_mask is not None:
            img_mask = torch.ones(B, n_img, device=x.device, dtype=torch.bool)
            txt_mask = text_mask.to(device=x.device, dtype=torch.bool)
            seq_mask = torch.cat([img_mask, txt_mask], dim=1)
            x_seq = x_seq * seq_mask.unsqueeze(-1).to(x_seq.dtype)

        for i, block in enumerate(self.blocks):
            if i == self.text_drop_block:
                x_seq = x_seq[:, :n_img, :]
                seq_mask = None

            if seq_mask is not None:
                x_seq = x_seq * seq_mask.unsqueeze(-1).to(x_seq.dtype)

            if self.training and self.gradient_checkpointing:
                def run_block(
                    x_input,
                    block=block,
                    n_img=n_img,
                    trunk_h=trunk_h,
                    token_mask=seq_mask,
                    text_hidden=txt_for_cross,
                    text_mask=text_mask,
                ):
                    return block(
                        x_input,
                        n_img,
                        trunk_h=trunk_h,
                        token_mask=token_mask,
                        text_hidden=text_hidden,
                        text_mask=text_mask,
                    )

                x_seq = checkpoint(run_block, x_seq, use_reentrant=False)
            else:
                x_seq = block(
                    x_seq,
                    n_img,
                    trunk_h=trunk_h,
                    token_mask=seq_mask,
                    text_hidden=txt_for_cross,
                    text_mask=text_mask,
                )

        img_tokens = x_seq[:, :n_img, :]
        shift, scale = self.final_mod(trunk_h).chunk(2, dim=-1)
        out = self.final_norm(img_tokens)
        out = out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, -1, self.latent_size, self.latent_size)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiT(n_blocks=12).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params     : {total:.1f}M")
    print(f"Trainable params : {train_:.1f}M")
    print(f"Text drop block  : {model.text_drop_block}")
    print(f"Full attn blocks : {sorted(model.full_attn_blocks)}")

    components = {
        "patch_embed": model.patch_embed,
        "text_proj": model.text_proj,
        "time_embed": model.time_embed,
        "cond_proj": model.cond_proj,
        "cond_trunk": model.cond_trunk,
        "blocks": model.blocks,
        "final_mod": model.final_mod,
        "out_proj": model.out_proj,
    }
    print("\nPer-component parameter counts:")
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters()) / 1e6
        print(f"  {name:<20} {n:.2f}M")

    B = 2
    x = torch.randn(B, 32, model.latent_size, model.latent_size, device=device)
    t = torch.rand(B, device=device)
    text_hidden = torch.randn(B, 384, 768, device=device)
    text_mask = torch.ones(B, 384, dtype=torch.long, device=device)
    text_mask[:, 352:] = 0

    with torch.no_grad():
        out = model(x, t, text_hidden, text_mask)

    print(f"\nInput  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    print("✓ Shape check passed")

    if device.type == "cuda":
        print(f"VRAM used    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
