"""EchoMimic V3 Flash Transformer built on the MLX Wan 2.1 layers."""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .audio import AudioProjection, split_audio_for_latents
from .wan.layers import Head, WanAttentionBlock
from .wan.model import WanModel, sinusoidal_embedding_1d


class EchoMimicModel(nn.Module):
    """MLX representation of the released EchoMimic V3 Flash Transformer."""

    def __init__(
        self,
        *,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        audio_window: int = 5,
        audio_blocks: int = 12,
        audio_channels: int = 768,
        audio_intermediate_dim: int = 512,
        audio_context_tokens: int = 32,
        vae_scale: int = 4,
    ) -> None:
        super().__init__()
        if text_len != 512:
            raise ValueError("the released EchoMimic checkpoint requires text_len=512")

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.freq_dim = freq_dim
        self.audio_window = audio_window
        self.vae_scale = vae_scale

        self.audio_injection = AudioProjection(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            blocks=audio_blocks,
            channels=audio_channels,
            intermediate_dim=audio_intermediate_dim,
            output_dim=dim,
            context_tokens=audio_context_tokens,
            norm_output_audio=True,
        )

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approx="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        clip_dim = 1280
        self.img_emb_norm1 = nn.LayerNorm(clip_dim)
        self.img_emb_linear1 = nn.Linear(clip_dim, clip_dim)
        self.img_emb_linear2 = nn.Linear(clip_dim, dim)
        self.img_emb_norm2 = nn.LayerNorm(dim)

        self.blocks = [
            WanAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                cross_attn_norm,
                eps,
                cross_attn_type="i2v_audio",
            )
            for _ in range(num_layers)
        ]
        self.head = Head(dim, out_dim, patch_size, eps)

    def _embed_image(self, clip_features: mx.array) -> mx.array:
        x = self.img_emb_norm1(clip_features)
        x = self.img_emb_linear1(x)
        x = nn.gelu(x)
        x = self.img_emb_linear2(x)
        return self.img_emb_norm2(x)

    def compute_time_embedding(self, timestep: mx.array) -> tuple[mx.array, mx.array]:
        embedding = sinusoidal_embedding_1d(self.freq_dim, timestep)
        time_embedding = self.time_embedding(embedding)
        return time_embedding, self.time_projection(time_embedding)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        *,
        audio: mx.array,
        clip_fea: mx.array,
        first_frame: mx.array,
        block_residual: mx.array | None = None,
        precomputed_time: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run one denoiser evaluation using channels-last MLX tensors.

        Args:
            x: Noisy video latents ``[T, H, W, 16]``.
            t: Diffusion timestep ``[1]``.
            context: Padded UMT5 tokens ``[512, text_dim]``.
            audio: Wav2Vec windows ``[1, source_frames, 5, 12, 768]``.
            clip_fea: Image encoder output ``[1, 257, 1280]``.
            first_frame: Wan I2V condition ``[T, H, W, 20]``.
        """

        if context.shape[0] != self.text_len:
            raise ValueError(f"expected {self.text_len} text tokens, got {context.shape[0]}")
        if x.shape[:-1] != first_frame.shape[:-1]:
            raise ValueError(
                f"latent and first-frame grids must match: {x.shape} vs {first_frame.shape}"
            )
        if x.shape[-1] + first_frame.shape[-1] != self.in_dim:
            raise ValueError(
                "latent and conditioning channels do not match patch embedding: "
                f"{x.shape[-1]} + {first_frame.shape[-1]} != {self.in_dim}"
            )

        x = mx.concatenate([x, first_frame], axis=-1)
        x = self.patch_embedding(x[None])
        _, latent_frames, latent_height, latent_width, _ = x.shape
        grid_sizes = [[latent_frames, latent_height, latent_width]]
        x = x.reshape(1, latent_frames * latent_height * latent_width, self.dim)

        first_audio, following_audio = split_audio_for_latents(
            audio, vae_scale=self.vae_scale, audio_window=self.audio_window
        )
        audio_context = self.audio_injection(first_audio, following_audio)
        if audio_context.shape[1] != latent_frames:
            raise ValueError(
                "audio/latent frame mismatch: "
                f"audio has {audio_context.shape[1]}, latents have {latent_frames}"
            )

        text_context = self.text_embedding(context[None])
        image_context = self._embed_image(clip_fea)
        joint_context = mx.concatenate([image_context, text_context], axis=1)

        if precomputed_time is None:
            time_embedding, modulation = self.compute_time_embedding(t)
        else:
            time_embedding, modulation = precomputed_time
        # The released Flash forward computes the timestep MLP in float32 and
        # then casts both tensors back to the latent dtype before the blocks.
        time_embedding = time_embedding.astype(x.dtype)
        modulation = modulation.astype(x.dtype)
        modulation = modulation.reshape(1, 6, self.dim)

        if block_residual is not None:
            x = x + block_residual
            new_residual = block_residual
        else:
            x_in = x
            block_context = (joint_context, audio_context, latent_frames)
            for block in self.blocks:
                x = block(x, modulation, grid_sizes, block_context)
            new_residual = x - x_in

        x = self.head(x, time_embedding)
        patch_t, patch_h, patch_w = self.patch_size
        output = rearrange(
            x[0],
            "(ft fh fw) (pt ph pw c) -> (ft pt) (fh ph) (fw pw) c",
            ft=latent_frames,
            fh=latent_height,
            fw=latent_width,
            pt=patch_t,
            ph=patch_h,
            pw=patch_w,
        )
        return output, new_residual

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Reuse Apple's Wan conversion while preserving EchoMimic audio keys."""

        return WanModel.sanitize(weights)


@partial(mx.compile, shapeless=True)
def audio_cfg(
    noise_unconditional: mx.array,
    noise_text: mx.array,
    noise_audio: mx.array,
    text_scale: float,
    audio_scale: float,
) -> mx.array:
    """Combine unconditional, text-conditioned, and audio-conditioned predictions."""

    return (
        noise_unconditional
        + text_scale * (noise_text - noise_unconditional)
        + audio_scale * (noise_audio - noise_text)
    )


@partial(mx.compile, shapeless=True)
def flash_audio_cfg(
    noise_without_audio: mx.array,
    noise_with_audio: mx.array,
    audio_scale: float,
) -> mx.array:
    """Apply the two-branch audio CFG used by the released Flash pipeline."""

    return noise_without_audio + audio_scale * (noise_with_audio - noise_without_audio)
