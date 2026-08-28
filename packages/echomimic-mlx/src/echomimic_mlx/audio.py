# Adapted from antgroup/echomimic_v3 and VideoX-Fun at pinned revisions.
# Modified for MLX by echomimic-v3-mlx contributors.
# Licensed under Apache-2.0; see the repository LICENSE.
"""EchoMimic V3 audio grouping and projection in MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def split_audio_for_latents(
    audio: mx.array,
    *,
    vae_scale: int = 4,
    audio_window: int = 5,
) -> tuple[mx.array, mx.array]:
    """Reproduce EchoMimic's source-frame to latent-frame audio grouping.

    Args:
        audio: ``[batch, source_frames, window, hidden_states, channels]``.
        vae_scale: Number of source frames represented by a later latent frame.
        audio_window: Number of Wav2Vec neighbor positions per source frame.

    Returns:
        A pair containing audio for the first latent frame and for all later
        latent frames. With the released Flash configuration their shapes are
        ``[B, 1, 5, 12, 768]`` and ``[B, T-1, 8, 12, 768]``.
    """

    if audio.ndim != 5:
        raise ValueError(f"audio must have rank 5, got shape {audio.shape}")

    batch, source_frames, window, hidden_states, channels = audio.shape
    if source_frames < 1:
        raise ValueError("audio must contain at least one source frame")
    if window != audio_window:
        raise ValueError(f"expected audio window {audio_window}, got {window}")
    if vae_scale < 2:
        raise ValueError("vae_scale must be at least 2")
    if (source_frames - 1) % vae_scale:
        raise ValueError(
            "source frame count must be 1 + N * vae_scale; "
            f"got {source_frames} frames and vae_scale={vae_scale}"
        )

    first = audio[:, :1]
    following_latents = (source_frames - 1) // vae_scale
    following_width = audio_window + vae_scale - 1
    if following_latents == 0:
        following = mx.zeros(
            (batch, 0, following_width, hidden_states, channels), dtype=audio.dtype
        )
        return first, following

    later = audio[:, 1:].reshape(
        batch,
        following_latents,
        vae_scale,
        audio_window,
        hidden_states,
        channels,
    )
    middle = audio_window // 2

    leading = later[:, :, :1, : middle + 1].reshape(
        batch, following_latents, middle + 1, hidden_states, channels
    )
    center = later[:, :, 1:-1, middle : middle + 1].reshape(
        batch, following_latents, vae_scale - 2, hidden_states, channels
    )
    trailing = later[:, :, -1:, middle:].reshape(
        batch,
        following_latents,
        audio_window - middle,
        hidden_states,
        channels,
    )
    following = mx.concatenate([leading, center, trailing], axis=2)
    return first, following


class AudioProjection(nn.Module):
    """Project Wav2Vec hidden states into per-latent audio context tokens."""

    def __init__(
        self,
        *,
        seq_len: int = 5,
        seq_len_vf: int = 8,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 1536,
        context_tokens: int = 32,
        norm_output_audio: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.seq_len_vf = seq_len_vf
        self.blocks = blocks
        self.channels = channels
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.context_tokens = context_tokens

        self.proj1 = nn.Linear(seq_len * blocks * channels, intermediate_dim)
        self.proj1_vf = nn.Linear(seq_len_vf * blocks * channels, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def __call__(self, first: mx.array, following: mx.array) -> mx.array:
        if first.ndim != 5 or following.ndim != 5:
            raise ValueError("first and following audio tensors must both have rank 5")
        if first.shape[0] != following.shape[0]:
            raise ValueError("first and following audio batches must match")
        if first.shape[2:] != (self.seq_len, self.blocks, self.channels):
            raise ValueError(
                "unexpected first-frame audio shape: "
                f"expected (*, *, {self.seq_len}, {self.blocks}, {self.channels}), "
                f"got {first.shape}"
            )
        if following.shape[2:] != (self.seq_len_vf, self.blocks, self.channels):
            raise ValueError(
                "unexpected following-frame audio shape: "
                f"expected (*, *, {self.seq_len_vf}, {self.blocks}, {self.channels}), "
                f"got {following.shape}"
            )

        batch, first_frames = first.shape[:2]
        following_frames = following.shape[1]

        first = first.reshape(batch * first_frames, -1)
        first = mx.maximum(self.proj1(first), 0)
        first = first.reshape(batch, first_frames, self.intermediate_dim)

        if following_frames:
            following = following.reshape(batch * following_frames, -1)
            following = mx.maximum(self.proj1_vf(following), 0)
            following = following.reshape(batch, following_frames, self.intermediate_dim)
            combined = mx.concatenate([first, following], axis=1)
        else:
            combined = first

        latent_frames = combined.shape[1]
        combined = combined.reshape(batch * latent_frames, self.intermediate_dim)
        combined = mx.maximum(self.proj2(combined), 0)
        context = self.proj3(combined).reshape(
            batch * latent_frames, self.context_tokens, self.output_dim
        )
        context = self.norm(context)
        return context.reshape(batch, latent_frames, self.context_tokens, self.output_dim)


def make_audio_windows(features: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Gather clamped neighboring Wav2Vec features for every source frame.

    Args:
        features: Array shaped ``[frames, hidden_states, channels]``.
        window: Odd number of source-frame positions in every window.

    Returns:
        Array shaped ``[1, frames, window, hidden_states, channels]``.
    """

    features = np.asarray(features)
    if features.ndim != 3:
        raise ValueError(f"features must have rank 3, got shape {features.shape}")
    if features.shape[0] < 1:
        raise ValueError("features must contain at least one frame")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")

    radius = window // 2
    centers = np.arange(features.shape[0], dtype=np.int64)[:, None]
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)[None, :]
    indices = np.clip(centers + offsets, 0, features.shape[0] - 1)
    return features[indices][None]


def extract_wav2vec_windows(
    audio_path: str,
    model_path: str,
    *,
    source_frames: int,
    fps: int = 25,
    sample_rate: int = 16_000,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """Run the official Flash Wav2Vec preprocessing path on CPU.

    Feature extraction stays in PyTorch/CPU. The convolutional features are
    linearly interpolated to one position per source video frame before the 12
    encoder hidden states are collected, matching the released Flash code.
    """

    if source_frames < 1:
        raise ValueError("source_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    import librosa
    import pyloudnorm as pyln
    import torch
    import torch.nn.functional as functional
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_samples = int(source_frames / fps * sample_rate)
    waveform = waveform[:duration_samples]
    if waveform.size == 0:
        raise ValueError(f"audio file has no samples: {audio_path}")

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(waveform)
    if np.isfinite(loudness) and abs(loudness) <= 100:
        waveform = pyln.normalize.loudness(waveform, loudness, target_lufs)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
    model.eval()

    values = extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
    with torch.inference_mode():
        convolutional = model.feature_extractor(values).transpose(1, 2)
        convolutional = functional.interpolate(
            convolutional.transpose(1, 2),
            size=source_frames,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        hidden, _ = model.feature_projection(convolutional)
        encoded = model.encoder(
            hidden,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = encoded.hidden_states
    if hidden_states is None or len(hidden_states) != 13:
        count = 0 if hidden_states is None else len(hidden_states)
        raise ValueError(f"expected 13 Wav2Vec hidden-state tensors, got {count}")

    features = torch.stack(hidden_states[1:], dim=1)[0].permute(1, 0, 2).cpu().float().numpy()
    expected_shape = (source_frames, 12, 768)
    if features.shape != expected_shape:
        raise ValueError(f"expected Wav2Vec features {expected_shape}, got {features.shape}")
    return make_audio_windows(features, window=5)
