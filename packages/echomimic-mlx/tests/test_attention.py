from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional
from echomimic_mlx.wan.layers import WanI2VAudioCrossAttention


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def test_audio_cross_attention_matches_pytorch_cpu_and_released_q_audio_behavior() -> None:
    rng = np.random.default_rng(7)
    dim, heads, latent_frames = 12, 3, 2
    layer = WanI2VAudioCrossAttention(dim, heads)
    shapes = {
        "q.weight": (dim, dim),
        "q.bias": (dim,),
        "kv.weight": (2 * dim, dim),
        "kv.bias": (2 * dim,),
        "o.weight": (dim, dim),
        "o.bias": (dim,),
        "norm_q.weight": (dim,),
        "norm_k.weight": (dim,),
        "k_img.weight": (dim, dim),
        "k_img.bias": (dim,),
        "v_img.weight": (dim, dim),
        "v_img.bias": (dim,),
        "norm_k_img.weight": (dim,),
        "q_audio.weight": (dim, dim),
        "q_audio.bias": (dim,),
        "k_audio.weight": (dim, dim),
        "k_audio.bias": (dim,),
        "v_audio.weight": (dim, dim),
        "v_audio.bias": (dim,),
        "norm_k_audio.weight": (dim,),
    }
    params = {
        key: (rng.standard_normal(shape) * 0.08).astype(np.float32) for key, shape in shapes.items()
    }
    layer.load_weights([(key, mx.array(value)) for key, value in params.items()], strict=True)

    x_np = rng.standard_normal((1, 6, dim)).astype(np.float32)
    image_np = rng.standard_normal((1, 2, dim)).astype(np.float32)
    text_np = rng.standard_normal((1, 512, dim)).astype(np.float32)
    audio_np = rng.standard_normal((1, latent_frames, 4, dim)).astype(np.float32)
    context_np = np.concatenate([image_np, text_np], axis=1)
    actual = layer(mx.array(x_np), (mx.array(context_np), mx.array(audio_np), latent_frames))
    mx.eval(actual)

    p = {key: torch.from_numpy(value) for key, value in params.items()}
    x = torch.from_numpy(x_np)
    context = torch.from_numpy(context_np)
    audio = torch.from_numpy(audio_np)
    head_dim = dim // heads

    q = _rms_norm(functional.linear(x, p["q.weight"], p["q.bias"]), p["norm_q.weight"])
    q = q.reshape(1, 6, heads, head_dim).transpose(1, 2)
    kv = functional.linear(context[:, 2:], p["kv.weight"], p["kv.bias"])
    k, v = kv.chunk(2, dim=-1)
    k = _rms_norm(k, p["norm_k.weight"])
    k = k.reshape(1, 512, heads, head_dim).transpose(1, 2)
    v = v.reshape(1, 512, heads, head_dim).transpose(1, 2)
    text_out = functional.scaled_dot_product_attention(q, k, v)

    image = context[:, :2]
    image_k = (
        _rms_norm(
            functional.linear(image, p["k_img.weight"], p["k_img.bias"]),
            p["norm_k_img.weight"],
        )
        .reshape(1, 2, heads, head_dim)
        .transpose(1, 2)
    )
    image_v = functional.linear(image, p["v_img.weight"], p["v_img.bias"])
    image_v = image_v.reshape(1, 2, heads, head_dim).transpose(1, 2)
    image_out = functional.scaled_dot_product_attention(q, image_k, image_v)

    audio_flat = audio.reshape(latent_frames, 4, dim)
    audio_k = (
        _rms_norm(
            functional.linear(audio_flat, p["k_audio.weight"], p["k_audio.bias"]),
            p["norm_k_audio.weight"],
        )
        .reshape(latent_frames, 4, heads, head_dim)
        .transpose(1, 2)
    )
    audio_v = functional.linear(audio_flat, p["v_audio.weight"], p["v_audio.bias"])
    audio_v = audio_v.reshape(latent_frames, 4, heads, head_dim).transpose(1, 2)
    audio_q = q.transpose(1, 2).reshape(latent_frames, 3, heads, head_dim).transpose(1, 2)
    audio_out = functional.scaled_dot_product_attention(audio_q, audio_k, audio_v)
    audio_out = audio_out.transpose(1, 2).reshape(1, 6, heads, head_dim).transpose(1, 2)

    combined = (text_out + image_out + audio_out).transpose(1, 2).reshape(1, 6, dim)
    expected = functional.linear(combined, p["o.weight"], p["o.bias"])
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=4e-5, atol=4e-5)

    zeroed = dict(params)
    zeroed["q_audio.weight"] = np.zeros_like(params["q_audio.weight"])
    zeroed["q_audio.bias"] = np.zeros_like(params["q_audio.bias"])
    layer.load_weights([(key, mx.array(value)) for key, value in zeroed.items()], strict=True)
    without_q_audio = layer(
        mx.array(x_np), (mx.array(context_np), mx.array(audio_np), latent_frames)
    )
    mx.eval(without_q_audio)
    np.testing.assert_array_equal(np.array(actual), np.array(without_q_audio))
