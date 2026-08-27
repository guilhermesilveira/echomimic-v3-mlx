from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from echomimic_mlx.audio import AudioProjection, make_audio_windows, split_audio_for_latents


def test_split_audio_for_latents_matches_released_torch_indexing() -> None:
    values = np.arange(1 * 17 * 5 * 2 * 3, dtype=np.float32).reshape(1, 17, 5, 2, 3)
    audio_torch = torch.from_numpy(values)

    later = audio_torch[:, 1:].reshape(1, 4, 4, 5, 2, 3)
    expected_first = audio_torch[:, :1]
    expected_later = torch.cat(
        [
            later[:, :, :1, :3].reshape(1, 4, 3, 2, 3),
            later[:, :, 1:-1, 2:3].reshape(1, 4, 2, 2, 3),
            later[:, :, -1:, 2:].reshape(1, 4, 3, 2, 3),
        ],
        dim=2,
    )

    first, following = split_audio_for_latents(mx.array(values))
    np.testing.assert_array_equal(np.array(first), expected_first.numpy())
    np.testing.assert_array_equal(np.array(following), expected_later.numpy())
    assert following.shape == (1, 4, 8, 2, 3)


def test_split_audio_rejects_frame_count_not_aligned_to_vae_stride() -> None:
    with pytest.raises(ValueError, match=r"1 \+ N \* vae_scale"):
        split_audio_for_latents(mx.zeros((1, 16, 5, 2, 3)))


def test_make_audio_windows_clamps_both_edges() -> None:
    features = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
    windows = make_audio_windows(features)
    assert windows.shape == (1, 4, 5, 1, 1)
    np.testing.assert_array_equal(
        windows[0, :, :, 0, 0],
        [
            [0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3],
            [0, 1, 2, 3, 3],
            [1, 2, 3, 3, 3],
        ],
    )


def test_audio_projection_matches_pytorch_cpu() -> None:
    rng = np.random.default_rng(4)
    model = AudioProjection(
        seq_len=5,
        seq_len_vf=8,
        blocks=2,
        channels=3,
        intermediate_dim=4,
        output_dim=6,
        context_tokens=2,
    )
    shapes = {
        "proj1.weight": (4, 30),
        "proj1.bias": (4,),
        "proj1_vf.weight": (4, 48),
        "proj1_vf.bias": (4,),
        "proj2.weight": (4, 4),
        "proj2.bias": (4,),
        "proj3.weight": (12, 4),
        "proj3.bias": (12,),
        "norm.weight": (6,),
        "norm.bias": (6,),
    }
    parameters = {
        key: (rng.standard_normal(shape) * 0.1).astype(np.float32) for key, shape in shapes.items()
    }
    model.load_weights([(key, mx.array(value)) for key, value in parameters.items()], strict=True)

    first_np = rng.standard_normal((1, 1, 5, 2, 3)).astype(np.float32)
    following_np = rng.standard_normal((1, 2, 8, 2, 3)).astype(np.float32)
    actual = model(mx.array(first_np), mx.array(following_np))
    mx.eval(actual)

    p = {key: torch.from_numpy(value) for key, value in parameters.items()}
    first = torch.from_numpy(first_np).reshape(1, -1)
    following = torch.from_numpy(following_np).reshape(2, -1)
    first = functional.relu(functional.linear(first, p["proj1.weight"], p["proj1.bias"]))
    following = functional.relu(
        functional.linear(following, p["proj1_vf.weight"], p["proj1_vf.bias"])
    )
    combined = torch.cat([first.reshape(1, 1, 4), following.reshape(1, 2, 4)], dim=1)
    combined = functional.relu(
        functional.linear(combined.reshape(3, 4), p["proj2.weight"], p["proj2.bias"])
    )
    expected = functional.linear(combined, p["proj3.weight"], p["proj3.bias"])
    expected = functional.layer_norm(
        expected.reshape(3, 2, 6),
        (6,),
        p["norm.weight"],
        p["norm.bias"],
        eps=1e-5,
    ).reshape(1, 3, 2, 6)

    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=2e-5, atol=2e-5)
