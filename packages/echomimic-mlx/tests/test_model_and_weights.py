from __future__ import annotations

import mlx.core as mx
import numpy as np
from echomimic_mlx.echo_model import EchoMimicModel, audio_cfg, flash_audio_cfg
from echomimic_mlx.pipeline import SmokeConfig, build_i2v_latent_mask
from echomimic_mlx.weights import audit_weights
from mlx.utils import tree_flatten


def _tiny_model() -> EchoMimicModel:
    return EchoMimicModel(
        patch_size=(1, 2, 2),
        in_dim=36,
        dim=24,
        ffn_dim=32,
        freq_dim=8,
        text_dim=16,
        out_dim=16,
        num_heads=2,
        num_layers=1,
        audio_blocks=2,
        audio_channels=4,
        audio_intermediate_dim=8,
        audio_context_tokens=3,
    )


def test_tiny_complete_transformer_forward_has_expected_shape() -> None:
    model = _tiny_model()
    output, residual = model(
        mx.zeros((2, 4, 4, 16)),
        mx.array([500.0]),
        mx.zeros((512, 16)),
        audio=mx.zeros((1, 5, 5, 2, 4)),
        clip_fea=mx.zeros((1, 257, 1280)),
        first_frame=mx.zeros((2, 4, 4, 20)),
    )
    mx.eval(output, residual)
    assert output.shape == (2, 4, 4, 16)
    assert residual.shape == (1, 8, 24)
    assert bool(mx.isfinite(output).all().item())
    assert bool(mx.isfinite(residual).all().item())


def test_audio_cfg_matches_three_branch_formula() -> None:
    unconditional = mx.array([1.0, 3.0])
    text = mx.array([2.0, 5.0])
    audio = mx.array([4.0, 9.0])
    actual = audio_cfg(unconditional, text, audio, 6.0, 3.0)
    np.testing.assert_allclose(np.array(actual), [13.0, 27.0])


def test_flash_audio_cfg_matches_released_two_branch_formula() -> None:
    without_audio = mx.array([1.0, 3.0])
    with_audio = mx.array([4.0, 9.0])
    actual = flash_audio_cfg(without_audio, with_audio, 3.0)
    np.testing.assert_allclose(np.array(actual), [10.0, 21.0])


def test_i2v_latent_mask_keeps_only_first_latent_known() -> None:
    mask = build_i2v_latent_mask(5, 2, 3, dtype=mx.float16)
    assert mask.shape == (5, 2, 3, 4)
    assert bool((mask[0] == 1).all().item())
    assert bool((mask[1:] == 0).all().item())


def test_smoke_config_enforces_flash_geometry_and_fps() -> None:
    SmokeConfig().validate()
    for config in (
        SmokeConfig(width=510),
        SmokeConfig(frames=16),
        SmokeConfig(fps=16),
    ):
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid smoke config accepted: {config}")


def test_weight_audit_reports_missing_unexpected_and_shape_mismatch() -> None:
    model = _tiny_model()
    weights = dict(tree_flatten(model.parameters()))
    removed_key = next(iter(weights))
    weights.pop(removed_key)
    shape_key = next(iter(weights))
    weights[shape_key] = mx.zeros((1,))
    weights["not_in_model"] = mx.zeros((1,))

    audit = audit_weights(model, weights)
    assert removed_key in audit.missing
    assert shape_key in audit.shape_mismatches
    assert audit.unexpected == ("not_in_model",)
    assert not audit.valid


def test_sanitize_transposes_conv_merges_attention_and_keeps_audio_keys() -> None:
    raw = {
        "model.patch_embedding.weight": mx.zeros((24, 36, 1, 2, 2)),
        "model.blocks.0.self_attn.q.weight": mx.ones((24, 24)),
        "model.blocks.0.self_attn.k.weight": mx.ones((24, 24)) * 2,
        "model.blocks.0.self_attn.v.weight": mx.ones((24, 24)) * 3,
        "model.blocks.0.cross_attn.k.weight": mx.ones((24, 24)) * 4,
        "model.blocks.0.cross_attn.v.weight": mx.ones((24, 24)) * 5,
        "model.blocks.0.cross_attn.q_audio.weight": mx.ones((24, 24)) * 6,
        "model.blocks.0.modulation": mx.zeros((1, 6, 24)),
        "model.head.modulation": mx.zeros((1, 2, 24)),
    }
    converted = EchoMimicModel.sanitize(raw)
    assert converted["patch_embedding.weight"].shape == (24, 1, 2, 2, 36)
    assert converted["blocks.0.self_attn.qkv.weight"].shape == (72, 24)
    assert converted["blocks.0.cross_attn.kv.weight"].shape == (48, 24)
    assert "blocks.0.cross_attn.q_audio.weight" in converted
    np.testing.assert_array_equal(
        np.array(converted["blocks.0.modulation"])[0, :, 0], [0, 1, 0, 0, 1, 0]
    )
    np.testing.assert_array_equal(np.array(converted["head.modulation"])[0, :, 0], [0, 1])
