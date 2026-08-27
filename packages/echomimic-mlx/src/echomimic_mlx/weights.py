"""Strict EchoMimic checkpoint loading and audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .echo_model import EchoMimicModel


@dataclass(frozen=True)
class WeightAudit:
    expected: int
    provided: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatches)


def audit_weights(model: EchoMimicModel, weights: dict[str, mx.array]) -> WeightAudit:
    expected = dict(tree_flatten(model.parameters()))
    missing = tuple(sorted(set(expected) - set(weights)))
    unexpected = tuple(sorted(set(weights) - set(expected)))
    shape_mismatches = tuple(
        sorted(
            key
            for key in set(expected) & set(weights)
            if tuple(expected[key].shape) != tuple(weights[key].shape)
        )
    )
    return WeightAudit(
        expected=len(expected),
        provided=len(weights),
        missing=missing,
        unexpected=unexpected,
        shape_mismatches=shape_mismatches,
    )


def load_flash_transformer(
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> tuple[EchoMimicModel, WeightAudit]:
    checkpoint = Path(checkpoint)
    if checkpoint.suffix != ".safetensors":
        raise ValueError(f"expected a .safetensors checkpoint, got {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    raw = mx.load(str(checkpoint))
    weights = EchoMimicModel.sanitize(raw)
    model = EchoMimicModel()
    audit = audit_weights(model, weights)
    if strict and not audit.valid:
        raise ValueError(
            "checkpoint does not exactly match EchoMimicModel: "
            f"missing={len(audit.missing)}, unexpected={len(audit.unexpected)}, "
            f"shape_mismatches={len(audit.shape_mismatches)}"
        )
    model.load_weights(list(weights.items()), strict=strict)
    return model, audit
