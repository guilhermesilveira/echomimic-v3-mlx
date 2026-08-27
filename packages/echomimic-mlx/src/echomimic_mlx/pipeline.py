"""End-to-end EchoMimic V3 Flash inference on MLX/Metal."""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .audio import extract_wav2vec_windows
from .echo_model import flash_audio_cfg
from .wan.clip import CLIPVisionEncoder, preprocess_clip_image
from .wan.sampler import FlowUniPCMultistepScheduler
from .wan.t5 import T5Encoder, create_umt5_xxl_encoder
from .wan.tokenizers import T5Tokenizer
from .wan.utils import _load_weights, save_video
from .wan.vae import WanVAE
from .weights import WeightAudit, load_flash_transformer


@dataclass(frozen=True)
class ModelPaths:
    """All local official-weight paths needed for one Flash run."""

    transformer: Path
    vae: Path
    t5: Path
    clip: Path
    tokenizer: Path
    wav2vec: Path

    def validate(self) -> None:
        files = {
            "transformer": self.transformer,
            "vae": self.vae,
            "t5": self.t5,
            "clip": self.clip,
            "tokenizer": self.tokenizer,
        }
        for name, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(f"{name} weights not found: {path}")
        if not self.wav2vec.is_dir():
            raise FileNotFoundError(f"wav2vec model directory not found: {self.wav2vec}")


@dataclass(frozen=True)
class SmokeConfig:
    width: int = 512
    height: int = 512
    frames: int = 17
    steps: int = 8
    fps: int = 25
    audio_guidance: float = 3.0
    shift: float = 5.0
    seed: int = 43
    dtype: str = "bfloat16"
    cache_limit: int = 0

    def validate(self) -> None:
        if self.width % 16 or self.height % 16:
            raise ValueError("width and height must be divisible by 16")
        if self.frames < 1 or (self.frames - 1) % 4:
            raise ValueError("frames must be 1 + N * 4 for the Wan VAE")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.fps != 25:
            raise ValueError("the released Flash audio alignment requires 25 fps")
        if self.dtype not in {"bfloat16", "float16"}:
            raise ValueError("dtype must be bfloat16 or float16")

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return mx.bfloat16 if self.dtype == "bfloat16" else mx.float16


@dataclass(frozen=True)
class StageMeasurement:
    name: str
    seconds: float
    mlx_peak_gib: float
    process_peak_gib_cumulative: float


@dataclass(frozen=True)
class GenerationReport:
    output: str
    config: dict[str, object]
    transformer_audit: dict[str, object]
    stages: tuple[StageMeasurement, ...]
    total_seconds: float
    limitations: tuple[str, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


class _Measurements:
    def __init__(self, log: Callable[[str], None]) -> None:
        self.items: list[StageMeasurement] = []
        self.log = log

    def run(self, name: str, operation: Callable, *args, **kwargs) -> object:
        self.log(f"[{name}] starting")
        mx.reset_peak_memory()
        started = time.perf_counter()
        result = operation(*args, **kwargs)
        seconds = time.perf_counter() - started
        peak = mx.get_peak_memory() / 1024**3
        process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
        measurement = StageMeasurement(name, seconds, peak, process_peak)
        self.items.append(measurement)
        self.log(f"[{name}] {seconds:.2f}s, MLX peak {peak:.2f} GiB")
        return result


def _audit_component(model: object, weights: dict[str, mx.array]) -> WeightAudit:
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
    return WeightAudit(len(expected), len(weights), missing, unexpected, shape_mismatches)


def _load_component(
    model: object,
    path: Path,
    sanitizer: Callable,
    *,
    allowed_missing: tuple[str, ...] = (),
) -> object:
    path = Path(path)
    weights = sanitizer(_load_weights(str(path)))
    audit = _audit_component(model, weights)
    disallowed_missing = tuple(sorted(set(audit.missing) - set(allowed_missing)))
    if disallowed_missing or audit.unexpected or audit.shape_mismatches:
        raise ValueError(
            f"{path.name} does not strictly match {type(model).__name__}: "
            f"missing={len(disallowed_missing)}, unexpected={len(audit.unexpected)}, "
            f"shape_mismatches={len(audit.shape_mismatches)}; "
            f"missing_keys={disallowed_missing[:10]}, unexpected_keys={audit.unexpected[:10]}, "
            f"shape_keys={audit.shape_mismatches[:10]}"
        )
    model.load_weights(list(weights.items()), strict=not allowed_missing)
    mx.eval(model.parameters())
    return model


def _encode_text(model: T5Encoder, tokenizer: T5Tokenizer, text: str, dtype: mx.Dtype) -> mx.array:
    tokens = tokenizer(text)
    ids = tokens["input_ids"]
    mask = tokens["attention_mask"]
    embeddings = model(ids, mask=mask)
    sequence_length = int(mask.sum().item())
    context = embeddings[0, :sequence_length]
    if sequence_length < 512:
        context = mx.concatenate(
            [context, mx.zeros((512 - sequence_length, context.shape[-1]), dtype=context.dtype)]
        )
    return context.astype(dtype)


def build_i2v_latent_mask(
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    dtype: mx.Dtype,
) -> mx.array:
    """Build the four mask channels used by Wan Fun first-frame conditioning."""

    if min(latent_frames, latent_height, latent_width) < 1:
        raise ValueError("latent dimensions must be positive")
    first = mx.ones((1, latent_height, latent_width, 4), dtype=dtype)
    rest = mx.zeros((latent_frames - 1, latent_height, latent_width, 4), dtype=dtype)
    return mx.concatenate([first, rest], axis=0)


def _prepare_image_conditioning(
    vae: WanVAE,
    image_path: Path,
    config: SmokeConfig,
) -> mx.array:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    scale = max(config.width / image_width, config.height / image_height)
    resized = image.resize(
        (round(image_width * scale), round(image_height * scale)), Image.Resampling.LANCZOS
    )
    left = (resized.width - config.width) // 2
    top = (resized.height - config.height) // 2
    image = resized.crop((left, top, left + config.width, top + config.height))

    image_array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    video = mx.concatenate(
        [
            mx.array(image_array)[None],
            mx.zeros((config.frames - 1, config.height, config.width, 3)),
        ],
        axis=0,
    )
    encoded = vae.encode(video)
    mask = build_i2v_latent_mask(
        encoded.shape[0], encoded.shape[1], encoded.shape[2], dtype=config.mlx_dtype
    )
    conditioning = mx.concatenate([mask, encoded.astype(config.mlx_dtype)], axis=-1)
    mx.eval(conditioning)
    return conditioning


def _mux_audio(silent_video: Path, audio_path: Path, output_path: Path, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _denoise_latents(
    model: object,
    latents: mx.array,
    context: mx.array,
    audio: mx.array,
    clip_features: mx.array,
    first_frame: mx.array,
    config: SmokeConfig,
    log: Callable[[str], None],
) -> mx.array:
    zero_audio = mx.zeros_like(audio)
    sampler = FlowUniPCMultistepScheduler()
    sampler.set_timesteps(config.steps, shift=config.shift)
    flow = mx.compile(model.__call__, inputs=[model.state])

    for index, timestep in enumerate(sampler.timesteps, start=1):
        t_value = timestep.reshape(1).astype(mx.float32)
        noise_with_audio, _ = flow(
            latents,
            t=t_value,
            context=context,
            audio=audio,
            clip_fea=clip_features,
            first_frame=first_frame,
        )
        noise_without_audio, _ = flow(
            latents,
            t=t_value,
            context=context,
            audio=zero_audio,
            clip_fea=clip_features,
            first_frame=first_frame,
        )
        prediction = flash_audio_cfg(
            noise_without_audio,
            noise_with_audio,
            config.audio_guidance,
        )
        latents = sampler.step(prediction, timestep, latents)
        mx.eval(latents)
        log(f"[denoise] step {index}/{config.steps}")
    return latents


def _load_flash_materialized(path: Path) -> tuple[object, WeightAudit]:
    model, audit = load_flash_transformer(path, strict=True)
    mx.eval(model.parameters())
    return model, audit


def _decode_materialized(vae: WanVAE, latents: mx.array) -> mx.array:
    video = vae.decode(latents)
    mx.eval(video)
    return video


def generate_smoke_video(
    *,
    image_path: Path,
    audio_path: Path,
    prompt: str,
    output_path: Path,
    paths: ModelPaths,
    config: SmokeConfig | None = None,
    log: Callable[[str], None] = print,
) -> GenerationReport:
    """Generate one short Flash video and return measured technical evidence."""

    config = config or SmokeConfig()
    paths.validate()
    config.validate()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(config.cache_limit)
    mx.random.seed(config.seed)
    measurements = _Measurements(log)
    total_started = time.perf_counter()

    def encode_audio() -> mx.array:
        encoded_audio = mx.array(
            extract_wav2vec_windows(
                str(audio_path),
                str(paths.wav2vec),
                source_frames=config.frames,
                fps=config.fps,
            )
        ).astype(config.mlx_dtype)
        mx.eval(encoded_audio)
        return encoded_audio

    audio = measurements.run("wav2vec_cpu", encode_audio)

    def encode_prompt() -> mx.array:
        tokenizer = T5Tokenizer(str(paths.tokenizer))
        t5 = _load_component(create_umt5_xxl_encoder(), paths.t5, T5Encoder.sanitize)
        context = _encode_text(t5, tokenizer, prompt, config.mlx_dtype)
        mx.eval(context)
        del t5
        mx.clear_cache()
        return context

    context = measurements.run("umt5", encode_prompt)

    def encode_clip() -> mx.array:
        clip = _load_component(CLIPVisionEncoder(), paths.clip, CLIPVisionEncoder.sanitize)
        features = clip(preprocess_clip_image(str(image_path))).astype(config.mlx_dtype)
        mx.eval(features)
        del clip
        mx.clear_cache()
        return features

    clip_features = measurements.run("clip", encode_clip)

    vae = measurements.run(
        "vae_load",
        _load_component,
        WanVAE(),
        paths.vae,
        WanVAE.sanitize,
        allowed_missing=("mean", "std"),
    )
    first_frame = measurements.run(
        "vae_encode",
        _prepare_image_conditioning,
        vae,
        image_path,
        config,
    )
    mx.eval(first_frame)

    model, audit = measurements.run(
        "transformer_load",
        _load_flash_materialized,
        paths.transformer,
    )
    latent_shape = (
        (config.frames - 1) // 4 + 1,
        config.height // 8,
        config.width // 8,
        16,
    )
    latents = mx.random.normal(latent_shape).astype(config.mlx_dtype)
    latents = measurements.run(
        "denoise",
        _denoise_latents,
        model,
        latents,
        context,
        audio,
        clip_features,
        first_frame,
        config,
        log,
    )
    del model
    mx.clear_cache()

    video = measurements.run("vae_decode", _decode_materialized, vae, latents)
    del vae
    mx.clear_cache()

    silent_path = output_path.with_name(f"{output_path.stem}.silent.mp4")

    def save_output() -> None:
        if not save_video(video, str(silent_path), fps=config.fps):
            raise RuntimeError("FFmpeg failed while encoding the silent video")
        _mux_audio(silent_path, audio_path, output_path, config.frames / config.fps)
        silent_path.unlink(missing_ok=True)

    measurements.run("encode_mp4", save_output)
    total_seconds = time.perf_counter() - total_started
    report = GenerationReport(
        output=str(output_path),
        config=asdict(config),
        transformer_audit=asdict(audit),
        stages=tuple(measurements.items),
        total_seconds=total_seconds,
        limitations=(
            "This MP4 proves pipeline integration only; it is not evidence of usable quality.",
            "The released Flash forward retains but does not use q_audio.",
            "The released 2512 Flash path applies audio CFG only; text CFG is not applied.",
            "Attention and VAE latency remain separate acceptance gates.",
            f"A {config.frames / config.fps:.2f}-second sample cannot establish "
            "long-window temporal continuity.",
        ),
    )
    report.write(output_path.with_suffix(".json"))
    return report
