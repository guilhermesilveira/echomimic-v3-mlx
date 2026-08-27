"""Command-line entrypoint for the first local technical smoke."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .pipeline import ModelPaths, SmokeConfig, generate_smoke_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--wav2vec", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio-guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--cache-limit", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    paths = ModelPaths(
        transformer=args.transformer,
        vae=args.vae,
        t5=args.t5,
        clip=args.clip,
        tokenizer=args.tokenizer,
        wav2vec=args.wav2vec,
    )
    config = SmokeConfig(
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        fps=args.fps,
        audio_guidance=args.audio_guidance,
        shift=args.shift,
        seed=args.seed,
        dtype=args.dtype,
        cache_limit=args.cache_limit,
    )
    report = generate_smoke_video(
        image_path=args.image,
        audio_path=args.audio,
        prompt=args.prompt,
        output_path=args.output,
        paths=paths,
        config=config,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
