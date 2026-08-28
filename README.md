# EchoMimic V3 Flash Inference for MLX

Native Apple Silicon port of the released EchoMimic V3 Flash inference path,
based on Apple's Wan 2.1 implementation in `mlx-examples`. This is not a port
of the complete EchoMimic V3 project.

The implementation was validated on a Mac mini M4 Pro with a 20-core GPU and
64 GB of unified memory. A generated file is not automatically a usable result:
attention, VAE decode time, temporal continuity, identity, and lip sync are
separate acceptance gates.

## Current status

- Upstream source and model revisions are pinned in `upstreams.lock.json`.
- The Wan 2.1 MLX implementation is vendored under the original MIT headers.
- The EchoMimic audio projection, per-frame audio cross-attention, strict weight
  conversion, scheduler, audio CFG, masks, VAE path, and FFmpeg output are
  implemented on top of the 1.3B Wan I2V architecture.
- Two complete local smokes passed at 512x512, 17 frames, and eight steps in
  73.27 and 115.76 seconds. The peak measured Metal allocation was stable at
  21.83 GiB; the generated frames were identical across both runs.
- Thirteen synthetic shape and component-level PyTorch/CPU equivalence tests
  pass. Full real-weight numerical equivalence is not claimed.
- Model weights, private portraits, private audio, and generated videos are
  intentionally excluded from Git.

## Scope boundary

Implemented:

- single-image EchoMimic V3 Flash inference;
- 25 fps Wav2Vec audio alignment on CPU;
- strict Flash Transformer conversion and MLX/Metal execution;
- fixed first-frame I2V conditioning, audio CFG, UniPC sampling, Wan VAE decode,
  and MP4 muxing.

Not implemented:

- EchoMimic V3 Preview, training, TA2.0, Gradio, or ComfyUI;
- TeaCache, RiFLEX, overlap/chunk-based long-video generation, or callbacks;
- user-provided inpaint/IP masks, end-image conditioning, or the Preview
  pipeline's text/negative CFG path.

The pinned Flash `_2512` denoising path evaluates real-audio and zero-audio
branches with the same positive text context. This port reproduces that effective
two-branch behavior rather than advertising text CFG that the released forward
does not apply.

## Development

```bash
uv sync --all-extras --python 3.11
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build --all-packages
```

The workspace root is intentionally a non-package project; the distributable
member is `packages/echomimic-mlx`. The CLI entrypoint is
`uv run echomimic-mlx-smoke --help`. It requires the
official pinned model files listed in `upstreams.lock.json`; weights are not
downloaded implicitly.

## License

New MLX port code is distributed under Apache-2.0. The vendored Apple Wan 2.1
implementation remains under its original MIT terms. See `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `third_party/licenses/`.

See `docs/ARCHITECTURE.md` for the compatibility findings and
`docs/FIRST_SMOKE.md` for the exact first-run evidence and command shape.
