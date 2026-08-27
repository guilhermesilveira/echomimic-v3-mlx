# EchoMimic V3 Flash for MLX

Native Apple Silicon port of EchoMimic V3 Flash, based on Apple's Wan 2.1
implementation in `mlx-examples`.

The target machine is a Mac mini M4 Pro with a 20-core GPU and 64 GB of unified
memory. The first milestone is a short 512x512, 17-frame, audio-driven smoke
video. A generated file is not automatically a usable result: attention, VAE
decode time, temporal continuity, identity, and PT-BR lip sync are separate
acceptance gates.

## Current status

- Upstream source and model revisions are pinned in `upstreams.lock.json`.
- The Wan 2.1 MLX implementation is vendored under the original MIT headers.
- The EchoMimic audio projection, per-frame audio cross-attention, strict weight
  conversion, scheduler, audio CFG, masks, VAE path, and FFmpeg output are
  implemented on top of the 1.3B Wan I2V architecture.
- Two complete local smokes passed at 512x512, 17 frames, and eight steps in
  73.27 and 115.76 seconds. The peak measured Metal allocation was stable at
  21.83 GiB; the generated frames were identical across both runs.
- Thirteen synthetic and PyTorch/CPU equivalence tests pass.
- Model weights, private portraits, private audio, and generated videos are
  intentionally excluded from Git.

## Development

```bash
uv sync --all-extras --python 3.11
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The CLI entrypoint is `uv run echomimic-mlx-smoke --help`. It requires the
official pinned model files listed in `upstreams.lock.json`; weights are not
downloaded implicitly.

See `docs/ARCHITECTURE.md` for the compatibility findings and
`docs/FIRST_SMOKE.md` for the exact first-run evidence and command shape.
