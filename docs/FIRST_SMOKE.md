# First local Flash smoke

## Result boundary

The first end-to-end EchoMimic V3 Flash video was generated locally with
MLX/Metal on a Mac mini M4 Pro (20-core GPU, 64 GB unified memory). It proves
that the pinned official checkpoints can be converted strictly and executed
through Wav2Vec, UMT5, CLIP, Wan I2V conditioning, Flash audio attention,
UniPC denoising, Wan VAE decode, and MP4 audio muxing.

This is a technical integration result, not a usable-quality acceptance. The
clip is only 0.68 seconds long and cannot establish long-window continuity,
identity stability, or PT-BR lip-sync quality.

## Inputs and provenance

No personal portrait or audio was used. The image, audio, and prompt came from
the official `antgroup/echomimic_v3` demo at revision
`7e89489ca51c0d008fc1963ec6c03fc5bd0b9397`:

- `datasets/echomimicv3_demos/imgs/01.jpg`
- `datasets/echomimicv3_demos/audios/01.WAV`
- `datasets/echomimicv3_demos/prompts/01.txt`

Those assets, all model weights, and generated outputs were excluded from Git.
The current checkout does not contain generated outputs. Model and source
revisions are recorded in `upstreams.lock.json`.

## Configuration

| Setting | Value |
| --- | ---: |
| output | 512x512 |
| video frames | 17 |
| frame rate | 25 fps |
| duration | 0.68 s |
| denoising steps | 8 |
| scheduler shift | 5.0 |
| audio guidance | 3.0 |
| seed | 43 |
| MLX dtype | bfloat16 |

Eight steps were retained as requested. The released Flash guidance describes
five steps for a talking head and 15-25 for larger body motion, so the pinned
checkpoint and scheduler did not require reducing or increasing this smoke.

## Reproduction command shape

Install the locked environment with Python 3.11, then provide the official
pinned files already downloaded under `models/`:

```bash
uv sync --all-extras --python 3.11

uv run echomimic-mlx-smoke \
  --image /path/to/echomimic_v3/datasets/echomimicv3_demos/imgs/01.jpg \
  --audio /path/to/echomimic_v3/datasets/echomimicv3_demos/audios/01.WAV \
  --prompt "A person is holding an object in a relaxed pose. As the video progresses, the character speaks while arm and body movements are minimal and consistent with a natural speaking posture. Hand movements remain minimal. Don't blink too often. Preserve background integrity matching the reference image's spatial configuration, lighting conditions, and color temperature." \
  --transformer models/huggingface/hub/models--BadToBest--EchoMimicV3/snapshots/311e176905a8c4c24b240b530488fe636ce4d249/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors \
  --vae models/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP/snapshots/fc913c34361f4ec879e2f9c78b4f11ae50a937d1/Wan2.1_VAE.pth \
  --t5 models/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP/snapshots/fc913c34361f4ec879e2f9c78b4f11ae50a937d1/models_t5_umt5-xxl-enc-bf16.pth \
  --clip models/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP/snapshots/fc913c34361f4ec879e2f9c78b4f11ae50a937d1/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --tokenizer models/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP/snapshots/fc913c34361f4ec879e2f9c78b4f11ae50a937d1/google/umt5-xxl/tokenizer.json \
  --wav2vec models/chinese-wav2vec2-base \
  --output outputs/first-smoke-512x512-17f-8steps.mp4
```

The CLI writes a JSON report beside the MP4. It never downloads a model or
copies an input asset into the repository.

## First-run measurements

Wall-clock figures include materialization inside each stage. `MLX peak` is
reset per stage. The process peak is cumulative because macOS exposes a
high-water mark rather than an independently resettable stage value.

| Stage | Time | MLX peak |
| --- | ---: | ---: |
| Wav2Vec on CPU | 4.48 s | 0.00 GiB |
| UMT5 | 4.50 s | 21.83 GiB |
| CLIP | 1.24 s | 4.45 GiB |
| VAE load | 0.11 s | 0.48 GiB |
| VAE encode | 5.38 s | 15.05 GiB |
| Transformer load | 0.75 s | 4.00 GiB |
| 8-step denoise | 47.10 s | 4.83 GiB |
| VAE decode | 9.38 s | 17.57 GiB |
| MP4 encode and mux | 0.29 s | 0.17 GiB |
| **Complete run** | **73.27 s** | **21.83 GiB maximum** |

The cumulative process peak was 37.02 GiB. The full Flash Transformer audit
reported 1,023 expected and 1,023 provided tensors, with no missing,
unexpected, or shape-mismatched keys.

A second end-to-end validation of the final code took 115.76 seconds under a
different machine-load state. Its per-stage MLX peaks matched the first run,
including 21.83 GiB for UMT5, 15.05 GiB for VAE encode, 4.83 GiB for denoising,
and 17.57 GiB for VAE decode. Its cumulative process high-water mark was
16.55 GiB. Treat 73.27-115.76 seconds as the observed end-to-end range from
these two runs, not as a throughput benchmark.

## Media verification

`ffprobe` confirmed an H.264 stream at 512x512, 25 fps, exactly 17 frames, and
an AAC stereo stream at 44.1 kHz. Five sampled frames retained the portrait,
held object, and background while showing progressive eye and mouth motion.
Per-frame luminance differences were non-zero throughout the sequence, ruling
out a frozen video in this smoke. The sampled-frame contact sheets from the
first and final runs were byte-identical, confirming deterministic visual
output for the fixed seed and configuration.

## Remaining acceptance gates

- Attention is still the dominant denoising cost: 47.10 seconds for eight
  steps on only five latent frames.
- VAE encode/decode reached 15.05/17.57 GiB peak Metal allocation and needs
  separate optimization before longer clips.
- The released Flash forward retains checkpoint `q_audio` parameters but does
  not use their projected values.
- The released `_2512` path performs audio CFG only; it does not run text CFG.
- Longer authorized input is required to test temporal continuity, identity
  drift, natural blinking, and PT-BR phoneme alignment.
