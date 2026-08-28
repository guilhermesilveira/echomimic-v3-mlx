# Architecture and compatibility record

## Fixed baseline

The port is based on EchoMimic V3 Flash rather than TA2.0. The implementation
starts from Apple's MLX Wan 2.1 code and adds only the EchoMimic-specific path.

Pinned revisions live in `upstreams.lock.json`; model artifacts are downloaded
locally and never committed.

## Confirmed checkpoint topology

The official Flash and Wan Fun configs agree on the base Transformer:

| Field | Value |
| --- | ---: |
| hidden dimension | 1536 |
| FFN dimension | 8960 |
| heads | 12 |
| layers | 30 |
| input channels | 36 |
| output channels | 16 |
| patch | 1x2x2 |
| mode | I2V |

Header-only inspection found 983 tensors in the Wan Fun base and 1,203 tensors
in the Flash checkpoint. The 220 additional tensors are exactly:

- 10 tensors in the audio projection module;
- 7 tensors per Transformer block across 30 blocks: `q_audio`, `k_audio`,
  `v_audio`, and `norm_k_audio` parameters.

This makes the initial port a bounded extension of the existing 1.3B I2V
topology rather than a different Transformer family.

The MLX sanitizer merges the checkpoint's split query/key/value tensors into
the fused projections used by Apple's implementation. After conversion, the
full Transformer has 1,023 expected and 1,023 provided tensors, with no
missing, unexpected, or shape-mismatched entries.

## Audio path

Wav2Vec produces features for each source video frame with shape
`[frames, window=5, hidden_states=12, channels=768]`. EchoMimic groups four
source frames per latent frame. The first latent uses 5 window positions; each
later latent uses 8 selected window positions. The projection emits 32 audio
tokens of width 1536 per latent frame.

Every Transformer block performs text attention, image attention, and a third
audio attention. Audio attention is local to the corresponding latent frame;
it does not attend to the complete video audio sequence.

The upstream Flash code declares `q_audio`, but overwrites its projected value
before attention and therefore does not use that projection in the released
forward pass. The MLX port retains the tensor for checkpoint completeness and
reproduces the released behavior until CPU/PyTorch equivalence proves a reason
to change it.

The released `_2512` Flash pipeline runs two Transformer branches with the same
positive text context: zero audio and real audio. It combines them with audio
CFG only. The port reproduces that behavior; it does not add text CFG that the
released Flash path does not execute.

## Precision, scheduler, and conditioning

The first smoke uses bfloat16 weights and activations, the upstream UniPC flow
scheduler, shift 5.0, audio guidance 3.0, and seed 43. Eight denoising steps are
within the released Flash range: upstream describes five steps for a talking
head and more for larger body motion. No checkpoint requirement forced a
change from the requested eight-step smoke.

The I2V conditioning tensor contains four mask channels followed by the 16 VAE
latent channels. Only the first latent frame is marked by the mask; the
remaining frames are generated from noise while retaining first-frame image,
CLIP, text, and aligned audio conditioning.

## First-video gates

1. Strict checkpoint-key conversion with no missing or unexpected tensors.
2. Synthetic shape tests for audio grouping, projection, cross-attention, and
   the complete small Transformer.
3. Numerical comparison of synthetic audio projection and cross-attention
   components against PyTorch/CPU.
4. Load the full Flash Transformer and record active/peak Metal memory.
5. Generate 17 frames at 512x512 with eight steps.
6. Decode through the Wan VAE and mux the source audio with FFmpeg.

All six gates passed for the first technical smoke at the stated scope. The
tests do not establish full real-weight numerical equivalence for every Wan,
VAE, text, or image-encoder component. Detailed measurements and the exact
acceptance boundary are recorded in `FIRST_SMOKE.md`.
