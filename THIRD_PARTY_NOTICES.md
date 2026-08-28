# Third-party notices

## mlx-examples Wan 2.1

Files under `packages/echomimic-mlx/src/echomimic_mlx/wan` are adapted from
Apple's `ml-explore/mlx-examples` Wan 2.1 implementation, pinned in
`upstreams.lock.json`. Those files retain their original copyright headers and
are distributed under the MIT License. A copy is stored at
`third_party/licenses/mlx-examples-MIT.txt`.

## EchoMimic V3 and Wan Fun

The audio architecture and checkpoint mapping are derived from the official
EchoMimic V3 and VideoX-Fun/Wan Fun projects. Their source revisions are pinned
in `upstreams.lock.json` and are licensed under Apache-2.0. The MLX-specific
files identify that they were changed. The repository-level `LICENSE` contains
the Apache-2.0 terms.

Official model checkpoints are referenced by repository and immutable revision
only. They are not redistributed in source archives or wheels. Users remain
responsible for reviewing the corresponding model card before downloading or
using a checkpoint.

## Chinese Wav2Vec2 base

The Flash inference path uses `TencentGameMate/chinese-wav2vec2-base` at the
revision pinned in `upstreams.lock.json`. Its ModelScope model card declares
the model MIT licensed. The audio adapter in this repository calls the
Transformers implementation and does not vendor the model weights.

## Project license

Code authored specifically for this MLX port is distributed under Apache-2.0.
Vendored files remain subject to their upstream terms as described above.
