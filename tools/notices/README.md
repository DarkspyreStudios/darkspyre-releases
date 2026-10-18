# Driver archive notices

`tools/ggml_drivers.py extract-upstream` copies these licence notices into each driver archive under `licenses/`.
`docs/ggml-drivers.md` lists which archive carries which notice. Each file is the upstream text, byte for byte.

| File | Source |
| --- | --- |
| `TensorSharp-LICENSE.txt` | `LICENSE` in zhongkaifu/TensorSharp at tag `v2026.10.11`, commit `f20fade4aa1feb67b5e2d43dad584c2af6865b1e` |
| `ggml-LICENSE.txt` | `licenses/GGML-LICENSE.txt` in upstream's `tensorsharp-cli-2026.10.11-linux-arm64-cuda13-GB10.tar.gz` |
| `NVIDIA-cudart.txt` | `licenses/NVIDIA-cudart.txt` in the same GB10 archive |
| `NVIDIA-cuBLAS.txt` | `licenses/NVIDIA-cuBLAS.txt` in the same GB10 archive |
| `mlx-c-LICENSE.txt` | `LICENSE` in ml-explore/mlx-c at commit `ebc88f10caa1b625e6b581437a8dea6df8a70085`, the commit that `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` pins at TensorSharp `v2026.10.11` |
| `mlx-LICENSE.txt` | `LICENSE` in ml-explore/mlx at tag `v0.32.2`, commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, the version that mlx-c `ebc88f10` fetches in its `CMakeLists.txt` |

The GB10 archive was checked against upstream's `tensorsharp-2026.10.11-linux-arm64-cuda13-GB10-SHA256SUMS` before
the notices were taken from it.
