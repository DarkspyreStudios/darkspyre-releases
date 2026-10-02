# TensorSharp direct CUDA builds

The `tensorsharp-cuda-builds` component stores TensorSharp direct CUDA library builds.
Its tags use `tensorsharp-cuda-builds-v<artifact-version>`. Its archives use
`tensorsharp-cuda-<artifact-version>-<rid>-<variant>.zip`.

This component is separate from GGML drivers. It contains a managed CUDA backend
and PTX modules. It does not supply `GgmlOps.dll` or satisfy a GGML driver catalog.
The artifact version identifies a build archive, not a published NuGet package.

Each release is immutable, uses `--latest=false`, and contains a portable-path ZIP
plus `build-info.json`. The build information records source identity, measured
toolchains, runtime prerequisites and actual hardware qualification. SHA-256 and
per-file identities belong to the separate supplier catalog input handed to the
trusted consumer catalog. Public release metadata is not a trust authority.

## RTX 3090 Ti build

Artifact `2.8.6.7-sm86.1` is a prerelease build for `win-x64` and variant
`cuda12-sm86`. It comes from TensorSharp source tag `v2.8.6.7` plus the CUDA infinity
fix on commit `c7fd85b5609a92c7c56dbce116914fc97d757fa3`.
The build uses CUDA 12.8.61, MSVC 14.44.35207 and .NET SDK 10.0.201. Both modules
contain PTX 8.7 targeting `sm_86`.

The GPU smoke runs on an RTX 3090 Ti with NVIDIA driver 595.97. Both modules load.
Fill, addition, cuBLAS multiplication and DSV4 argmax infinity/tie cases pass with
zero CPU fallback operations. The tested driver is a qualification baseline;
it is not a claim about NVIDIA's minimum supported driver.

This archive contains the backend DLL, matching managed support DLLs, PTX, package
dependency metadata and the TensorSharp BSD license. CUDA runtime libraries remain
external prerequisites. The verified smoke resolves CUDA 12 cuBLAS DLLs from the
installed toolkit. The archive is not an installation-ready driver with a complete
runtime dependency closure. It is excluded from automatic downloader eligibility
until the supplier and consumer certify that closure and exact build compatibility.
Full Qwen inference and speed are not qualified by the tiny-tensor smoke.
