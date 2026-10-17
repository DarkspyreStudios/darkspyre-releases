# GGML baseline driver packages

A source build of the Darkspyre app bundles one GGML driver per platform. The build reads it from a
`Darkspyre.TensorSharp.Backends.GGML.Native.<rid>.<version>.nupkg` file under the folder the Core setting
`native-drivers-location` names, and copies the package whose native ABI equals the `nativeAbi` in TensorSharp's
`eng/ggml-required-exports.json`. agent-ds `docs/workspace-setup.md` describes the setup.

Each release holds the packages for one native ABI.

| Item | Form | Example |
| --- | --- | --- |
| Tag | `ggml-packages-<abi16>`, where `<abi16>` is the first 16 characters of the native ABI | `ggml-packages-6dd126abcccee668` |
| Release title | `GGML driver packages <abi16>` | `GGML driver packages 6dd126abcccee668` |
| Assets | One `Darkspyre.TensorSharp.Backends.GGML.Native.<rid>.<version>.nupkg` per RID, and `SHA256SUMS` | |

A package's `ggml/baseline.artifact.json` holds its full `nativeAbi`, which must start with the tag's `<abi16>`.
Every release is created with `--latest=false`. A published release is never edited. A package built for a new
native ABI goes in a new release.
