# GGML driver releases

Optional GGML driver archives are published as GitHub release assets in
`DarkspyreStudios/darkspyre-releases`. Each archive holds the native files for one runtime
identifier (RID) and one variant, such as `linux-x64` with `cuda12`. The driver files are upstream
TensorSharp's own builds, taken unchanged from upstream's release archives. `tools/ggml_drivers.py`
extracts them, validates a release, writes the files to upload, checks GitHub for collisions and
verifies a published release. `docs/ggml-drivers-publish-checklist.md` lists the steps of a
publication.

The tool never creates a tag or a release and never uploads an asset. Publishing a driver release is
an ops task: the agent that prepares the release runs the `gh release` commands in the checklist,
with no per-version approval.

## Trust

GitHub is a locator. It is not a trust authority. The trusted SHA-256 catalog ships inside the
`core.inference` module package, and the host verifies every download against that catalog. The
host must not accept a hash, size, version, RID or asset name from GitHub or from `release.json`
in place of the catalog's. `release.json` therefore carries no hashes.

A published version is immutable. The tool refuses a version whose tag, release or archive name
already exists. A corrected build gets a new TensorSharp version.

## Release layout

| Item | Form | Example |
| --- | --- | --- |
| Tag | `ggml-drivers-v<version>` | `ggml-drivers-v2.9.0` |
| Release title | `GGML drivers <version>` | `GGML drivers 2.9.0` |
| Archive asset | `ggml-<version>-<rid>-<variant>.<format>` | `ggml-2.9.0-linux-x64-cuda12.tar.gz` |
| Release manifest asset | `release.json` | `release.json` |
| Download URL | `https://github.com/DarkspyreStudios/darkspyre-releases/releases/download/<tag>/<asset>` | |

`<version>` is the TensorSharp package version the natives belong to. `<rid>`, `<variant>` and
`<format>` are the artifact's `rid`, `variant` and `archive.format` fields. The archive asset name
must equal the artifact's `archive.name`, and the tool refuses a manifest where it differs. A
release holds exactly `release.json` and one archive per hosted RID and variant.

GitHub release assets share one flat namespace per release, so the RID and variant are part of the
asset name. Asset names use only letters, digits, `.`, `-` and `_`, so GitHub stores them unchanged.

Every GGML driver release is created with `--latest=false`. The repository holds several
components, and GitHub's "latest" marker must not point at a driver release.

## Extraction from upstream TensorSharp

`extract-upstream` builds the driver archives from upstream TensorSharp's CLI release archives,
`tensorsharp-cli-<version>-<suffix>` in zhongkaifu/TensorSharp's release `v<version>`. Each driver
file sits at the root of its upstream archive. A constant in the tool lists the variants:

| RID | Variant | Upstream archive suffix | Backends | Entry library | Preload, in order | Other files | Notices |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `osx-arm64` | `metal` | `osx-arm64.tar.gz` | cpu, metal | `libGgmlOps.dylib` | | | TensorSharp, ggml |
| `osx-arm64` | `mlx` | `osx-arm64.tar.gz` | mlx | `libmlxc.dylib` | | `libmlx.dylib`, `mlx.metallib` | TensorSharp, MLX, mlx-c |
| `linux-x64` | `cpu` | `linux-x64-cpu.tar.gz` | cpu | `libGgmlOps.so` | | | TensorSharp, ggml |
| `linux-x64` | `cuda12` | `linux-x64-cuda.tar.gz` | cpu, cuda | `libGgmlOps.so` | `libcudart.so.12`, `libcublasLt.so.12`, `libcublas.so.12` | | TensorSharp, ggml, NVIDIA cudart, NVIDIA cuBLAS |
| `linux-arm64` | `cuda13-gb10` | `linux-arm64-cuda13-GB10.tar.gz` | cpu, cuda | `libGgmlOps.so` | `libcudart.so.13`, `libcublasLt.so.13`, `libcublas.so.13` | | TensorSharp, ggml, NVIDIA cudart, NVIDIA cuBLAS |
| `win-x64` | `cpu` | `win-x64-cpu.zip` | cpu | `GgmlOps.dll` | | | TensorSharp, ggml |
| `win-x64` | `cuda12` | `win-x64-cuda.zip` | cpu, cuda | `GgmlOps.dll` | `cudart64_12.dll`, `cublasLt64_12.dll`, `cublas64_12.dll` | | TensorSharp, ggml, NVIDIA cudart, NVIDIA cuBLAS |

The `cuda13-gb10` row carries the note "Built for NVIDIA GB10 (Grace Blackwell) GPUs only.",
because upstream builds its CUDA code for that GPU alone. No other row has a note.
`libjaccl.dylib`, MLX's distributed backend, is not extracted, because no driver file links it.

Each driver archive holds the row's files at its root, in the table's order, then each notice under
`licenses/`. The notices come from `tools/notices/`, whose `README.md` names each file's source.
Entries carry mode 0644 and no timestamp, so equal input gives equal archives.

`extract-upstream` refuses, and leaves no output, when:

- an upstream archive is missing, or has no listed file at its root, or holds one as anything other
  than a regular file;
- the GB10 archive's SHA-256 differs from the line for it in
  `tensorsharp-<version>-linux-arm64-cuda13-GB10-SHA256SUMS` in the same folder, or that file is
  missing or lists the archive other than once. Upstream publishes sums for that archive only.

It writes the driver archives and `tensorsharp-native-artifacts.json`, the artifact manifest
`generate` reads, to the output folder. Each artifact records the upstream tag commit as both
`tensorSharp.packageCommit` and `tensorSharp.nativeSourceCommit`.

## Input: the TensorSharp native artifact manifest

One JSON file is the single input for archive names, sizes, hashes and contents. It has the
`tensorsharp-native-artifacts` shape, and `extract-upstream` writes it. The tool reads the fields
below and ignores every other field.

Every artifact must have an archive. At least one artifact is required.

| Field | Rule |
| --- | --- |
| `schema` | `tensorsharp-native-artifacts/1`. |
| `artifacts[].rid` | One of `osx-arm64`, `osx-x64`, `linux-x64`, `linux-arm64`, `win-x64`, `win-arm64`. |
| `artifacts[].variant` | Lowercase letters and digits with inner hyphens, such as `cpu`, `cuda12`, `cuda13-gb10`. Each RID and variant pair is hosted once. |
| `artifacts[].delivery` | Absent or `variant-archive` on a hosted artifact. |
| `artifacts[].tensorSharp.packageVersion` | The release version: two to four numeric components with an optional prerelease suffix. Equal on every hosted artifact. |
| `artifacts[].tensorSharp.packageCommit` | The full lowercase TensorSharp commit id of the release. Equal on every hosted artifact. |
| `artifacts[].tensorSharp.nativeSourceCommit` | The full lowercase commit id the natives were built from. Equal on every hosted artifact. |
| `artifacts[].backends` | A non-empty list of distinct lowercase backend names, such as `["cpu", "cuda"]`. |
| `artifacts[].entryLibrary` | The path of the library the engine loads. It must be a regular file listed in `files`. |
| `artifacts[].preload` | Optional, default empty. The libraries the engine loads by absolute path, in order, before the entry library. Each must be a top-level regular file listed in `files`, other than `entryLibrary`, and appear once. |
| `artifacts[].note` | Optional. One line of at most 200 characters that the app shows on the variant's row, such as a hardware limit. |
| `artifacts[].files` | Every native file in the archive: `{path, size, sha256}` for a regular file. |
| `artifacts[].notices` | Every license and notice file in the archive: a path, or `{path, size, sha256}`. At least one is required. |
| `artifacts[].totalSize` | Optional. When present, the sum of the `files` sizes. |
| `artifacts[].archive` | `{name, format, size, sha256}`. `format` is `tar.gz` or `zip`. `name` follows the release layout. |

Paths are relative POSIX paths. A path must not be absolute or hold a backslash, a control
character, an empty, `.` or `..` segment, a character Windows cannot store (`<>:"|?*`), a segment
ending in a dot or space, or a reserved Windows device name such as `con` or `nul`. Paths are unique
case-insensitively. A path in both `files` and `notices` is refused.

The archives sit in one directory, under their `archive.name` values.

## Archive rules

The tool opens every hosted archive and refuses it when:

- its length or SHA-256 differs from `archive.size` or `archive.sha256`;
- a listed file or notice is missing;
- a regular file's length or SHA-256 differs from its `files` entry, or from its `notices` entry
  when that entry carries them;
- it holds a file that neither `files` nor `notices` lists;
- it holds a directory that contains no listed file;
- an entry name breaks the path rules above, or appears twice;
- two entry names differ only by letter case;
- an entry is a symbolic link, a hard link, a device or a FIFO;
- a zip entry is encrypted;
- a listed path passes through an entry that is not a directory.

A leading `./` on tar entry names is accepted and removed. `files` and `notices` together describe
the archive exactly, so an unlisted executable cannot enter a release.

## Generated files

`generate` writes one release directory, by default `tmp/releases/<tag>/`:

```
<tag>/assets/release.json
<tag>/assets/ggml-<version>-<rid>-<variant>.<format>
<tag>/ggml-drivers-<version>.catalog.json
<tag>/notes.md
```

`assets/` holds exactly the files to upload. The archives are hard links to the input archives, or
copies when the input sits on another filesystem. The catalog input and the release notes sit
outside `assets/` and are not uploaded.

The catalog input derives from the artifact manifest and the measured archives. `release.json`
derives from the catalog input alone, and `generate`, `verify-listing` and `verify-published`
regenerate it and require byte equality. Artifacts are sorted by RID, then variant. `files` and
`notices` are sorted by path. Both files are two-space-indented ASCII JSON with a trailing newline,
so equal input gives equal bytes.

### Catalog input

`ggml-drivers-<version>.catalog.json` is the input AgentDS uses to generate the trusted
`core.inference` module catalog. It is handed to the AgentDS integration owner and is not uploaded.

```json
{
  "schema": "darkspyre.ggml-drivers.catalog/1",
  "component": "ggml-drivers",
  "version": "2.9.0",
  "tag": "ggml-drivers-v2.9.0",
  "repository": "DarkspyreStudios/darkspyre-releases",
  "tensorSharp": {
    "version": "2.9.0",
    "commit": "<40 hex digits>",
    "nativeSourceCommit": "<40 hex digits>"
  },
  "source": {"schema": "tensorsharp-native-artifacts/1", "sha256": "<sha256 of the artifact manifest>"},
  "artifacts": [
    {
      "rid": "linux-x64",
      "variant": "cuda12",
      "backends": ["cpu", "cuda"],
      "entryLibrary": "libGgmlOps.so",
      "preload": ["libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"],
      "asset": {
        "name": "ggml-2.9.0-linux-x64-cuda12.tar.gz",
        "format": "tar.gz",
        "size": 734003,
        "sha256": "<64 hex digits>",
        "url": "https://github.com/DarkspyreStudios/darkspyre-releases/releases/download/ggml-drivers-v2.9.0/ggml-2.9.0-linux-x64-cuda12.tar.gz"
      },
      "files": [
        {"path": "libGgmlOps.so", "size": 630929664, "sha256": "<64 hex digits>"},
        {"path": "libcublas.so.12", "size": 108244960, "sha256": "<64 hex digits>"},
        {"path": "libcublasLt.so.12", "size": 491106832, "sha256": "<64 hex digits>"},
        {"path": "libcudart.so.12", "size": 716128, "sha256": "<64 hex digits>"}
      ],
      "notices": [
        {"path": "licenses/NVIDIA-cuBLAS.txt", "size": 67875, "sha256": "<64 hex digits>"},
        {"path": "licenses/NVIDIA-cudart.txt", "size": 67875, "sha256": "<64 hex digits>"},
        {"path": "licenses/TensorSharp-LICENSE.txt", "size": 1498, "sha256": "<64 hex digits>"},
        {"path": "licenses/ggml-LICENSE.txt", "size": 1078, "sha256": "<64 hex digits>"}
      ]
    }
  ]
}
```

Every notice carries the size and SHA-256 measured from the validated archive, whether or not the
artifact manifest listed them. `files` entries carry the manifest's values, which the archive check
confirmed. `preload` is always present, possibly empty.
An artifact with a note carries it as `note` after `preload`; the catalog has no `ggml` block.

### release.json

`release.json` is uploaded with the archives. It locates each archive and carries no hashes.

```json
{
  "schema": "darkspyre.ggml-drivers.release/1",
  "component": "ggml-drivers",
  "version": "2.9.0",
  "tag": "ggml-drivers-v2.9.0",
  "repository": "DarkspyreStudios/darkspyre-releases",
  "tensorSharp": {"version": "2.9.0", "commit": "<40 hex digits>"},
  "source": {"sha256": "<sha256 of the artifact manifest>"},
  "assets": [
    {
      "rid": "linux-x64",
      "variant": "cuda12",
      "name": "ggml-2.9.0-linux-x64-cuda12.tar.gz",
      "format": "tar.gz",
      "size": 734003,
      "url": "https://github.com/DarkspyreStudios/darkspyre-releases/releases/download/ggml-drivers-v2.9.0/ggml-2.9.0-linux-x64-cuda12.tar.gz"
    }
  ]
}
```

`source.sha256` ties `release.json` to the catalog input built from the same artifact manifest.

## Collision refusal

`check-absent` lists the repository's tags and releases with read-only `gh api` GET requests,
100 per page across every page. It refuses the release when:

- the tag exists;
- a release for the tag exists, published or draft;
- any release holds an asset with one of the archive names.

`release.json` appears in every release and is not part of the asset name check. A failed `gh`
call is a refusal, because the tool cannot confirm what GitHub holds. GitHub lists draft releases
only to accounts with push access, so `check-absent` must run as such an account.

## Listing check

`verify-listing` finds the one release for the tag and refuses it unless it holds exactly
`release.json` and the archives, each with state `uploaded` and the generated size. When GitHub
reports an asset `digest`, it must equal `sha256:<catalog sha256>`, or the SHA-256 of the generated
`release.json`. A published release must also have its tag. The command reports whether the release
is a draft.

## Download behaviour

A download URL answers `302 Found` with a `Location` on `release-assets.githubusercontent.com`.
That location is signed and expires after a few minutes. It answers with
`Content-Type: application/octet-stream`, `Accept-Ranges: bytes`, `ETag`, `Last-Modified` and no
`Content-Encoding`, so byte offsets and the SHA-256 refer to the stored archive. A single byte range
answers `206` with `Content-Range: bytes <first>-<last>/<size>`.

A resumable client keeps the partial file and the `ETag` of its first answer. To resume, it
requests the `github.com` download URL again, so it follows a fresh signed redirect, and sends
`Range: bytes=<partial length>-` with `If-Range: <etag>`. It appends a `206` answer whose
`Content-Range` starts at the partial length and ends at the expected size. It restarts the file on
a `200` answer and discards the partial file on a `416` answer. It verifies size and SHA-256
against the trusted catalog after the download completes, whatever the server returned.

`verify-published` downloads `release.json` and every archive this way. It requires
`release.json` to equal the generated bytes. For each archive it requires a `206` answer to a
last-byte range request, then downloads with resume and applies every archive rule against the
catalog input. It deletes each download after checking it, unless `--keep` is given.

## Commands

Run from the repository root. Each command exits 0 on success, 1 on a refused release and 2 on a
usage error.

```bash
# Write the driver archives and tmp/upstream/<version>/tensorsharp-native-artifacts.json from
# upstream's CLI archives and the GB10 SHA256SUMS file, downloaded into <upstream-dir>.
python3 tools/ggml_drivers.py extract-upstream --version <version> --archives <upstream-dir> \
  --commit <upstream tag commit> [--out <dir>]

# Check the artifact manifest and its archives.
python3 tools/ggml_drivers.py validate <artifacts.json> <archive-dir>

# Validate, then write tmp/releases/ggml-drivers-v<version>/.
python3 tools/ggml_drivers.py generate <artifacts.json> <archive-dir> [--out <dir>]

# Refuse when the tag, a release for it or an archive name already exists on GitHub.
python3 tools/ggml_drivers.py check-absent tmp/releases/ggml-drivers-v<version>

# Check the tag's release, draft or published, holds exactly the generated assets.
python3 tools/ggml_drivers.py verify-listing tmp/releases/ggml-drivers-v<version>

# Download every asset through its public URL and check it.
python3 tools/ggml_drivers.py verify-published tmp/releases/ggml-drivers-v<version> [--keep]
```

`--gh <command>` before the command name replaces the `gh` executable.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests build fixture archives, an artifact manifest in the DSA-98 draft shape and synthetic
upstream CLI archives under `tmp/tests/`, and remove them afterwards. `tests/stub_gh.py` answers the `gh api` listings from a
JSON state file and exits with an error for any other command. `tests/github_server.py` serves
assets on 127.0.0.1 through a `302` redirect to single-use signed URLs, with range, `If-Range` and
dropped-connection behaviour. No test reaches the public network.
