# darkspyre-releases

This public repository holds Darkspyre project release artifacts as GitHub release assets. The
source tree holds only the documentation and the tooling that validates, prepares and verifies
those releases. It holds no source code for the artifacts themselves, and no artifact is committed
to git.

## Components

| Component | Tag | Documentation |
| --- | --- | --- |
| GGML driver archives | `ggml-drivers-v<version>` | [docs/ggml-drivers.md](docs/ggml-drivers.md) |

Each component uses its own tag prefix, so the releases of different components never share a tag.
A published release is immutable. A corrected build is published under a new version.

## Trust

GitHub is a download locator. It is not a trust authority. Each consumer verifies a download
against a trusted catalog that ships separately from the download. For GGML drivers, the trusted
SHA-256 catalog ships inside the `core.inference` module package.

## Layout

| Path | Contents |
| --- | --- |
| `docs/` | Release conventions and the publish checklist for each component. |
| `tools/` | Python tooling. It needs Python 3.10 or later and the standard library only. |
| `tests/` | Tests against local fixtures, a local HTTP server and a stub `gh`. |
| `tmp/` | Scratch output, ignored by git. |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests make no network calls.
