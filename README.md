# darkspyre-releases

<!-- ownership-notice:start -->
> **Proprietary software.** Copyright © 2026 Brandon Mathis. All rights reserved.
> Published releases may be run under the terms in LICENSE.md. No other
> permission is granted to use, copy, modify, or distribute this software
> without prior written authorization.
> Any contribution, including code, reviews, patches, issues and designs, is
> assigned to Brandon Mathis on submission. See [LICENSE.md](LICENSE.md) and
> [CONTRIBUTING.md](CONTRIBUTING.md).
<!-- ownership-notice:end -->

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

The tests use local fixture archives, a stub GitHub CLI and a loopback HTTP server. They make no
public network calls. Fixture files live under `/Volumes/Data/tmp/dsa-675-release-catalog/` and
are removed after each test.
