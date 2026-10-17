# Apple Foundation Models helper

A signed macOS source build of the Darkspyre app copies the prebuilt helper `darkspyre-apple-fm` into the bundle.
The build reads its path from the Core setting `apple-helper-location` and never compiles it.

| Item | Form | Example |
| --- | --- | --- |
| Tag | `apple-fm-helper-<rid>-<sha12>`, where `<sha12>` is the first 12 characters of the helper's SHA-256 | `apple-fm-helper-osx-arm64-309bb91798d1` |
| Release title | `Apple Foundation Models helper <rid> <sha12>` | |
| Assets | `darkspyre-apple-fm` and `SHA256SUMS` | |

A downloaded helper needs `chmod +x`. Every release is created with `--latest=false`. A published release is never
edited. A new build of the helper goes in a new release.
