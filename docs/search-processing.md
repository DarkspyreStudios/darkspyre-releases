# Search processing payload

Darkspyre Search builds need a prepared OCR/media payload (leptonica, tesseract, its tessdata and ffmpeg) in
`Search/artifacts/processing/<rid>`. Search's `build/prepare-processing.py` produces it, and git does not hold it.
agent-ds `scripts/workspace-env.sh` downloads it into a workspace that lacks it.

| Item | Form | Example |
| --- | --- | --- |
| Tag | `search-processing-<rid>-<sha12>`, where `<sha12>` is the first 12 characters of the payload's `processing-provenance.json` SHA-256 | `search-processing-osx-arm64-e3dbe3336fd5` |
| Assets | `search-processing-<rid>.tar.gz`, the payload folder's contents, and `SHA256SUMS` | |

`processing-provenance.json` lists every file's SHA-256. Every release is created with `--latest=false`. A published
release is never edited. A new payload goes in a new release.
