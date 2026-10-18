# GGML driver publish checklist

This checklist publishes one GGML driver version as the GitHub release
`ggml-drivers-v<version>` in `DarkspyreStudios/darkspyre-releases`. `docs/ggml-drivers.md`
describes the layout, the generated files and the validation rules. Every step must pass before the
next one starts. A refused step stops the publication.

In the commands below, `<tag>` is `ggml-drivers-v<version>`, `<dir>` is `tmp/releases/<tag>`,
`<upstream>` is `tmp/downloads/v<version>` and `<extracted>` is `tmp/upstream/<version>`.

## Rules

- Publishing is an ops task. The agent that prepares the release runs every `gh release` command
  below, with no per-version approval.
- No script creates, edits or deletes a release.
- The release is created as a draft. A draft is not public and creates no tag, so an incomplete
  upload never becomes a visible release.
- A published release must never be edited, have an asset replaced or deleted, or have its tag moved.
  A corrected build gets a new version from the TensorSharp release.
- The archives, sizes and hashes come only from the artifact manifest that `extract-upstream` writes.
  They must not be edited in this repository.
- Upstream's archives are untrusted data. Nothing in them is executed; the tool only reads them.

## One-time repository preparation

1. Give the repository a default branch with at least one commit. `gh release create` tags the
   default branch head when the draft is published.
2. In the repository settings, enable release immutability, so GitHub itself refuses changes to a
   published release's assets and tag.
3. Confirm `gh auth status --hostname github.com` shows an account with push access to the
   repository. GitHub lists draft releases only to such accounts.

## Each release

1. Download upstream TensorSharp's six CLI archives and the GB10 sums file into the empty folder
   `<upstream>`, then extract the drivers. `<commit>` is the full commit of upstream's tag
   `v<version>` (`gh api repos/zhongkaifu/TensorSharp/git/ref/tags/v<version>`):

   ```bash
   gh release download v<version> -R zhongkaifu/TensorSharp -D <upstream> \
     -p 'tensorsharp-cli-<version>-*' -p 'tensorsharp-<version>-linux-arm64-cuda13-GB10-SHA256SUMS'
   python3 tools/ggml_drivers.py extract-upstream --version <version> --archives <upstream> --commit <commit>
   ```

2. Run the tests from the repository root and confirm they pass:

   ```bash
   python3 -m unittest discover -s tests -v
   ```

3. Validate and generate the release directory:

   ```bash
   python3 tools/ggml_drivers.py generate <extracted>/tensorsharp-native-artifacts.json <extracted>
   ```

   Record the catalog input SHA-256 it prints. Read `<dir>/notes.md` and `<dir>/assets/release.json`.
4. Confirm the version does not exist on GitHub:

   ```bash
   python3 tools/ggml_drivers.py check-absent <dir>
   ```

5. Confirm `<dir>/assets/` holds only `release.json` and the archives listed in `release.json`.
6. Create the draft release with every asset:

   ```bash
   gh release create <tag> <dir>/assets/* \
     --repo DarkspyreStudios/darkspyre-releases \
     --draft --latest=false \
     --title "GGML drivers <version>" \
     --notes-file <dir>/notes.md
   ```

   When an upload fails, the draft keeps the assets that finished. Upload the rest to the draft with
   `gh release upload <tag> <files> --repo DarkspyreStudios/darkspyre-releases`, or delete the
   draft with `gh release delete <tag> --repo DarkspyreStudios/darkspyre-releases` and repeat this
   step. Only a draft may be changed this way.
7. Check the draft's asset listing:

   ```bash
   python3 tools/ggml_drivers.py verify-listing <dir>
   ```

   It must report a draft release. A note about a missing digest is not a failure. Step 10 checks
   every hash.
8. Publish the draft:

   ```bash
   gh release edit <tag> --repo DarkspyreStudios/darkspyre-releases --draft=false --latest=false
   ```

9. Check the published listing and the tag:

   ```bash
   python3 tools/ggml_drivers.py verify-listing <dir>
   ```

   It must report a published release.
10. Download and check every asset through its public URL:

    ```bash
    python3 tools/ggml_drivers.py verify-published <dir>
    ```

    This downloads every archive once, with resume, and deletes each one after checking it.
11. Hand `<dir>/ggml-drivers-<version>.catalog.json` and its SHA-256 from step 3 to the AgentDS
    integration owner. The `core.inference` module catalog must be generated from this file. Its
    RID, variant, asset, size, SHA-256, file and notice entries must equal the catalog input's.
12. Keep a copy of the catalog input and `release.json` outside `tmp/`, then delete `<dir>`,
    `<extracted>` and `<upstream>`.

## Withdrawal

The tooling cannot withdraw or replace a published version. Removing a release breaks every
installed module package whose catalog points at it, so a withdrawal is an owner decision.
