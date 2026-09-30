# GGML driver publish checklist

This checklist publishes one GGML driver version as the GitHub release
`ggml-drivers-v<version>` in `DarkspyreStudios/darkspyre-releases`. `docs/ggml-drivers.md`
describes the layout, the generated files and the validation rules. Every step must pass before the
next one starts. A refused step stops the publication.

In the commands below, `<tag>` is `ggml-drivers-v<version>` and `<dir>` is
`tmp/releases/<tag>`.

## Rules

- The owner must explicitly approve publication of each version before step 6.
- The owner runs every `gh release` command by hand. No script creates, edits or deletes a release.
- The release is created as a draft. A draft is not public and creates no tag, so an incomplete
  upload never becomes a visible release.
- A published release must never be edited, have an asset replaced or deleted, or have its tag moved.
  A corrected build gets a new version from the TensorSharp release.
- The archives, sizes and hashes come only from the TensorSharp native artifact manifest. They must
  not be edited in this repository.

## One-time repository preparation

1. Give the repository a default branch with at least one commit. `gh release create` tags the
   default branch head when the draft is published.
2. In the repository settings, enable release immutability, so GitHub itself refuses changes to a
   published release's assets and tag.
3. Confirm `gh auth status --hostname github.com` shows an account with push access to the
   repository. GitHub lists draft releases only to such accounts.

## Each release

1. Obtain the TensorSharp native artifact manifest and its archives from the validated TensorSharp
   release. Put the archives in one directory.
2. Run the tests from the repository root and confirm they pass:

   ```bash
   python3 -m unittest discover -s tests -v
   ```

3. Validate and generate the release directory:

   ```bash
   python3 tools/ggml_drivers.py generate <artifacts.json> <archive-dir>
   ```

   Record the catalog input SHA-256 it prints. Read `<dir>/notes.md` and `<dir>/assets/release.json`.
4. Confirm the version does not exist on GitHub:

   ```bash
   python3 tools/ggml_drivers.py check-absent <dir>
   ```

5. Confirm `<dir>/assets/` holds only `release.json` and the archives listed in `release.json`.
6. Confirm the owner has approved publication of this version.
7. Create the draft release with every asset:

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
8. Check the draft's asset listing:

   ```bash
   python3 tools/ggml_drivers.py verify-listing <dir>
   ```

   It must report a draft release. A note about a missing digest is not a failure. Step 11 checks
   every hash.
9. Publish the draft:

   ```bash
   gh release edit <tag> --repo DarkspyreStudios/darkspyre-releases --draft=false --latest=false
   ```

10. Check the published listing and the tag:

    ```bash
    python3 tools/ggml_drivers.py verify-listing <dir>
    ```

    It must report a published release.
11. Download and check every asset through its public URL:

    ```bash
    python3 tools/ggml_drivers.py verify-published <dir>
    ```

    This downloads every archive once, with resume, and deletes each one after checking it.
12. Hand `<dir>/ggml-drivers-<version>.catalog.json` and its SHA-256 from step 3 to the AgentDS
    integration owner. The `core.inference` module catalog must be generated from this file. Its
    RID, variant, asset, size, SHA-256, file and notice entries must equal the catalog input's.
13. Keep a copy of the catalog input and `release.json` outside `tmp/`, then delete `<dir>`.

## Withdrawal

The tooling cannot withdraw or replace a published version. Removing a release breaks every
installed module package whose catalog points at it, so a withdrawal is an owner decision.
