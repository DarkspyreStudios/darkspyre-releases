#!/usr/bin/env python3
"""Validate, generate and verify GGML driver releases hosted as GitHub release assets.

docs/ggml-drivers.md describes the input, the release layout, both generated documents and the
validation rules. docs/ggml-drivers-publish-checklist.md describes a publication.

Commands:
  validate          check a TensorSharp artifact manifest against its local archives
  generate          validate, then write a release directory with the assets to upload and the catalog input
  check-absent      refuse when the tag, a release for it or any archive name already exists on GitHub
  verify-listing    check a release's GitHub asset listing against a release directory
  verify-published  download a published release through its public URLs and check every asset

This tool never creates a tag or a release and never uploads an asset. It calls `gh` only for
read-only `gh api` GET requests.

Every command exits 0 on success, 1 on a refused release and 2 on a usage error.
"""
import argparse
import hashlib
import http.client
import json
import os
import posixpath
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "DarkspyreStudios/darkspyre-releases"
GITHUB = "https://github.com"

COMPONENT = "ggml-drivers"
TAG_PREFIX = "ggml-drivers-v"
ARCHIVE_PREFIX = "ggml"
RELEASE_ASSET = "release.json"
ASSETS_DIR = "assets"
NOTES_FILE = "notes.md"
RELEASE_SCHEMA = "darkspyre.ggml-drivers.release/1"
CATALOG_SCHEMA = "darkspyre.ggml-drivers.catalog/1"
INPUT_SCHEMAS = frozenset({"tensorsharp-native-artifacts/draft-1", "tensorsharp-native-artifacts/1"})
HOSTED_DELIVERY = "variant-archive"

RIDS = frozenset({"osx-arm64", "osx-x64", "linux-x64", "linux-arm64", "win-x64", "win-arm64"})
FORMATS = ("tar.gz", "zip")
VERSION = re.compile(r"^[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z]+(\.[0-9A-Za-z]+)*)?$")
COMMIT = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
VARIANT = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
BACKEND = re.compile(r"^[a-z][a-z0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul"}
                             | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)})
WINDOWS_FORBIDDEN = set('<>:"|?*')

CHUNK = 1024 * 1024
PAGE_SIZE = 100


class Refused(Exception):
    """A release failed validation. Each argument is one problem."""


def dumps(value) -> bytes:
    """Serialize a generated document. The output is byte-stable for equal input."""
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def tag_for(version: str) -> str:
    return f"{TAG_PREFIX}{version}"


def archive_name(version: str, rid: str, variant: str, fmt: str) -> str:
    return f"{ARCHIVE_PREFIX}-{version}-{rid}-{variant}.{fmt}"


def asset_url(tag: str, name: str, base: str = GITHUB, repository: str = REPOSITORY) -> str:
    return f"{base.rstrip('/')}/{repository}/releases/download/{tag}/{name}"


def catalog_file_name(version: str) -> str:
    return f"{COMPONENT}-{version}.catalog.json"


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def relative_path_problem(value) -> str | None:
    """Return why value is not a clean relative POSIX path that extracts safely on every RID, or None."""
    if not isinstance(value, str) or not value:
        return "is not a non-empty string"
    if "\\" in value or any(ord(c) < 32 for c in value):
        return "contains a backslash or a control character"
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return "is absolute"
    for segment in value.split("/"):
        if segment in ("", ".", ".."):
            return "has an empty, '.' or '..' segment"
        if WINDOWS_FORBIDDEN & set(segment):
            return "contains a character Windows cannot store"
        if segment.endswith((".", " ")):
            return "has a segment ending in a dot or space"
        if segment.split(".")[0].casefold() in WINDOWS_RESERVED:
            return "uses a reserved Windows device name"
    return None


# Input: the TensorSharp native artifact manifest.

def check_file_list(label: str, value, problems: list[str], notices: bool) -> list[dict]:
    """Normalize a files or notices list. A payload entry has size and sha256, or link. A notice may be a bare path."""
    if not isinstance(value, list) or not value:
        problems.append(f"{label} must be a non-empty list")
        return []
    seen: set[str] = set()
    clean: list[dict] = []
    for index, item in enumerate(value):
        where = f"{label}[{index}]"
        if notices and isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            problems.append(f"{where} must be an object{' or a path' if notices else ''}")
            continue
        path = item.get("path")
        problem = relative_path_problem(path)
        if problem:
            problems.append(f"{where} path {path!r} {problem}")
            continue
        if path.casefold() in seen:
            problems.append(f"{where} path {path} is listed more than once (case-insensitively)")
            continue
        seen.add(path.casefold())
        entry: dict = {"path": path}
        if "link" in item and not notices:
            if not isinstance(item["link"], str) or not item["link"]:
                problems.append(f"{where} link must be a non-empty string")
                continue
            if "size" in item or "sha256" in item:
                problems.append(f"{where} is a link and must not carry size or sha256")
                continue
            entry["link"] = item["link"]
        elif "size" in item or "sha256" in item or not notices:
            size, digest = item.get("size"), item.get("sha256")
            if not is_int(size) or size < 0:
                problems.append(f"{where} size {size!r} must be a non-negative integer")
                continue
            if not isinstance(digest, str) or not SHA256.match(digest):
                problems.append(f"{where} sha256 {digest!r} must be 64 lowercase hex digits")
                continue
            entry.update(size=size, sha256=digest)
        clean.append(entry)
    return clean


def normalize_input_inventory(label: str, files: list[dict], notices: list[dict],
                              problems: list[str]) -> tuple[list[dict], list[dict]]:
    """Project exact matching source overlap into disjoint payload and notice lists."""
    by_path = {notice["path"].casefold(): notice for notice in notices}
    payload = []
    for item in files:
        notice = by_path.get(item["path"].casefold())
        if notice is None:
            payload.append(item)
        elif ("link" not in item and "size" in item and "sha256" in item
              and "size" in notice and "sha256" in notice and item == notice):
            continue
        else:
            problems.append(f"{label} path {item['path']} overlaps files and notices without identical regular-file metadata")
            payload.append(item)
    return payload, list(notices)


def is_hosted(artifact: dict) -> bool:
    return artifact.get("archive") is not None or artifact.get("delivery") == HOSTED_DELIVERY


def parse_input(data) -> dict:
    """Validate a TensorSharp artifact manifest and return the hosted release it describes.

    Artifacts with a null archive and no variant-archive delivery are delivered in baseline packages
    and are skipped. Unknown fields are ignored.
    """
    if not isinstance(data, dict):
        raise Refused("the artifact manifest must be a JSON object")
    problems: list[str] = []
    schema = data.get("schema")
    if schema not in INPUT_SCHEMAS:
        problems.append(f"schema {schema!r} must be one of {', '.join(sorted(INPUT_SCHEMAS))}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise Refused(*problems, "artifacts must be a list")

    identity: dict[str, set] = {"version": set(), "commit": set(), "nativeSourceCommit": set(), "ggml": set()}
    pairs: set[tuple] = set()
    hosted: list[dict] = []
    for index, artifact in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            problems.append(f"{label} must be an object")
            continue
        if not is_hosted(artifact):
            continue
        rid, variant = artifact.get("rid"), artifact.get("variant")
        label = f"{label} ({rid}/{variant})"
        if rid not in RIDS:
            problems.append(f"{label} rid {rid!r} must be one of {', '.join(sorted(RIDS))}")
        if not isinstance(variant, str) or not VARIANT.match(variant):
            problems.append(f"{label} variant {variant!r} must be lowercase letters, digits and inner hyphens")
        if (rid, variant) in pairs:
            problems.append(f"{label} repeats rid {rid} variant {variant}")
        pairs.add((rid, variant))
        delivery = artifact.get("delivery")
        if delivery is not None and delivery != HOSTED_DELIVERY:
            problems.append(f"{label} has an archive, so delivery must be {HOSTED_DELIVERY!r}, not {delivery!r}")

        ts = artifact.get("tensorSharp")
        if not isinstance(ts, dict):
            problems.append(f"{label} tensorSharp must be an object")
            ts = {}
        version, commit, native = ts.get("packageVersion"), ts.get("packageCommit"), ts.get("nativeSourceCommit")
        if not isinstance(version, str) or not VERSION.match(version):
            problems.append(f"{label} tensorSharp.packageVersion {version!r} must look like 2.8.6.7 or 2.9.0-rc.1")
        if not isinstance(commit, str) or not COMMIT.match(commit):
            problems.append(f"{label} tensorSharp.packageCommit {commit!r} must be a full lowercase git commit id")
        if not isinstance(native, str) or not COMMIT.match(native):
            problems.append(f"{label} tensorSharp.nativeSourceCommit {native!r} must be a full lowercase git commit id")
        identity["version"].add(version)
        identity["commit"].add(commit)
        identity["nativeSourceCommit"].add(native)
        ggml = artifact.get("ggml")
        if (not isinstance(ggml, dict) or not isinstance(ggml.get("version"), str) or not ggml["version"]
                or not isinstance(ggml.get("commit"), str) or not COMMIT.match(ggml["commit"])):
            problems.append(f"{label} ggml must hold a version and a full lowercase commit id")
            ggml = {}
        identity["ggml"].add((ggml.get("version"), ggml.get("commit")))

        native_abi = artifact.get("nativeAbi")
        if not isinstance(native_abi, str) or not SHA256.fullmatch(native_abi):
            problems.append(f"{label} nativeAbi {native_abi!r} must be 64 lowercase hex digits")

        backends = artifact.get("backends")
        if (not isinstance(backends, list) or not backends
                or not all(isinstance(b, str) and BACKEND.match(b) for b in backends)
                or len(set(backends)) != len(backends)):
            problems.append(f"{label} backends {backends!r} must be a non-empty list of distinct lowercase names")
            backends = []

        files = check_file_list(f"{label} files", artifact.get("files"), problems, notices=False)
        notices = check_file_list(f"{label} notices", artifact.get("notices"), problems, notices=True)
        total = artifact.get("totalSize")
        if total is not None and total != sum(f.get("size", 0) for f in files):
            problems.append(f"{label} totalSize {total!r} is not the sum of its file sizes")
        files, notices = normalize_input_inventory(label, files, notices, problems)
        entry = artifact.get("entryLibrary")
        if not any(f["path"] == entry and "link" not in f for f in files):
            problems.append(f"{label} entryLibrary {entry!r} must be a regular file listed in files")

        archive = artifact.get("archive")
        if not isinstance(archive, dict):
            problems.append(f"{label} has delivery {HOSTED_DELIVERY!r} but no archive object")
            continue
        fmt, name, size, digest = (archive.get(k) for k in ("format", "name", "size", "sha256"))
        if fmt not in FORMATS:
            problems.append(f"{label} archive.format {fmt!r} must be one of {', '.join(FORMATS)}")
        elif isinstance(version, str) and name != archive_name(version, rid, variant, fmt):
            problems.append(f"{label} archive.name {name!r} must be {archive_name(version, rid, variant, fmt)}")
        if not is_int(size) or size <= 0:
            problems.append(f"{label} archive.size {size!r} must be a positive integer")
        if not isinstance(digest, str) or not SHA256.match(digest):
            problems.append(f"{label} archive.sha256 {digest!r} must be 64 lowercase hex digits")
        hosted.append({
            "rid": rid, "variant": variant, "backends": backends, "entryLibrary": entry,
            "nativeAbi": native_abi,
            "archive": {"name": name, "format": fmt, "size": size, "sha256": digest},
            "files": files, "notices": notices,
        })

    for key, values in identity.items():
        if len(values) > 1:
            problems.append(f"hosted artifacts disagree on {'ggml' if key == 'ggml' else 'tensorSharp.' + key}; "
                            "one release holds one build")
    if not hosted and not problems:
        problems.append(f"no artifact has delivery {HOSTED_DELIVERY!r} and an archive; there is nothing to host")
    if problems:
        raise Refused(*problems)
    (version,), (commit,), (native,), ((ggml_version, ggml_commit),) = (identity[k] for k in identity)
    return {
        "schema": schema, "version": version, "commit": commit, "nativeSourceCommit": native,
        "ggml": {"version": ggml_version, "commit": ggml_commit}, "artifacts": hosted,
    }


# Archive inspection.

def hash_stream(handle) -> tuple[int, str]:
    digest, length = hashlib.sha256(), 0
    while block := handle.read(CHUNK):
        digest.update(block)
        length += len(block)
    return length, digest.hexdigest()


def read_entries(path: Path, fmt: str) -> tuple[dict[str, dict], list[str]]:
    """Return {name: {kind, target, size, sha256}} for every archive entry, and the entry-level problems."""
    entries: dict[str, dict] = {}
    problems: list[str] = []

    def add(raw: str, kind: str, **fields) -> None:
        name = raw[2:] if raw.startswith("./") else raw
        if kind == "dir":
            name = name.rstrip("/")
        if name in ("", "."):
            return
        problem = relative_path_problem(name)
        if problem:
            problems.append(f"entry {raw!r} {problem}")
        elif name in entries:
            problems.append(f"entry {name} appears more than once")
        else:
            entries[name] = {"kind": kind, **fields}

    try:
        if fmt == "tar.gz":
            with tarfile.open(path, "r|gz") as archive:
                for member in archive:
                    if member.isdir():
                        add(member.name, "dir")
                    elif member.isfile():
                        size, digest = hash_stream(archive.extractfile(member))
                        add(member.name, "file", size=size, sha256=digest)
                    elif member.issym():
                        add(member.name, "link", target=member.linkname)
                    else:
                        problems.append(f"entry {member.name!r} is a hard link or special file")
        else:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    mode = info.external_attr >> 16
                    if info.flag_bits & 0x1:
                        problems.append(f"entry {info.filename!r} is encrypted")
                    elif stat.S_ISLNK(mode):
                        problems.append(f"entry {info.filename!r} is a symbolic link; zip archives must not hold links")
                    elif info.is_dir():
                        add(info.filename, "dir")
                    else:
                        with archive.open(info) as handle:
                            size, digest = hash_stream(handle)
                        add(info.filename, "file", size=size, sha256=digest)
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as error:
        problems.append(f"cannot be read as {fmt}: {error}")
    return entries, problems


def check_archive_bytes(path: Path, archive: dict) -> list[str]:
    if not path.is_file() or path.is_symlink():
        return [f"{archive['name']}: {path} is missing or is not a regular file"]
    size = path.stat().st_size
    if size != archive["size"]:
        return [f"{archive['name']}: size is {size}, the manifest says {archive['size']}"]
    digest = sha256_file(path)
    if digest != archive["sha256"]:
        return [f"{archive['name']}: sha256 is {digest}, the manifest says {archive['sha256']}"]
    return []


def check_archive(path: Path, artifact: dict) -> tuple[list[str], list[dict]]:
    """Check one archive's bytes and exact contents. Return the problems and the measured notices."""
    archive = artifact["archive"]
    overlap = {f["path"].casefold() for f in artifact["files"]} & {n["path"].casefold() for n in artifact["notices"]}
    if overlap:
        return [f"{archive['name']}: lists {', '.join(sorted(overlap))} in both files and notices"], []
    problems = check_archive_bytes(path, archive)
    if problems:
        return problems, []
    label = archive["name"]
    entries, entry_problems = read_entries(path, archive["format"])
    problems = [f"{label}: {p}" for p in entry_problems]

    listed: dict[str, tuple[str, dict]] = {f["path"]: ("payload", f) for f in artifact["files"]}
    listed.update({n["path"]: ("notice", n) for n in artifact["notices"]})
    folded: dict[str, str] = {}
    for name in entries:
        if name.casefold() in folded:
            problems.append(f"{label}: entries {folded[name.casefold()]} and {name} differ only by case")
        folded[name.casefold()] = name

    ancestors = {"/".join(n.split("/")[:i]) for n in listed for i in range(1, n.count("/") + 1)}
    for name, found in sorted(entries.items()):
        kind = found["kind"]
        if kind == "dir":
            if name not in ancestors:
                problems.append(f"{label}: directory {name} holds no listed file")
            continue
        if name not in listed:
            problems.append(f"{label}: {'symbolic link' if kind == 'link' else 'file'} {name} is not listed in files or notices")
            continue
        role, expected = listed[name]
        if kind == "link":
            target = found["target"]
            if "link" not in expected:
                problems.append(f"{label}: {name} is a symbolic link, the manifest lists a regular {role} file")
                continue
            if target != expected["link"]:
                problems.append(f"{label}: symbolic link {name} -> {target!r}, the manifest says {expected['link']!r}")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target or ""))
            if (not target or target.startswith("/") or "\\" in target
                    or resolved == ".." or resolved.startswith("../")):
                problems.append(f"{label}: symbolic link {name} -> {target!r} escapes the archive")
            elif entries.get(resolved, {}).get("kind") != "file":
                problems.append(f"{label}: symbolic link {name} -> {target!r} does not name a regular file entry")
            continue
        if "link" in expected:
            problems.append(f"{label}: {name} is a regular file, the manifest lists a symbolic link")
            continue
        if "size" in expected and (found["size"], found["sha256"]) != (expected["size"], expected["sha256"]):
            problems.append(f"{label}: {role} {name} is {found['size']} bytes with sha256 {found['sha256']}, "
                            f"the manifest says {expected['size']} bytes with sha256 {expected['sha256']}")
    for name in sorted(listed):
        if name not in entries:
            problems.append(f"{label}: {listed[name][0]} {name} is missing from the archive")
        for ancestor in (a for a in ancestors if name.startswith(a + "/")):
            if entries.get(ancestor, {"kind": "dir"})["kind"] != "dir":
                problems.append(f"{label}: {name} passes through {ancestor}, which is not a directory")
    notices = [{"path": n["path"], "size": entries[n["path"]]["size"], "sha256": entries[n["path"]]["sha256"]}
               for n in artifact["notices"] if entries.get(n["path"], {}).get("kind") == "file"]
    return problems, notices


# Generated documents. The catalog input derives from the artifact manifest and the measured
# archives; release.json derives from the catalog input alone.

def load_json(path: Path) -> tuple[object, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Refused(f"cannot read {path}: {error}") from error
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as error:
        raise Refused(f"{path} is not valid JSON: {error}") from error


def validate(manifest_path: Path, archive_dir: Path) -> dict:
    """Validate the manifest and every hosted archive. Return the catalog input."""
    data, raw = load_json(manifest_path)
    release = parse_input(data)
    problems: list[str] = []
    measured: dict[tuple, list[dict]] = {}
    for artifact in release["artifacts"]:
        found, notices = check_archive(archive_dir / artifact["archive"]["name"], artifact)
        problems += found
        measured[(artifact["rid"], artifact["variant"])] = notices
    if problems:
        raise Refused(*problems)
    return build_catalog(release, hashlib.sha256(raw).hexdigest(), measured)


def build_catalog(release: dict, source_sha256: str, notices: dict[tuple, list[dict]]) -> dict:
    version = release["version"]
    tag = tag_for(version)
    artifacts = []
    for a in sorted(release["artifacts"], key=lambda a: (a["rid"], a["variant"])):
        archive = a["archive"]
        artifacts.append({
            "rid": a["rid"],
            "variant": a["variant"],
            "backends": list(a["backends"]),
            "entryLibrary": a["entryLibrary"],
            "nativeAbi": a["nativeAbi"],
            "asset": {
                "name": archive["name"],
                "format": archive["format"],
                "size": archive["size"],
                "sha256": archive["sha256"],
                "url": asset_url(tag, archive["name"]),
            },
            "files": sorted(({k: f[k] for k in ("path", "link", "size", "sha256") if k in f} for f in a["files"]),
                            key=lambda f: f["path"]),
            "notices": sorted(notices[(a["rid"], a["variant"])], key=lambda n: n["path"]),
        })
    return {
        "schema": CATALOG_SCHEMA,
        "component": COMPONENT,
        "version": version,
        "tag": tag,
        "repository": REPOSITORY,
        "tensorSharp": {"version": version, "commit": release["commit"],
                        "nativeSourceCommit": release["nativeSourceCommit"]},
        "ggml": dict(release["ggml"]),
        "source": {"schema": release["schema"], "sha256": source_sha256},
        "artifacts": artifacts,
    }


def build_release(catalog: dict) -> dict:
    """The public release manifest. It locates assets and carries no hashes."""
    return {
        "schema": RELEASE_SCHEMA,
        "component": COMPONENT,
        "version": catalog["version"],
        "tag": catalog["tag"],
        "repository": catalog["repository"],
        "tensorSharp": {"version": catalog["tensorSharp"]["version"], "commit": catalog["tensorSharp"]["commit"]},
        "source": {"sha256": catalog["source"]["sha256"]},
        "assets": [
            {"rid": a["rid"], "variant": a["variant"], "name": a["asset"]["name"],
             "format": a["asset"]["format"], "size": a["asset"]["size"], "url": a["asset"]["url"]}
            for a in catalog["artifacts"]
        ],
    }


def build_notes(catalog: dict) -> str:
    lines = [
        f"GGML driver archives for TensorSharp {catalog['version']}, built from TensorSharp commit "
        f"{catalog['tensorSharp']['commit']} and ggml {catalog['ggml']['version']}.",
        "",
        "The trusted SHA-256 catalog for these archives ships inside the core.inference module package. "
        f"{RELEASE_ASSET} locates the archives and is not a trust source.",
        "",
        "| RID | Variant | Asset | Bytes |",
        "| --- | --- | --- | ---: |",
    ]
    lines += [f"| {a['rid']} | {a['variant']} | {a['asset']['name']} | {a['asset']['size']} |"
              for a in catalog["artifacts"]]
    return "\n".join(lines) + "\n"


def validate_catalog(catalog) -> list[str]:
    """Check a catalog input read back from disk: structure and every derived field."""
    if not isinstance(catalog, dict):
        return ["the catalog input must be a JSON object"]
    problems: list[str] = []
    version = catalog.get("version")
    if catalog.get("schema") != CATALOG_SCHEMA:
        problems.append(f"schema must be {CATALOG_SCHEMA}")
    if catalog.get("component") != COMPONENT or catalog.get("repository") != REPOSITORY:
        problems.append(f"component and repository must be {COMPONENT} and {REPOSITORY}")
    if not isinstance(version, str) or not VERSION.match(version) or catalog.get("tag") != tag_for(version):
        problems.append(f"version {version!r} and tag {catalog.get('tag')!r} do not match")
        return problems
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return problems + ["artifacts must be a non-empty list"]
    keys = []
    for index, a in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(a, dict):
            problems.append(f"{label} must be an object")
            continue
        try:
            native_abi = a.get("nativeAbi")
            if not isinstance(native_abi, str) or not SHA256.fullmatch(native_abi):
                problems.append(f"{label} nativeAbi {native_abi!r} must be 64 lowercase hex digits")
            files = check_file_list(f"{label} files", a.get("files"), problems, notices=False)
            notices = check_file_list(f"{label} notices", a.get("notices"), problems, notices=True)
            overlap = {f["path"].casefold() for f in files} & {n["path"].casefold() for n in notices}
            if overlap:
                problems.append(f"{label} lists {', '.join(sorted(overlap))} in both files and notices")
            entry = a.get("entryLibrary")
            if not any(f["path"] == entry and "link" not in f for f in files):
                problems.append(f"{label} entryLibrary {entry!r} must be a regular file listed in files")
            asset = a["asset"]
            expected = archive_name(version, a["rid"], a["variant"], asset["format"])
            if a["rid"] not in RIDS or not VARIANT.match(a["variant"]) or asset["format"] not in FORMATS:
                problems.append(f"artifacts[{index}] has an unknown rid, variant or format")
            if asset["name"] != expected or asset["url"] != asset_url(catalog["tag"], expected):
                problems.append(f"artifacts[{index}] asset name or url does not follow the layout for {expected}")
            if not is_int(asset["size"]) or asset["size"] <= 0 or not SHA256.match(asset["sha256"]):
                problems.append(f"artifacts[{index}] asset size or sha256 is malformed")
            keys.append((a["rid"], a["variant"]))
        except (KeyError, TypeError):
            problems.append(f"artifacts[{index}] is missing a required field")
    if keys != sorted(set(keys)):
        problems.append("artifacts must be sorted by rid and variant with no repeats")
    return problems


def release_dir_paths(out: Path, version: str) -> tuple[Path, Path, Path]:
    return out / ASSETS_DIR, out / catalog_file_name(version), out / NOTES_FILE


def generate(manifest_path: Path, archive_dir: Path, out: Path) -> dict:
    """Validate, then write out/assets/ (archives and release.json), the catalog input and the notes."""
    catalog = validate(manifest_path, archive_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refused(f"{out} already exists and is not empty; remove it or choose another --out")
    assets, catalog_path, notes_path = release_dir_paths(out, catalog["version"])
    assets.mkdir(parents=True)
    for artifact in catalog["artifacts"]:
        source, destination = archive_dir / artifact["asset"]["name"], assets / artifact["asset"]["name"]
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    (assets / RELEASE_ASSET).write_bytes(dumps(build_release(catalog)))
    catalog_path.write_bytes(dumps(catalog))
    notes_path.write_text(build_notes(catalog), encoding="utf-8")
    problems = verify_release_dir(out)
    if problems:
        raise Refused(*problems)
    return catalog


def find_catalog(out: Path) -> Path:
    found = sorted(out.glob(f"{COMPONENT}-*.catalog.json"))
    if len(found) != 1:
        raise Refused(f"{out} must hold exactly one {COMPONENT}-<version>.catalog.json")
    return found[0]


def load_release_dir(out: Path) -> tuple[dict, bytes]:
    """Load and check a release directory's catalog input. Return it and the release.json bytes it implies."""
    path = find_catalog(out)
    catalog, raw = load_json(path)
    problems = validate_catalog(catalog)
    if problems:
        raise Refused(*problems)
    if raw != dumps(catalog) or path.name != catalog_file_name(catalog["version"]):
        raise Refused(f"{path} is not in the generated byte form or is misnamed")
    return catalog, dumps(build_release(catalog))


def verify_release_dir(out: Path) -> list[str]:
    """Check that out/assets holds exactly release.json and the archives, with the catalog's sizes and hashes."""
    try:
        catalog, release_bytes = load_release_dir(out)
    except Refused as error:
        return list(error.args)
    assets = out / ASSETS_DIR
    expected = {RELEASE_ASSET} | {a["asset"]["name"] for a in catalog["artifacts"]}
    present = {p.name for p in assets.iterdir()} if assets.is_dir() else set()
    problems = [f"{ASSETS_DIR}/ holds unexpected {name}" for name in sorted(present - expected)]
    problems += [f"{ASSETS_DIR}/ is missing {name}" for name in sorted(expected - present)]
    if problems:
        return problems
    if (assets / RELEASE_ASSET).read_bytes() != release_bytes:
        problems.append(f"{RELEASE_ASSET} does not match the catalog input")
    for artifact in catalog["artifacts"]:
        problems += check_archive_bytes(assets / artifact["asset"]["name"], artifact["asset"])
    return problems


# GitHub listing through read-only `gh api` GET requests.

def gh_api_pages(gh: list[str], path: str) -> list:
    items: list = []
    page = 1
    while True:
        command = [*gh, "api", "--method", "GET", f"{path}?per_page={PAGE_SIZE}&page={page}"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Refused(f"cannot run {' '.join(command)}: {error}") from error
        if result.returncode != 0:
            raise Refused(f"{' '.join(command)} failed ({result.returncode}): {result.stderr.strip()}; "
                          "cannot confirm what GitHub holds")
        try:
            batch = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise Refused(f"{' '.join(command)} did not return JSON: {error}") from error
        if not isinstance(batch, list):
            raise Refused(f"{' '.join(command)} did not return a list")
        items += batch
        if len(batch) < PAGE_SIZE:
            return items
        page += 1


def github_listing(gh: list[str], repository: str) -> tuple[set[str], list[dict]]:
    tags = {t.get("name") for t in gh_api_pages(gh, f"repos/{repository}/tags")}
    releases = gh_api_pages(gh, f"repos/{repository}/releases")
    return tags, releases


def check_absent(catalog: dict, gh: list[str], repository: str) -> None:
    """Refuse when the tag, any release for it (draft included) or any archive name already exists."""
    tag = catalog["tag"]
    names = {a["asset"]["name"] for a in catalog["artifacts"]}
    tags, releases = github_listing(gh, repository)
    problems = []
    if tag in tags:
        problems.append(f"tag {tag} already exists in {repository}")
    for release in releases:
        label = f"{'draft ' if release.get('draft') else ''}release {release.get('tag_name')!r}"
        if release.get("tag_name") == tag:
            problems.append(f"{label} already exists in {repository}")
        for asset in release.get("assets") or []:
            if asset.get("name") in names:
                problems.append(f"asset {asset['name']} already exists in {label}")
    if problems:
        raise Refused(*problems, "published versions are immutable; a corrected build needs a new TensorSharp version")


def verify_listing(out: Path, gh: list[str], repository: str) -> tuple[bool, list[str]]:
    """Check the tag's release on GitHub holds exactly the release directory's assets, fully uploaded.

    Return whether the release is a draft, and notes that do not fail the check.
    """
    catalog, release_bytes = load_release_dir(out)
    tag = catalog["tag"]
    tags, releases = github_listing(gh, repository)
    matching = [r for r in releases if r.get("tag_name") == tag]
    if len(matching) != 1:
        raise Refused(f"expected one release for tag {tag} in {repository}, found {len(matching)}")
    release = matching[0]
    draft = bool(release.get("draft"))
    expected = {a["asset"]["name"]: (a["asset"]["size"], a["asset"]["sha256"]) for a in catalog["artifacts"]}
    expected[RELEASE_ASSET] = (len(release_bytes), hashlib.sha256(release_bytes).hexdigest())
    listed = {a.get("name"): a for a in release.get("assets") or []}
    problems = [f"release {tag} holds unexpected asset {n}" for n in sorted(set(listed) - set(expected))]
    problems += [f"release {tag} is missing asset {n}" for n in sorted(set(expected) - set(listed))]
    notes = []
    for name in sorted(set(expected) & set(listed)):
        asset, (size, digest) = listed[name], expected[name]
        if asset.get("state") != "uploaded":
            problems.append(f"{name} state is {asset.get('state')!r}, not 'uploaded'")
        if asset.get("size") != size:
            problems.append(f"{name} size is {asset.get('size')}, expected {size}")
        if asset.get("digest") is None:
            notes.append(f"{name} has no digest in the GitHub listing; verify-published checks its hash")
        elif asset.get("digest") != f"sha256:{digest}":
            problems.append(f"{name} digest is {asset.get('digest')}, expected sha256:{digest}")
    if not draft and tag not in tags:
        problems.append(f"release {tag} is published but tag {tag} is missing")
    if problems:
        raise Refused(*problems)
    return draft, notes


# Download over HTTP with resume.

class Transfer:
    """Counts what a download needed. A resumed request carries a Range header."""

    def __init__(self) -> None:
        self.requests = 0
        self.resumed = 0
        self.restarted = 0


def http_open(url: str, headers: dict | None = None, method: str = "GET"):
    request = urllib.request.Request(url, method=method, headers={
        "User-Agent": "darkspyre-releases-verify/1",
        "Accept": "application/octet-stream",
        **(headers or {}),
    })
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        return error


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    match = re.match(r"^bytes (\d+)-(\d+)/(\d+)$", value or "")
    return tuple(int(g) for g in match.groups()) if match else None


def download(url: str, destination: Path, size: int, attempts: int = 6) -> Transfer:
    """Download url to destination, resuming a partial file with Range and If-Range.

    Each attempt requests the github.com URL again, so an expired signed redirect is replaced by a
    fresh one. A 206 answer is appended, a 200 answer restarts the file and a 416 answer discards it.
    """
    transfer = Transfer()
    etag: str | None = None
    failures: list[str] = []
    for _ in range(attempts):
        have = destination.stat().st_size if destination.exists() else 0
        if have > size:
            destination.unlink()
            have, etag = 0, None
        if have == size:
            return transfer
        headers = {}
        if have:
            headers["Range"] = f"bytes={have}-"
            if etag:
                headers["If-Range"] = etag
        transfer.requests += 1
        try:
            with http_open(url, headers) as response:
                if response.status == 206 and have:
                    span = parse_content_range(response.headers.get("Content-Range"))
                    if not span or span[0] != have or span[2] != size:
                        failures.append(f"answered Content-Range {response.headers.get('Content-Range')} for {have}-")
                        destination.unlink()
                        etag = None
                        continue
                    transfer.resumed += 1
                    mode = "ab"
                elif response.status == 200:
                    if have:
                        transfer.restarted += 1
                    mode = "wb"
                elif response.status == 416:
                    failures.append("answered 416 to a resume")
                    destination.unlink(missing_ok=True)
                    etag = None
                    continue
                else:
                    failures.append(f"answered {response.status}")
                    continue
                if response.headers.get("Content-Encoding"):
                    raise Refused(f"{url} answered Content-Encoding {response.headers.get('Content-Encoding')}; "
                                  "archives must be served as stored")
                etag = response.headers.get("ETag") or etag
                with destination.open(mode) as handle:
                    while block := response.read(CHUNK):
                        handle.write(block)
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout, TimeoutError) as error:
            failures.append(f"transfer failed: {error!r}")
    have = destination.stat().st_size if destination.exists() else 0
    if have != size:
        raise Refused(f"{url} did not download completely ({have} of {size} bytes): {'; '.join(failures)}")
    return transfer


def probe_range(url: str, size: int) -> str | None:
    """Return why url cannot resume, or None when a last-byte range answers 206 correctly."""
    try:
        with http_open(url, {"Range": f"bytes={size - 1}-"}) as response:
            body = response.read()
            span = response.headers.get("Content-Range")
            if response.status != 206 or span != f"bytes {size - 1}-{size - 1}/{size}" or len(body) != 1:
                return f"a range request for the last byte answered {response.status} with Content-Range {span}"
            if response.headers.get("Accept-Ranges") not in (None, "bytes"):
                return f"Accept-Ranges is {response.headers.get('Accept-Ranges')}"
    except (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout, TimeoutError) as error:
        return f"range request failed: {error!r}"
    return None


def verify_published(out: Path, base_url: str, work: Path, keep: bool = False) -> dict[str, Transfer]:
    """Download release.json and every archive through the public URLs and check them against the catalog."""
    catalog, release_bytes = load_release_dir(out)

    def public(url: str) -> str:
        return base_url.rstrip("/") + url[len(GITHUB):]

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    problems: list[str] = []
    transfers: dict[str, Transfer] = {}
    release_path = work / RELEASE_ASSET
    try:
        transfers[RELEASE_ASSET] = download(public(asset_url(catalog["tag"], RELEASE_ASSET)), release_path, len(release_bytes))
        if release_path.read_bytes() != release_bytes:
            problems.append(f"the published {RELEASE_ASSET} does not match the catalog input")
    except Refused as error:
        problems += error.args
    for artifact in catalog["artifacts"]:
        asset = artifact["asset"]
        url, path = public(asset["url"]), work / asset["name"]
        problem = probe_range(url, asset["size"])
        if problem:
            problems.append(f"{asset['name']}: {problem}")
        try:
            transfers[asset["name"]] = download(url, path, asset["size"])
        except Refused as error:
            problems += error.args
            continue
        found, _ = check_archive(path, {"archive": asset, "files": artifact["files"], "notices": artifact["notices"]})
        problems += found
        if not keep:
            path.unlink()
    if not keep:
        shutil.rmtree(work)
    if problems:
        raise Refused(*problems)
    return transfers


# Command line.

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gh", default="gh", help="gh command line, split like a shell would (default: gh)")
    parser.add_argument("--repository", default=REPOSITORY, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    cmd = commands.add_parser("validate", help="check an artifact manifest against its local archives")
    cmd.add_argument("manifest", type=Path)
    cmd.add_argument("archive_dir", type=Path)

    cmd = commands.add_parser("generate", help="validate and write a release directory")
    cmd.add_argument("manifest", type=Path)
    cmd.add_argument("archive_dir", type=Path)
    cmd.add_argument("--out", type=Path, help="release directory (default: tmp/releases/<tag>)")

    cmd = commands.add_parser("check-absent", help="refuse when the tag, its release or an archive name exists")
    cmd.add_argument("release_dir", type=Path)

    cmd = commands.add_parser("verify-listing", help="check the tag's GitHub asset listing")
    cmd.add_argument("release_dir", type=Path)

    cmd = commands.add_parser("verify-published", help="download a published release and check every asset")
    cmd.add_argument("release_dir", type=Path)
    cmd.add_argument("--base-url", default=GITHUB, help=argparse.SUPPRESS)
    cmd.add_argument("--work", type=Path, help="download directory (default: <release_dir>/download)")
    cmd.add_argument("--keep", action="store_true", help="keep the downloaded files")

    args = parser.parse_args(argv)
    gh = shlex.split(args.gh)
    try:
        if args.command == "validate":
            catalog = validate(args.manifest, args.archive_dir)
            print(f"{COMPONENT} {catalog['version']}: {len(catalog['artifacts'])} archives are valid")
        elif args.command == "generate":
            data, _ = load_json(args.manifest)
            version = parse_input(data)["version"]
            out = args.out or ROOT / "tmp" / "releases" / tag_for(version)
            catalog = generate(args.manifest, args.archive_dir, out)
            assets, catalog_path, _ = release_dir_paths(out, catalog["version"])
            print(f"wrote {out}")
            print(f"tag {catalog['tag']}; upload every file in {assets}")
            print(f"catalog input {catalog_path} sha256 {sha256_file(catalog_path)}")
        elif args.command == "check-absent":
            catalog, _ = load_release_dir(args.release_dir)
            check_absent(catalog, gh, args.repository)
            print(f"tag {catalog['tag']} and its assets do not exist in {args.repository}")
        elif args.command == "verify-listing":
            draft, notes = verify_listing(args.release_dir, gh, args.repository)
            for note in notes:
                print(f"note: {note}")
            print(f"the {'draft' if draft else 'published'} release holds exactly the generated assets")
        elif args.command == "verify-published":
            work = args.work or args.release_dir / "download"
            transfers = verify_published(args.release_dir, args.base_url, work, args.keep)
            resumed = sum(t.resumed for t in transfers.values())
            print(f"{len(transfers)} assets downloaded and verified from {args.base_url} ({resumed} resumed requests)")
    except Refused as error:
        for problem in error.args:
            print(f"refused: {problem}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
