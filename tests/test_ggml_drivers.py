"""Tests for tools/ggml_drivers.py against local fixture archives, a stub gh and a local HTTP server.

Run from the repository root:

    python3 -m unittest discover -s tests -v

Fixtures are written under tmp/tests/ and removed afterwards. No test reaches the public network.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import ggml_drivers as drivers  # noqa: E402
from github_server import GitHubReleaseServer  # noqa: E402

WORK = ROOT / "tmp" / "tests"
STUB_GH = [sys.executable, str(ROOT / "tests" / "stub_gh.py")]
VERSION = "2.9.0"
TAG = f"ggml-drivers-v{VERSION}"
PACKAGE_COMMIT = "9e36cdd3e2b05612191451180e64394831d63d20"
NATIVE_COMMIT = "513de7b98fd7d0e52b19e1c42a6ebbe5bb5c98ff"
NOTICE = b"GGML is MIT licensed.\n"


def blob(seed: str, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_tar(path: Path, files: dict[str, bytes], links: dict[str, str] | None = None,
              dirs: list[str] | None = None, hard_links: dict[str, str] | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in dirs or []:
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            archive.addfile(info)
        for name, target in (hard_links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.LNKTYPE, target
            archive.addfile(info)


def write_zip(path: Path, files: dict[str, bytes], links: dict[str, str] | None = None) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o120777 << 16
            archive.writestr(info, target)


def file_entry(path: str, data: bytes) -> dict:
    return {"path": path, "size": len(data), "sha256": sha(data)}


class Fixture:
    """A directory holding fixture archives and a TensorSharp artifact manifest in the DSA-98 draft shape."""

    def __init__(self, directory: Path, version: str = VERSION):
        self.dir = directory
        self.dir.mkdir(parents=True)
        self.version = version
        self.data: dict = {"schema": "tensorsharp-native-artifacts/draft-1", "packages": [], "artifacts": []}

        so = blob("linux-cpu", 300_000)
        self.add("linux-x64", "cpu", "tar.gz", ["cpu"], "libGgmlOps.so",
                 files={"lib/libGgmlOps.so.1": so, "THIRD-PARTY-NOTICES.txt": NOTICE},
                 file_list=[{"path": "lib/libGgmlOps.so", "link": "libGgmlOps.so.1"}, file_entry("lib/libGgmlOps.so.1", so)],
                 notices=["THIRD-PARTY-NOTICES.txt"], links={"lib/libGgmlOps.so": "libGgmlOps.so.1"}, dirs=["lib"],
                 entry_path="lib/libGgmlOps.so.1")
        cuda, cudart = blob("cuda", 500_000), blob("cudart", 200_000)
        self.add("linux-x64", "cuda13", "tar.gz", ["cpu", "cuda"], "libGgmlOps.so",
                 files={"libGgmlOps.so": cuda, "libcudart.so.13": cudart,
                        "THIRD-PARTY-NOTICES.txt": NOTICE, "licenses/cuda.txt": b"CUDA EULA\n"},
                 notices=[file_entry("THIRD-PARTY-NOTICES.txt", NOTICE), "licenses/cuda.txt"])
        dll, vc = blob("win", 250_000), blob("vc", 1000)
        self.add("win-x64", "cpu", "zip", ["cpu"], "GgmlOps.dll",
                 files={"GgmlOps.dll": dll, "vcruntime140.dll": vc, "THIRD-PARTY-NOTICES.txt": NOTICE},
                 notices=["THIRD-PARTY-NOTICES.txt"])
        # A baseline artifact delivered in a package: no archive, so it is not hosted.
        self.data["artifacts"].append({
            "rid": "osx-arm64", "variant": "metal", "backends": ["cpu", "metal"],
            "delivery": "baseline-package", "archive": None, "notices": [],
            "files": [file_entry("libGgmlOps.dylib", b"dylib")], "entryLibrary": "libGgmlOps.dylib",
        })
        self.write()

    def add(self, rid, variant, fmt, backends, entry, files, notices, file_list=None, links=None, dirs=None,
            entry_path=None) -> None:
        name = f"ggml-{self.version}-{rid}-{variant}.{fmt}"
        path = self.dir / name
        notice_paths = {n if isinstance(n, str) else n["path"] for n in notices}
        if fmt == "tar.gz":
            write_tar(path, files, links=links, dirs=dirs)
        else:
            write_zip(path, files)
        payload = file_list or [file_entry(p, d) for p, d in files.items() if p not in notice_paths]
        artifact = {
            "rid": rid, "variant": variant, "backends": backends, "delivery": "variant-archive",
            "tensorSharp": {"packageVersion": self.version, "packageCommit": PACKAGE_COMMIT,
                            "nativeSourceCommit": NATIVE_COMMIT},
            "entryLibrary": entry_path or entry,
            "binaryIdentity": {"tsggmlExports": 312}, "build": {"host": "fixture"},
            "files": payload, "totalSize": sum(f.get("size", 0) for f in payload),
            "requires": {"os-package": ["libgomp.so.1"]},
            "archive": {"name": name, "format": fmt, "size": path.stat().st_size, "sha256": sha(path.read_bytes())},
            "notices": notices,
        }
        self.data["artifacts"].append(artifact)

    @property
    def path(self) -> Path:
        return self.dir / "tensorsharp-native-artifacts.json"

    def artifact(self, rid: str, variant: str) -> dict:
        return next(a for a in self.data["artifacts"] if (a["rid"], a["variant"]) == (rid, variant))

    def write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def rewrite(self, rid: str, variant: str, writer, *args, **kwargs) -> None:
        """Rewrite one archive with writer(path, ...) and record its new size and hash in the manifest."""
        artifact = self.artifact(rid, variant)
        path = self.dir / artifact["archive"]["name"]
        path.unlink()
        writer(path, *args, **kwargs)
        artifact["archive"].update(size=path.stat().st_size, sha256=sha(path.read_bytes()))
        self.write()


class ToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = WORK / self.id().rsplit(".", 1)[-1]
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        self.fixture = Fixture(self.work / "input")
        self.out = self.work / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)
        if WORK.exists() and not any(WORK.iterdir()):
            WORK.rmdir()

    def refused(self, call, *fragments: str) -> list[str]:
        with self.assertRaises(drivers.Refused) as caught:
            call()
        problems = list(caught.exception.args)
        for fragment in fragments:
            self.assertTrue(any(fragment in p for p in problems), f"{fragment!r} not in {problems}")
        return problems

    def validate(self) -> dict:
        return drivers.validate(self.fixture.path, self.fixture.dir)

    def generate(self) -> dict:
        return drivers.generate(self.fixture.path, self.fixture.dir, self.out)


class GenerateTests(ToolTestCase):
    def test_generates_release_directory(self) -> None:
        catalog = self.generate()
        assets = sorted(p.name for p in (self.out / "assets").iterdir())
        self.assertEqual(assets, [f"ggml-{VERSION}-linux-x64-cpu.tar.gz", f"ggml-{VERSION}-linux-x64-cuda13.tar.gz",
                                  f"ggml-{VERSION}-win-x64-cpu.zip", "release.json"])
        self.assertTrue((self.out / f"ggml-drivers-{VERSION}.catalog.json").is_file())
        self.assertTrue((self.out / "notes.md").is_file())
        self.assertEqual(catalog["tag"], TAG)
        self.assertEqual([(a["rid"], a["variant"]) for a in catalog["artifacts"]],
                         [("linux-x64", "cpu"), ("linux-x64", "cuda13"), ("win-x64", "cpu")])
        self.assertEqual(drivers.verify_release_dir(self.out), [])

    def test_locator_urls_and_release_json(self) -> None:
        catalog = self.generate()
        release = json.loads((self.out / "assets" / "release.json").read_bytes())
        name = f"ggml-{VERSION}-linux-x64-cuda13.tar.gz"
        url = f"https://github.com/DarkspyreStudios/darkspyre-releases/releases/download/{TAG}/{name}"
        self.assertEqual(catalog["artifacts"][1]["asset"]["url"], url)
        self.assertEqual(release["assets"][1]["url"], url)
        self.assertEqual(release["tag"], TAG)
        self.assertNotIn("sha256", json.dumps(release["assets"]))
        self.assertEqual(release["source"]["sha256"], catalog["source"]["sha256"])

    def test_licence_listed_as_file_and_notice_is_one_notice(self) -> None:
        cuda = next(a for a in self.fixture.data["artifacts"] if "licenses/cuda.txt" in a["notices"])
        licence = file_entry("licenses/cuda.txt", b"CUDA EULA\n")
        cuda["notices"] = [licence if n == "licenses/cuda.txt" else n for n in cuda["notices"]]
        cuda["files"].append(dict(licence))
        cuda["totalSize"] = sum(f.get("size", 0) for f in cuda["files"])
        self.fixture.write()
        catalog = self.generate()
        listed = next(a for a in catalog["artifacts"] if any(n["path"] == "licenses/cuda.txt" for n in a["notices"]))
        self.assertNotIn("licenses/cuda.txt", [f["path"] for f in listed["files"]])

    def test_catalog_carries_measured_notices_and_identity(self) -> None:
        catalog = self.generate()
        cuda = catalog["artifacts"][1]
        self.assertEqual(cuda["notices"], [file_entry("THIRD-PARTY-NOTICES.txt", NOTICE),
                                           file_entry("licenses/cuda.txt", b"CUDA EULA\n")])
        self.assertEqual(cuda["backends"], ["cpu", "cuda"])
        self.assertEqual(catalog["tensorSharp"], {"version": VERSION, "commit": PACKAGE_COMMIT,
                                                  "nativeSourceCommit": NATIVE_COMMIT})
        self.assertNotIn("ggml", catalog)
        self.assertEqual(catalog["source"]["sha256"], sha(self.fixture.path.read_bytes()))
        self.assertIn({"path": "lib/libGgmlOps.so", "link": "libGgmlOps.so.1"}, catalog["artifacts"][0]["files"])

    def test_output_is_deterministic(self) -> None:
        self.generate()
        first = {p.name: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        shutil.rmtree(self.out)
        self.fixture.data["artifacts"].reverse()
        self.fixture.write()
        self.generate()
        second = {p.name: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(first.keys(), second.keys())
        catalog_name = f"ggml-drivers-{VERSION}.catalog.json"
        strip = lambda raw: {**json.loads(raw), "source": None}  # noqa: E731
        self.assertEqual(strip(first[catalog_name]), strip(second[catalog_name]))

    def test_refuses_existing_output(self) -> None:
        self.out.mkdir()
        (self.out / "keep").write_text("x")
        self.refused(self.generate, "already exists")

    def test_verify_release_dir_detects_tampering(self) -> None:
        self.generate()
        assets = self.out / "assets"
        (assets / "extra.bin").write_bytes(b"x")
        self.assertTrue(any("unexpected extra.bin" in p for p in drivers.verify_release_dir(self.out)))
        (assets / "extra.bin").unlink()
        release = assets / "release.json"
        release.write_bytes(release.read_bytes().replace(b"linux-x64", b"linux-arm64"))
        self.assertTrue(any("does not match" in p for p in drivers.verify_release_dir(self.out)))

    def test_verify_release_dir_detects_changed_archive(self) -> None:
        self.generate()
        archive = self.out / "assets" / f"ggml-{VERSION}-win-x64-cpu.zip"
        data = bytearray(archive.read_bytes())
        data[10] ^= 1
        archive.unlink()
        archive.write_bytes(bytes(data))
        self.assertTrue(any("sha256 is" in p for p in drivers.verify_release_dir(self.out)))

    def test_cli_generate_and_validate(self) -> None:
        self.assertEqual(run_cli(*["validate", str(self.fixture.path), str(self.fixture.dir)]), 0)
        self.assertEqual(run_cli(*["generate", str(self.fixture.path), str(self.fixture.dir),
                                       "--out", str(self.out)]), 0)
        self.fixture.artifact("win-x64", "cpu")["archive"]["size"] += 1
        self.fixture.write()
        self.assertEqual(run_cli(*["validate", str(self.fixture.path), str(self.fixture.dir)]), 1)


class InputTests(ToolTestCase):
    def mutate(self, change, *fragments: str) -> None:
        change(self.fixture.data)
        self.fixture.write()
        self.refused(self.validate, *fragments)

    def test_schema(self) -> None:
        self.mutate(lambda d: d.update(schema="other/1"), "schema")

    def test_inconsistent_build_identity(self) -> None:
        self.mutate(lambda d: d["artifacts"][1]["tensorSharp"].update(packageVersion="2.9.1"), "disagree")

    def test_short_commit(self) -> None:
        self.mutate(lambda d: d["artifacts"][0]["tensorSharp"].update(packageCommit="9e36cdd3"), "packageCommit")

    def test_duplicate_rid_variant(self) -> None:
        self.mutate(lambda d: d["artifacts"].append(copy.deepcopy(d["artifacts"][2])), "repeats rid win-x64")

    def test_unknown_rid(self) -> None:
        self.mutate(lambda d: d["artifacts"][2].update(rid="win-x86"), "rid 'win-x86'")

    def test_archive_name_must_follow_layout(self) -> None:
        self.mutate(lambda d: d["artifacts"][2]["archive"].update(name="drivers.zip"), "archive.name")

    def test_archive_format(self) -> None:
        self.mutate(lambda d: d["artifacts"][2]["archive"].update(format="7z"), "archive.format")

    def test_variant_archive_without_archive(self) -> None:
        self.mutate(lambda d: d["artifacts"][2].update(archive=None), "no archive object")

    def test_archive_with_other_delivery(self) -> None:
        self.mutate(lambda d: d["artifacts"][2].update(delivery="baseline-package"), "delivery must be")

    def test_nothing_hosted(self) -> None:
        self.mutate(lambda d: d.update(artifacts=d["artifacts"][3:]), "nothing to host")

    def test_entry_library_must_be_a_payload_file(self) -> None:
        self.mutate(lambda d: d["artifacts"][0].update(entryLibrary="lib/libGgmlOps.so"), "entryLibrary")

    def test_file_in_files_and_notices(self) -> None:
        self.mutate(lambda d: d["artifacts"][2]["notices"].append("GgmlOps.dll"), "both files and notices")

    def test_notices_required(self) -> None:
        self.mutate(lambda d: d["artifacts"][2].update(notices=[]), "notices must be a non-empty list")

    def test_total_size(self) -> None:
        self.mutate(lambda d: d["artifacts"][2].update(totalSize=1), "totalSize")

    def test_unsafe_paths(self) -> None:
        for bad in ["../evil.dll", "/abs.dll", "C:/abs.dll", "a\\b.dll", "a//b.dll", "con.dll", "a:b.dll", "dot."]:
            with self.subTest(bad=bad):
                self.fixture = Fixture(self.work / f"input-{sha(bad.encode())[:8]}")
                self.mutate(lambda d: d["artifacts"][2]["files"].append({"path": bad, "size": 1, "sha256": "0" * 64}),
                            "path")

    def test_payload_file_needs_size_and_hash(self) -> None:
        self.mutate(lambda d: d["artifacts"][2]["files"].append({"path": "x.dll"}), "size")

    def test_preload_must_be_a_listed_top_level_file_other_than_the_entry(self) -> None:
        for preload in (["libGgmlOps.so"], ["missing.so"], ["libcudart.so.13", "libcudart.so.13"]):
            with self.subTest(preload=preload):
                self.mutate(lambda d: d["artifacts"][1].update(preload=preload), "preload")

    def test_note_is_one_short_line(self) -> None:
        for note in ("x" * 201, "two\nlines"):
            with self.subTest(note=note):
                self.mutate(lambda d: d["artifacts"][1].update(note=note), "note")


class ArchiveTests(ToolTestCase):
    def test_size_and_hash(self) -> None:
        artifact = self.fixture.artifact("win-x64", "cpu")
        artifact["archive"]["size"] += 1
        self.fixture.write()
        self.refused(self.validate, "size is")
        artifact["archive"]["size"] -= 1
        artifact["archive"]["sha256"] = "0" * 64
        self.fixture.write()
        self.refused(self.validate, "sha256 is")

    def test_missing_archive(self) -> None:
        (self.fixture.dir / self.fixture.artifact("win-x64", "cpu")["archive"]["name"]).unlink()
        self.refused(self.validate, "is missing or is not a regular file")

    def test_missing_payload_file(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE})
        self.refused(self.validate, "payload vcruntime140.dll is missing")

    def test_missing_notice(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000)})
        self.refused(self.validate, "notice THIRD-PARTY-NOTICES.txt is missing")

    def test_unlisted_file(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE, "extra.exe": b"MZ"})
        self.refused(self.validate, "file extra.exe is not listed")

    def test_payload_file_content_differs(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("other", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE})
        self.refused(self.validate, "payload GgmlOps.dll is 250000 bytes with sha256")

    def test_notice_content_differs_when_hashed(self) -> None:
        self.fixture.rewrite("linux-x64", "cuda13", write_tar, {
            "libGgmlOps.so": blob("cuda", 500_000), "libcudart.so.13": blob("cudart", 200_000),
            "THIRD-PARTY-NOTICES.txt": b"changed\n", "licenses/cuda.txt": b"CUDA EULA\n"})
        self.refused(self.validate, "notice THIRD-PARTY-NOTICES.txt is 8 bytes")

    def test_directory_without_listed_file(self) -> None:
        self.fixture.rewrite("linux-x64", "cuda13", write_tar, {
            "libGgmlOps.so": blob("cuda", 500_000), "libcudart.so.13": blob("cudart", 200_000),
            "THIRD-PARTY-NOTICES.txt": NOTICE, "licenses/cuda.txt": b"CUDA EULA\n"}, dirs=["empty"])
        self.refused(self.validate, "directory empty holds no listed file")

    def test_case_collision(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE, "GGMLOPS.dll": b"x"})
        self.refused(self.validate, "differ only by case")

    def test_unsafe_entry(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE, "../evil.dll": b"x"})
        self.refused(self.validate, "entry '../evil.dll'")

    def test_hard_link(self) -> None:
        self.fixture.rewrite("linux-x64", "cuda13", write_tar, {
            "libGgmlOps.so": blob("cuda", 500_000), "libcudart.so.13": blob("cudart", 200_000),
            "THIRD-PARTY-NOTICES.txt": NOTICE, "licenses/cuda.txt": b"CUDA EULA\n"},
            hard_links={"copy.so": "libGgmlOps.so"})
        self.refused(self.validate, "hard link or special file")

    def test_zip_symlink(self) -> None:
        self.fixture.rewrite("win-x64", "cpu", write_zip, {"GgmlOps.dll": blob("win", 250_000),
                                                           "vcruntime140.dll": blob("vc", 1000),
                                                           "THIRD-PARTY-NOTICES.txt": NOTICE},
                             links={"link.dll": "GgmlOps.dll"})
        self.refused(self.validate, "zip archives must not hold links")

    def test_escaping_symlink(self) -> None:
        so = blob("linux-cpu", 300_000)
        self.fixture.artifact("linux-x64", "cpu")["files"][0]["link"] = "../../etc/passwd"
        self.fixture.rewrite("linux-x64", "cpu", write_tar, {"lib/libGgmlOps.so.1": so, "THIRD-PARTY-NOTICES.txt": NOTICE},
                             links={"lib/libGgmlOps.so": "../../etc/passwd"}, dirs=["lib"])
        self.refused(self.validate, "escapes the archive")

    def test_symlink_target_must_match_manifest(self) -> None:
        so = blob("linux-cpu", 300_000)
        self.fixture.rewrite("linux-x64", "cpu", write_tar,
                             {"lib/libGgmlOps.so.1": so, "lib/other.so": so, "THIRD-PARTY-NOTICES.txt": NOTICE},
                             links={"lib/libGgmlOps.so": "other.so"}, dirs=["lib"])
        self.refused(self.validate, "the manifest says 'libGgmlOps.so.1'")

    def test_unlisted_symlink(self) -> None:
        so = blob("linux-cpu", 300_000)
        self.fixture.rewrite("linux-x64", "cpu", write_tar, {"lib/libGgmlOps.so.1": so, "THIRD-PARTY-NOTICES.txt": NOTICE},
                             links={"lib/libGgmlOps.so": "libGgmlOps.so.1", "lib/x.so": "libGgmlOps.so.1"}, dirs=["lib"])
        self.refused(self.validate, "symbolic link lib/x.so is not listed")

    def test_corrupt_archive(self) -> None:
        self.fixture.rewrite("linux-x64", "cuda13", lambda p: p.write_bytes(b"not a gzip stream"))
        self.refused(self.validate, "cannot be read as tar.gz")


UPSTREAM_VERSION = "2026.10.11"
UPSTREAM_COMMIT = "f20fade4aa1feb67b5e2d43dad584c2af6865b1e"
TS_GGML = ["TensorSharp-LICENSE.txt", "ggml-LICENSE.txt"]
TS_GGML_NVIDIA = [*TS_GGML, "NVIDIA-cudart.txt", "NVIDIA-cuBLAS.txt"]
# The variant table of TechDesign DSA-1112 5.9: (rid, variant) -> (upstream archive suffix, files in archive order,
# preload, notices).
UPSTREAM_ROWS = {
    ("osx-arm64", "metal"): ("osx-arm64.tar.gz", ["libGgmlOps.dylib"], [], TS_GGML),
    ("osx-arm64", "mlx"): ("osx-arm64.tar.gz", ["libmlxc.dylib", "libmlx.dylib", "mlx.metallib"], [],
                          ["TensorSharp-LICENSE.txt", "mlx-LICENSE.txt", "mlx-c-LICENSE.txt"]),
    ("linux-x64", "cpu"): ("linux-x64-cpu.tar.gz", ["libGgmlOps.so"], [], TS_GGML),
    ("linux-x64", "cuda12"): ("linux-x64-cuda.tar.gz",
                             ["libGgmlOps.so", "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"],
                             ["libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"], TS_GGML_NVIDIA),
    ("linux-arm64", "cuda13-gb10"): ("linux-arm64-cuda13-GB10.tar.gz",
                                    ["libGgmlOps.so", "libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"],
                                    ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"], TS_GGML_NVIDIA),
    ("win-x64", "cpu"): ("win-x64-cpu.zip", ["GgmlOps.dll"], [], TS_GGML),
    ("win-x64", "cuda12"): ("win-x64-cuda.zip", ["GgmlOps.dll", "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"],
                           ["cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"], TS_GGML_NVIDIA),
}


def read_archive(path: Path) -> dict[str, bytes]:
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            return {info.filename: archive.read(info) for info in archive.infolist()}
    with tarfile.open(path, "r:gz") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()}


class ExtractUpstreamTests(ToolTestCase):
    """T21: extract-upstream over synthetic upstream CLI archives that hold the driver files among other files."""

    def setUp(self) -> None:
        super().setUp()
        self.archives = self.work / "upstream"
        self.archives.mkdir()
        self.payload: dict[str, dict[str, bytes]] = {}
        for upstream, files, _, _ in UPSTREAM_ROWS.values():
            held = self.payload.setdefault(upstream, {"TensorSharp.Backends.MLX.dll": blob("managed", 1000),
                                                      "libjaccl.dylib": blob("jaccl", 1000)})
            held.update({name: blob(f"{upstream}:{name}", 3000) for name in files})
        for upstream, files in self.payload.items():
            self.write_upstream(upstream, files)

    def write_upstream(self, suffix: str, files: dict[str, bytes]) -> None:
        path = self.archives / f"tensorsharp-cli-{UPSTREAM_VERSION}-{suffix}"
        path.unlink(missing_ok=True)
        if suffix.endswith(".zip"):
            write_zip(path, files)
        else:
            write_tar(path, {f"./{name}": data for name, data in files.items()})
        if "GB10" in suffix:
            sums = self.archives / f"tensorsharp-{UPSTREAM_VERSION}-linux-arm64-cuda13-GB10-SHA256SUMS"
            sums.write_text(f"{sha(path.read_bytes())}  {path.name}\n"
                            f"{'0' * 64}  tensorsharp-server-{UPSTREAM_VERSION}-linux-arm64-cuda13-GB10.tar.gz\n")

    def extract(self) -> dict:
        return drivers.extract_upstream(UPSTREAM_VERSION, self.archives, UPSTREAM_COMMIT, self.out)

    def test_writes_one_archive_per_variant_row_with_exactly_its_files_and_notices(self) -> None:
        manifest = self.extract()
        names = {key: f"ggml-{UPSTREAM_VERSION}-{key[0]}-{key[1]}.{'zip' if row[0].endswith('.zip') else 'tar.gz'}"
                 for key, row in UPSTREAM_ROWS.items()}
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted([*names.values(), "tensorsharp-native-artifacts.json"]))
        self.assertEqual([(a["rid"], a["variant"]) for a in manifest["artifacts"]], list(UPSTREAM_ROWS))
        for artifact in manifest["artifacts"]:
            key = (artifact["rid"], artifact["variant"])
            upstream, files, preload, notices = UPSTREAM_ROWS[key]
            with self.subTest(key=key):
                expected = {name: self.payload[upstream][name] for name in files}
                expected.update({f"licenses/{n}": (ROOT / "tools" / "notices" / n).read_bytes() for n in notices})
                self.assertEqual(read_archive(self.out / names[key]), expected)
                self.assertEqual((artifact["entryLibrary"], artifact["preload"]), (files[0], preload))

        catalog = drivers.generate(self.out / "tensorsharp-native-artifacts.json", self.out, self.work / "release")
        rows = {(a["rid"], a["variant"]): a for a in catalog["artifacts"]}
        self.assertEqual(sorted(rows), sorted(UPSTREAM_ROWS))
        self.assertEqual({key: a["preload"] for key, a in rows.items()},
                         {key: row[2] for key, row in UPSTREAM_ROWS.items()})
        self.assertEqual({key: a.get("note") for key, a in rows.items() if "note" in a},
                         {("linux-arm64", "cuda13-gb10"): "Built for NVIDIA GB10 (Grace Blackwell) GPUs only."})
        self.assertEqual(catalog["tensorSharp"], {"version": UPSTREAM_VERSION, "commit": UPSTREAM_COMMIT,
                                                  "nativeSourceCommit": UPSTREAM_COMMIT})

    def test_refuses_an_upstream_archive_missing_a_listed_file(self) -> None:
        files = dict(self.payload["linux-x64-cuda.tar.gz"])
        del files["libcublasLt.so.12"]
        self.write_upstream("linux-x64-cuda.tar.gz", files)
        self.refused(self.extract, "has no libcublasLt.so.12")
        self.assertFalse(self.out.exists())

    def test_refuses_a_gb10_archive_that_differs_from_its_sha256sums(self) -> None:
        sums = self.archives / f"tensorsharp-{UPSTREAM_VERSION}-linux-arm64-cuda13-GB10-SHA256SUMS"
        sums.write_text(f"{'1' * 64}  tensorsharp-cli-{UPSTREAM_VERSION}-linux-arm64-cuda13-GB10.tar.gz\n")
        self.refused(self.extract, "SHA256SUMS says")
        self.assertFalse(self.out.exists())


def run_cli(*args: str) -> int:
    """Run the command line with its output captured."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return drivers.main(list(args))


def stub_state(path: Path, tags=(), releases=(), fail=None) -> None:
    path.write_text(json.dumps({"tags": list(tags), "releases": list(releases), "fail": fail}))


class GitHubListingTests(ToolTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.catalog = self.generate()
        self.state = self.work / "gh-state.json"
        self.log = self.work / "gh-log.jsonl"
        self.saved_env = {k: os.environ.get(k) for k in ("STUB_GH_STATE", "STUB_GH_LOG")}
        os.environ["STUB_GH_STATE"], os.environ["STUB_GH_LOG"] = str(self.state), str(self.log)
        stub_state(self.state)

    def tearDown(self) -> None:
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    def check_absent(self) -> None:
        drivers.check_absent(self.catalog, STUB_GH, drivers.REPOSITORY)

    def uploaded(self, draft: bool = True, digests: bool = True) -> dict:
        assets = []
        for path in sorted((self.out / "assets").iterdir()):
            data = path.read_bytes()
            asset = {"name": path.name, "size": len(data), "state": "uploaded"}
            if digests:
                asset["digest"] = f"sha256:{sha(data)}"
            assets.append(asset)
        return {"tag_name": TAG, "draft": draft, "assets": assets}

    def test_absent_on_empty_repository(self) -> None:
        self.check_absent()
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertTrue(calls)
        self.assertTrue(all(call[:3] == ["api", "--method", "GET"] for call in calls))

    def test_other_versions_do_not_collide(self) -> None:
        stub_state(self.state, tags=["ggml-drivers-v2.8.6.7"], releases=[
            {"tag_name": "ggml-drivers-v2.8.6.7", "draft": False,
             "assets": [{"name": "release.json"}, {"name": "ggml-2.8.6.7-win-x64-cpu.zip"}]}])
        self.check_absent()

    def test_existing_tag(self) -> None:
        stub_state(self.state, tags=[TAG])
        self.refused(self.check_absent, f"tag {TAG} already exists")

    def test_existing_tag_on_a_later_page(self) -> None:
        stub_state(self.state, tags=[f"other-{i}" for i in range(150)] + [TAG])
        self.refused(self.check_absent, f"tag {TAG} already exists")

    def test_existing_draft_release(self) -> None:
        stub_state(self.state, releases=[{"tag_name": TAG, "draft": True, "assets": []}])
        self.refused(self.check_absent, f"draft release '{TAG}' already exists")

    def test_existing_asset_name_in_another_release(self) -> None:
        stub_state(self.state, releases=[{"tag_name": "misfiled", "draft": False,
                                          "assets": [{"name": f"ggml-{VERSION}-linux-x64-cuda13.tar.gz"}]}])
        self.refused(self.check_absent, f"asset ggml-{VERSION}-linux-x64-cuda13.tar.gz already exists")

    def test_gh_failure_refuses(self) -> None:
        stub_state(self.state, fail="HTTP 401: Bad credentials")
        self.refused(self.check_absent, "cannot confirm what GitHub holds")

    def test_cli_check_absent(self) -> None:
        gh = f"{sys.executable} {ROOT / 'tests' / 'stub_gh.py'}"
        self.assertEqual(run_cli(*["--gh", gh, "check-absent", str(self.out)]), 0)
        stub_state(self.state, tags=[TAG])
        self.assertEqual(run_cli(*["--gh", gh, "check-absent", str(self.out)]), 1)

    def test_listing_of_complete_draft(self) -> None:
        stub_state(self.state, releases=[self.uploaded()])
        draft, notes = drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY)
        self.assertTrue(draft)
        self.assertEqual(notes, [])

    def test_listing_without_digests_notes_them(self) -> None:
        stub_state(self.state, releases=[self.uploaded(digests=False)])
        _, notes = drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY)
        self.assertEqual(len(notes), 4)

    def test_listing_of_published_release_needs_its_tag(self) -> None:
        stub_state(self.state, releases=[self.uploaded(draft=False)])
        self.refused(lambda: drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY), "tag")
        stub_state(self.state, tags=[TAG], releases=[self.uploaded(draft=False)])
        draft, _ = drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY)
        self.assertFalse(draft)

    def test_listing_refusals(self) -> None:
        cases = {
            "missing asset": (lambda r: r["assets"].pop(0), "is missing asset"),
            "extra asset": (lambda r: r["assets"].append({"name": "x.zip", "size": 1, "state": "uploaded"}),
                            "unexpected asset x.zip"),
            "partial upload": (lambda r: r["assets"][0].update(state="starter"), "state is 'starter'"),
            "size": (lambda r: r["assets"][0].update(size=1), "size is 1"),
            "digest": (lambda r: r["assets"][0].update(digest="sha256:" + "0" * 64), "digest is"),
        }
        for label, (change, fragment) in cases.items():
            with self.subTest(label):
                release = self.uploaded()
                change(release)
                stub_state(self.state, releases=[release])
                self.refused(lambda: drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY), fragment)

    def test_listing_without_release(self) -> None:
        self.refused(lambda: drivers.verify_listing(self.out, STUB_GH, drivers.REPOSITORY), "found 0")


class PublishedTests(ToolTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.catalog = self.generate()
        assets = {(TAG, p.name): p for p in (self.out / "assets").iterdir()}
        self.server = GitHubReleaseServer(drivers.REPOSITORY, assets).start()
        self.download = self.work / "download"

    def tearDown(self) -> None:
        self.server.stop()
        super().tearDown()

    def verify(self) -> dict:
        return drivers.verify_published(self.out, self.server.base_url, self.download)

    def requests_for(self, name: str) -> list[dict]:
        return [r for r in self.server.requests if r["path"].endswith(f"/{TAG}/{name}")]

    def test_verifies_published_release(self) -> None:
        transfers = self.verify()
        self.assertEqual(len(transfers), 4)
        self.assertTrue(all(t.resumed == 0 for t in transfers.values()))
        self.assertFalse(self.download.exists())

    def test_resumes_after_dropped_connection_with_fresh_redirect(self) -> None:
        name = f"ggml-{VERSION}-linux-x64-cuda13.tar.gz"
        self.server.truncate_once[name] = 100_000
        transfers = self.verify()
        self.assertEqual(transfers[name].resumed, 1)
        ranged = [r for r in self.requests_for(name) if r["range"] == "bytes=100000-"]
        self.assertEqual(len(ranged), 1)
        self.assertTrue(ranged[0]["if_range"])
        # The range probe, the first download and the resume each resolved their own redirect.
        self.assertEqual(len(self.requests_for(name)), 3)

    def test_if_range_mismatch_answers_full_file(self) -> None:
        name = f"ggml-{VERSION}-win-x64-cpu.zip"
        asset = self.catalog["artifacts"][2]["asset"]
        self.server.truncate_once[name] = 50_000
        url = self.server.base_url + asset["url"][len(drivers.GITHUB):]
        destination = self.work / name
        server = self.server
        original_open = drivers.http_open

        def open_and_rotate(url, headers=None, method="GET"):
            if headers and "Range" in headers:
                server.rotate_etag(TAG, name)
            return original_open(url, headers, method)

        drivers.http_open = open_and_rotate
        try:
            transfer = drivers.download(url, destination, asset["size"])
        finally:
            drivers.http_open = original_open
        self.assertEqual((transfer.resumed, transfer.restarted), (0, 1))
        self.assertEqual(sha(destination.read_bytes()), asset["sha256"])

    def test_refuses_server_without_ranges(self) -> None:
        self.server.ranges = False
        self.refused(self.verify, "a range request for the last byte answered 200")

    def test_refuses_changed_bytes(self) -> None:
        name = f"ggml-{VERSION}-win-x64-cpu.zip"
        path = self.work / "served.zip"
        data = bytearray((self.out / "assets" / name).read_bytes())
        data[20] ^= 1
        path.write_bytes(bytes(data))
        self.server.assets[(TAG, name)] = path
        self.refused(self.verify, f"{name}: sha256 is")

    def test_refuses_missing_asset(self) -> None:
        name = f"ggml-{VERSION}-linux-x64-cpu.tar.gz"
        del self.server.assets[(TAG, name)]
        self.refused(self.verify, "answered 404")

    def test_refuses_different_release_json(self) -> None:
        path = self.work / "release.json"
        path.write_bytes((self.out / "assets" / "release.json").read_bytes().replace(b"cpu", b"cpx"))
        self.server.assets[(TAG, "release.json")] = path
        self.refused(self.verify, "release.json does not match")

    def test_cli_verify_published(self) -> None:
        code = run_cli(*["verify-published", str(self.out), "--base-url", self.server.base_url,
                             "--work", str(self.download)])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
