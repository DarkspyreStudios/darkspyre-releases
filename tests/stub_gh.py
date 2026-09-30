#!/usr/bin/env python3
"""A stand-in for the gh CLI that answers only read-only `gh api` listings from a JSON state file.

STUB_GH_STATE names a JSON file:

    {"tags": ["name", ...], "releases": [{"tag_name": ..., "draft": ..., "assets": [...]}], "fail": "message"}

It answers `gh api --method GET repos/<owner>/<repo>/(tags|releases)?per_page=N&page=P` with that
page of the list, as GitHub does. When "fail" is set, every call exits 1 with that message.
STUB_GH_LOG, when set, receives each argument list as one JSON line. Any other command exits 2.
"""
import json
import os
import re
import sys


def main(argv: list[str]) -> int:
    if os.environ.get("STUB_GH_LOG"):
        with open(os.environ["STUB_GH_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps(argv) + "\n")
    with open(os.environ["STUB_GH_STATE"], encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("fail"):
        print(state["fail"], file=sys.stderr)
        return 1
    if len(argv) != 4 or argv[:3] != ["api", "--method", "GET"]:
        print(f"stub gh: unsupported command {argv}", file=sys.stderr)
        return 2
    match = re.match(r"^repos/[^/]+/[^/]+/(tags|releases)\?per_page=(\d+)&page=(\d+)$", argv[3])
    if not match:
        print(f"stub gh: unsupported path {argv[3]}", file=sys.stderr)
        return 2
    kind, per_page, page = match.group(1), int(match.group(2)), int(match.group(3))
    items = [{"name": t} for t in state.get("tags", [])] if kind == "tags" else state.get("releases", [])
    print(json.dumps(items[(page - 1) * per_page:page * per_page]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
