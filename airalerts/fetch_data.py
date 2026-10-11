"""Download the pinned source files and verify their SHA-256.

Run:  python -m airalerts.fetch_data
The URL contains the exact upstream commit, so you always get the same bytes.
"""
import sys
import urllib.request

from . import config
from .data import load_manifest, sha256_file


def main():
    manifest = load_manifest()
    repo, commit = manifest["upstream_repo"], manifest["upstream_commit"]
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, info in manifest["files"].items():
        path = config.RAW_DIR / name
        if path.exists() and sha256_file(path) == info["sha256"]:
            print(f"[OK]   {name}: already present, SHA-256 matches")
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/datasets/{name}"
        print(f"[GET]  {name} <- {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        got = sha256_file(tmp)
        if got != info["sha256"]:
            tmp.unlink()
            print(f"[FAIL] {name}: SHA-256 mismatch\n       expected {info['sha256']}\n       got      {got}")
            ok = False
            continue
        tmp.replace(path)
        print(f"[OK]   {name}: downloaded, SHA-256 matches")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
