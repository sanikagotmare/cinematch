"""
scripts/lfs_fetch.py
─────────────────────
Render's build environment doesn't have git-lfs installed.
This script downloads the actual CSV content from GitHub's LFS
media server using the pointer files already checked out.

Each LFS-tracked file in the repo is actually just a small pointer
file containing an oid (sha256 hash) and size. This script reads
those pointers and downloads the real file from GitHub's LFS API.
"""
import os
import re
import requests
from pathlib import Path

REPO = "sanikagotmare/cinematch"
FILES = [
    "data/tmdb_5000_movies.csv",
    "data/tmdb_5000_credits.csv",
]


def parse_pointer(path: str) -> dict | None:
    """Parse a Git LFS pointer file to extract oid and size."""
    content = Path(path).read_text(errors="ignore")
    if "git-lfs" not in content:
        return None  # already a real file, not a pointer

    oid_match  = re.search(r"oid sha256:([a-f0-9]+)", content)
    size_match = re.search(r"size (\d+)", content)
    if not oid_match or not size_match:
        return None

    return {"oid": oid_match.group(1), "size": int(size_match.group(1))}


def download_lfs_file(path: str, pointer: dict) -> None:
    """Download the real file content from GitHub's LFS batch API."""
    oid  = pointer["oid"]
    size = pointer["size"]

    batch_url = f"https://github.com/{REPO}.git/info/lfs/objects/batch"
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }

    print(f"  Requesting LFS object for {path} (oid={oid[:12]}...)")
    r = requests.post(batch_url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    obj = data["objects"][0]
    if "error" in obj:
        raise RuntimeError(f"LFS error for {path}: {obj['error']}")

    download_url = obj["actions"]["download"]["href"]

    print(f"  Downloading {size / 1024 / 1024:.1f} MB...")
    with requests.get(download_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"  Done: {path}")


def main():
    print("Checking for Git LFS pointer files...")
    for path in FILES:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue

        pointer = parse_pointer(path)
        if pointer is None:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} is already a real file ({size_mb:.1f} MB). Skipping.")
            continue

        download_lfs_file(path, pointer)

    print("LFS fetch complete.")


if __name__ == "__main__":
    main()
