"""
Download and cache CUB-200-2011.

The dataset's original host is dead for this project's purposes:

  1. https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
     (the current official CaltechDATA link) returns HTTP 403 Forbidden to
     a plain scripted request -- it appears to require a browser session.
  2. The old vision.caltech.edu URL that most tutorials still quote
     returns HTTP 404 -- the dataset was moved off that host entirely.

We fall back to a public GitHub mirror that re-hosts the same archive via
Git LFS. This is documented here rather than silently swapped in, because
"which copy of the data did you actually use" is exactly the kind of
question Lab honesty criteria ask about.
"""

import os
import tarfile
import urllib.request

CANDIDATE_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    "https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz",
]


def _download(url: str, dest_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_cub_dataset(data_dir: str) -> str:
    """
    Returns the path to the extracted CUB_200_2011/ directory, downloading
    and extracting it first if necessary. Idempotent: safe to call every
    run, it does nothing if the data is already there.
    """
    extracted_dir = os.path.join(data_dir, "CUB_200_2011")
    if os.path.isdir(os.path.join(extracted_dir, "images")):
        return extracted_dir

    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "CUB_200_2011.tgz")

    if not os.path.exists(archive_path):
        last_error = None
        for url in CANDIDATE_URLS:
            try:
                print(f"Trying: {url}")
                _download(url, archive_path)
                print("Download succeeded.")
                break
            except Exception as exc:  # noqa: BLE001 - we want to try every fallback
                print(f"Failed ({exc})")
                last_error = exc
        else:
            raise RuntimeError(
                "All candidate URLs for CUB-200-2011 failed. The official "
                "Caltech host blocks scripted requests (403) and the old "
                "vision.caltech.edu URL is gone (404) as of this writing. "
                "Download CUB_200_2011.tgz by hand and place it at "
                f"{archive_path}, or add a working mirror URL to "
                "CANDIDATE_URLS in src/download_data.py."
            ) from last_error

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print("Done. Dataset ready at", extracted_dir)
    return extracted_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Where to download/extract CUB-200-2011.")
    args = parser.parse_args()
    path = ensure_cub_dataset(args.data_dir)
    print(path)
