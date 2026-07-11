#!/usr/bin/env python3
"""
Download preprocessed rfMRI and diffusion NIfTI files for HCP subject
100307 from the hcp-openaccess S3 bucket.

hcp-openaccess is a Requester Pays bucket: it requires AWS credentials
belonging to an account with an approved ConnectomeDB / HCP Data Use
Terms agreement (see ~/.aws/credentials); downloads are billed to that
account.

Pulls only .nii.gz objects under:
    HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/
    HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_RL/
    HCP_1200/100307/MNINonLinear/Results/rfMRI_REST2_LR/
    HCP_1200/100307/MNINonLinear/Results/rfMRI_REST2_RL/
    HCP_1200/100307/T1w/Diffusion/
into data/raw/nifti/100307/, mirroring the path below MNINonLinear/T1w.

Run:
    python3 download_hcp_100307.py --dry-run   # list without downloading
    python3 download_hcp_100307.py             # download
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

BUCKET = "hcp-openaccess"
SUBJECT = "100307"
SUBJECT_PREFIX = f"HCP_1200/{SUBJECT}/"
PREFIXES = [
    f"{SUBJECT_PREFIX}MNINonLinear/Results/rfMRI_REST1_LR/",
    f"{SUBJECT_PREFIX}MNINonLinear/Results/rfMRI_REST1_RL/",
    f"{SUBJECT_PREFIX}MNINonLinear/Results/rfMRI_REST2_LR/",
    f"{SUBJECT_PREFIX}MNINonLinear/Results/rfMRI_REST2_RL/",
    f"{SUBJECT_PREFIX}T1w/Diffusion/",
]
DEST_ROOT = Path("/home/atotilca/pythongpu/data/raw/nifti") / SUBJECT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def iter_nifti_objects(s3_client, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
    """Yield (key, size) for every .nii.gz object under prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, RequestPayer="requester"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".nii.gz"):
                yield key, obj["Size"]


def dest_path_for_key(key: str) -> Path:
    """
    HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/x.nii.gz
        -> DEST_ROOT/MNINonLinear/Results/rfMRI_REST1_LR/x.nii.gz
    Strips the HCP_1200/<subject>/ prefix since DEST_ROOT already
    encodes the subject.
    """
    relative = key[len(SUBJECT_PREFIX):] if key.startswith(SUBJECT_PREFIX) else key
    return DEST_ROOT / relative


def already_downloaded(local_path: Path, remote_size: int) -> bool:
    return local_path.exists() and local_path.stat().st_size == remote_size


def atomic_download(s3_client, bucket: str, key: str, local_path: Path, remote_size: int) -> None:
    """
    Stream `key` to `local_path` atomically. The tempfile is created via
    mkstemp in local_path's own directory so the final os.replace is a
    same-filesystem rename: a network dropout mid-transfer leaves only an
    orphaned .part file behind, never a truncated file at the real path.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=local_path.parent, prefix=f".{local_path.name}.", suffix=".part"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            s3_client.download_fileobj(
                bucket, key, fh, ExtraArgs={"RequestPayer": "requester"}
            )
        downloaded_size = tmp_path.stat().st_size
        if downloaded_size != remote_size:
            raise IOError(
                f"size mismatch after download: expected {remote_size}, "
                f"got {downloaded_size} for {key}"
            )
        os.replace(tmp_path, local_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List matching objects and report sizes without downloading.",
    )
    args = ap.parse_args()

    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "adaptive"}))

    total_files = 0
    total_bytes = 0
    skipped = 0
    downloaded = 0

    try:
        for prefix in PREFIXES:
            log.info(f"listing s3://{BUCKET}/{prefix}")
            for key, size in iter_nifti_objects(s3, BUCKET, prefix):
                total_files += 1
                total_bytes += size
                local_path = dest_path_for_key(key)

                if already_downloaded(local_path, size):
                    skipped += 1
                    log.info(f"skip (matching size)  {local_path}")
                    continue

                if args.dry_run:
                    log.info(f"[dry-run] would download  {key}  ({size} bytes) -> {local_path}")
                    continue

                log.info(f"downloading  {key}  ({size / 1e6:.1f} MB)")
                atomic_download(s3, BUCKET, key, local_path, size)
                downloaded += 1
    except NoCredentialsError:
        log.error(
            "No AWS credentials found. hcp-openaccess is a Requester Pays "
            "bucket -- configure credentials for an AWS account with an "
            "approved HCP/ConnectomeDB data use agreement "
            "(aws configure, or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)."
        )
        raise SystemExit(1)
    except ClientError as e:
        log.error(f"S3 error: {e}")
        raise SystemExit(1)

    log.info(
        f"done. {total_files} .nii.gz files matched "
        f"({total_bytes / 1e9:.2f} GB total) -- "
        f"{downloaded} downloaded, {skipped} already present"
    )


if __name__ == "__main__":
    main()
