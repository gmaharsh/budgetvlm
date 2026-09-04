"""Optional S3 sync for experiment results (RunPod → durable storage).

Configure via env (preferred on RunPod) or configs/default.yaml:

    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1
    export BUDGETVLM_S3_URI=s3://my-bucket/budgetvlm/runs

Usage:

    python -m src.s3_sync upload --run-id runpod_fixed20
    python -m src.s3_sync download --run-id runpod_fixed20 --dest results_from_s3
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .utils import ensure_dir, get_logger, load_config, project_path

log = get_logger("s3")

DEFAULT_INCLUDE = (
    "results/predictions",
    "results/metrics",
    "results/figures",
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from s3://bucket/prefix."""
    uri = uri.strip()
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"Missing bucket in URI: {uri}")
    return bucket, prefix.rstrip("/")


def resolve_s3_uri(cfg: dict | None = None, uri: str | None = None) -> str | None:
    if uri:
        return uri
    env = os.environ.get("BUDGETVLM_S3_URI") or os.environ.get("BUDGETVLM_S3_BUCKET")
    if env:
        if env.startswith("s3://"):
            return env
        return f"s3://{env.rstrip('/')}/budgetvlm"
    if cfg:
        s3 = cfg.get("s3") or {}
        if s3.get("uri"):
            return str(s3["uri"])
        if s3.get("bucket"):
            prefix = str(s3.get("prefix") or "budgetvlm").strip("/")
            return f"s3://{s3['bucket']}/{prefix}"
    return None


def _client(region: str | None = None):
    try:
        import boto3
    except ImportError as e:
        raise SystemExit("boto3 is required for S3 sync. pip install boto3") from e
    kwargs = {}
    region = region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _iter_files(roots: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.name != ".gitkeep":
                files.append(f)
    return files


def upload_results(
    s3_uri: str,
    run_id: str | None = None,
    local_roots: Iterable[str | Path] | None = None,
    region: str | None = None,
    dry_run: bool = False,
) -> str:
    """Upload local result files under s3_uri/run_id/. Returns destination URI."""
    bucket, prefix = parse_s3_uri(s3_uri)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_prefix = "/".join(p for p in (prefix, run_id) if p)
    roots = [project_path(r) if not Path(r).is_absolute() else Path(r) for r in (local_roots or DEFAULT_INCLUDE)]
    files = _iter_files(roots)
    if not files:
        log.warning("No result files found under %s", list(roots))
        return f"s3://{bucket}/{dest_prefix}"

    client = None if dry_run else _client(region)
    base = project_path()
    for f in files:
        try:
            rel = f.relative_to(base)
        except ValueError:
            rel = Path(f.name)
        key = f"{dest_prefix}/{rel.as_posix()}"
        log.info("upload %s -> s3://%s/%s", f, bucket, key)
        if not dry_run:
            client.upload_file(str(f), bucket, key)
    log.info("Uploaded %d files to s3://%s/%s", len(files), bucket, dest_prefix)
    return f"s3://{bucket}/{dest_prefix}"


def download_results(
    s3_uri: str,
    run_id: str | None = None,
    dest_dir: str | Path | None = None,
    region: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Download s3_uri[/run_id] into dest_dir (default: results_from_s3/run_id)."""
    bucket, prefix = parse_s3_uri(s3_uri)
    full_prefix = "/".join(p for p in (prefix, run_id or "") if p)
    if full_prefix and not full_prefix.endswith("/"):
        full_prefix += "/"

    dest = Path(dest_dir) if dest_dir else project_path("results_from_s3", run_id or "latest")
    ensure_dir(dest)

    client = _client(region)
    token = None
    n = 0
    while True:
        kwargs = {"Bucket": bucket, "Prefix": full_prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(full_prefix) :] if full_prefix and key.startswith(full_prefix) else Path(key).name
            out = dest / rel
            ensure_dir(out.parent)
            log.info("download s3://%s/%s -> %s", bucket, key, out)
            if not dry_run:
                client.download_file(bucket, key, str(out))
            n += 1
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    log.info("Downloaded %d objects into %s", n, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync BudgetVLM results with S3")
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--uri", default=None, help="s3://bucket/prefix")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dest", default=None, help="local dest for download")
    ap.add_argument("--region", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    uri = resolve_s3_uri(cfg, args.uri)
    if not uri:
        raise SystemExit(
            "No S3 URI configured. Set BUDGETVLM_S3_URI=s3://bucket/prefix "
            "or configs.s3.uri / configs.s3.bucket"
        )

    region = args.region or (cfg.get("s3") or {}).get("region")
    if args.action == "upload":
        dest = upload_results(uri, run_id=args.run_id, region=region, dry_run=args.dry_run)
        print(dest)
    else:
        path = download_results(
            uri, run_id=args.run_id, dest_dir=args.dest, region=region, dry_run=args.dry_run
        )
        print(path)


if __name__ == "__main__":
    main()
