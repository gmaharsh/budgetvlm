"""S3 bucket ensure + results sync for RunPod / cloud experiments.

On pipeline start:
  - resolve bucket (env / config / auto from AWS account id)
  - create bucket if missing
  - update security settings (block public access, SSE-S3)

On pipeline end:
  - upload results/predictions, metrics, figures under run_id/

Env (preferred on RunPod):

    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1
    # optional overrides:
    export BUDGETVLM_S3_BUCKET=budgetvlm-704052814573
    export BUDGETVLM_S3_URI=s3://budgetvlm-704052814573/results
    export BUDGETVLM_AWS_ACCOUNT_ID=704052814573

Usage:

    python -m src.s3_sync ensure
    python -m src.s3_sync upload --run-id runpod_fixed20
    python -m src.s3_sync download --run-id runpod_fixed20 --dest results_from_s3
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .utils import ensure_dir, get_logger, load_config, project_path

log = get_logger("s3")

DEFAULT_INCLUDE = (
    "results/predictions",
    "results/metrics",
    "results/figures",
)
DEFAULT_PREFIX = "results"


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


def _region(cfg: dict | None = None, region: str | None = None) -> str:
    return (
        region
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or ((cfg or {}).get("s3") or {}).get("region")
        or "us-east-1"
    )


def _client(region: str | None = None):
    try:
        import boto3
    except ImportError as e:
        raise SystemExit("boto3 is required for S3 sync. pip install boto3") from e
    return boto3.client("s3", region_name=_region(region=region))


def _sts_client(region: str | None = None):
    import boto3

    return boto3.client("sts", region_name=_region(region=region))


def aws_account_id(region: str | None = None) -> str:
    env = os.environ.get("BUDGETVLM_AWS_ACCOUNT_ID") or os.environ.get("AWS_ACCOUNT_ID")
    if env:
        return env.strip()
    return str(_sts_client(region).get_caller_identity()["Account"])


def default_bucket_name(account_id: str | None = None, region: str | None = None) -> str:
    aid = account_id or aws_account_id(region)
    return f"budgetvlm-{aid}"


def resolve_s3_uri(cfg: dict | None = None, uri: str | None = None) -> str | None:
    """Resolve results URI from explicit arg, env, config, or auto account bucket."""
    if uri:
        return uri

    env_uri = os.environ.get("BUDGETVLM_S3_URI")
    if env_uri:
        return env_uri if env_uri.startswith("s3://") else f"s3://{env_uri.rstrip('/')}/{DEFAULT_PREFIX}"

    env_bucket = os.environ.get("BUDGETVLM_S3_BUCKET")
    if env_bucket:
        prefix = os.environ.get("BUDGETVLM_S3_PREFIX", DEFAULT_PREFIX).strip("/")
        return f"s3://{env_bucket}/{prefix}"

    s3 = (cfg or {}).get("s3") or {}
    if s3.get("uri"):
        return str(s3["uri"])
    if s3.get("bucket"):
        prefix = str(s3.get("prefix") or DEFAULT_PREFIX).strip("/")
        return f"s3://{s3['bucket']}/{prefix}"

    # Auto: budgetvlm-<account_id>/results when credentials work
    if s3.get("auto_create", True) or s3.get("enabled"):
        try:
            bucket = default_bucket_name(region=_region(cfg))
            prefix = str(s3.get("prefix") or DEFAULT_PREFIX).strip("/")
            return f"s3://{bucket}/{prefix}"
        except Exception as e:
            log.warning("Could not auto-resolve S3 bucket from AWS identity: %s", e)
            return None
    return None


def bucket_exists(client: Any, bucket: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        code = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
        err = e.response.get("Error", {}).get("Code", "")
        if code in (404, 403) or err in {"404", "NoSuchBucket", "NotFound"}:
            # 403 can mean bucket exists but owned by another account
            if code == 403 or err in {"403", "AccessDenied", "Forbidden"}:
                raise PermissionError(
                    f"Cannot access bucket s3://{bucket} (forbidden). "
                    "Choose another name or fix IAM permissions."
                ) from e
            return False
        raise


def ensure_bucket(
    s3_uri: str | None = None,
    *,
    cfg: dict | None = None,
    region: str | None = None,
    dry_run: bool = False,
) -> str:
    """Create bucket if missing; always re-apply block-public-access + encryption.

    Returns the results URI (s3://bucket/prefix).
    """
    cfg = cfg or {}
    region = _region(cfg, region)
    uri = resolve_s3_uri(cfg, s3_uri)
    if not uri:
        raise SystemExit(
            "No S3 URI configured and auto-resolve failed. "
            "Set AWS credentials and/or BUDGETVLM_S3_URI / BUDGETVLM_S3_BUCKET."
        )
    bucket, prefix = parse_s3_uri(uri)
    prefix = prefix or DEFAULT_PREFIX
    uri = f"s3://{bucket}/{prefix}"

    s3_cfg = cfg.get("s3") or {}
    if dry_run:
        log.info("[dry-run] ensure bucket s3://%s (region=%s)", bucket, region)
        return uri

    client = _client(region)
    exists = bucket_exists(client, bucket)
    if not exists:
        log.info("Creating bucket s3://%s in %s", bucket, region)
        if region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    else:
        log.info("Bucket s3://%s already exists — updating settings", bucket)

    # Block all public access
    if s3_cfg.get("block_public_access", True):
        client.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        log.info("Applied Block Public Access on s3://%s", bucket)

    # Default encryption (SSE-S3)
    if s3_cfg.get("ensure_encryption", True):
        client.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )
        log.info("Applied SSE-S3 encryption on s3://%s", bucket)

    if s3_cfg.get("ensure_versioning", False):
        client.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        log.info("Enabled versioning on s3://%s", bucket)

    # Ownership controls: BucketOwnerEnforced (ACLs off)
    try:
        client.put_bucket_ownership_controls(
            Bucket=bucket,
            OwnershipControls={
                "Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}],
            },
        )
    except Exception as e:
        log.warning("Could not set ownership controls: %s", e)

    # Persist resolved URI for later upload steps in the same shell via stdout marker
    log.info("Results will be stored under %s/<run_id>/", uri)
    return uri


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
    roots = [
        project_path(r) if not Path(r).is_absolute() else Path(r)
        for r in (local_roots or DEFAULT_INCLUDE)
    ]
    files = _iter_files(roots)
    if not files:
        log.warning("No result files found under %s", list(roots))
        return f"s3://{bucket}/{dest_prefix}"

    client = None if dry_run else _client(region)
    results_root = project_path("results")
    for f in files:
        try:
            if results_root in f.parents or f.parent == results_root:
                rel = f.relative_to(results_root)
            else:
                rel = f.relative_to(project_path())
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
            rel = (
                key[len(full_prefix) :]
                if full_prefix and key.startswith(full_prefix)
                else Path(key).name
            )
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
    ap = argparse.ArgumentParser(description="Ensure S3 bucket + sync BudgetVLM results")
    ap.add_argument("action", choices=["ensure", "upload", "download"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--uri", default=None, help="s3://bucket/prefix")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dest", default=None, help="local dest for download")
    ap.add_argument("--region", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    region = args.region or (cfg.get("s3") or {}).get("region")

    if args.action == "ensure":
        uri = ensure_bucket(args.uri, cfg=cfg, region=region, dry_run=args.dry_run)
        print(uri)
        return

    uri = resolve_s3_uri(cfg, args.uri)
    if not uri:
        raise SystemExit(
            "No S3 URI configured. Run: python -m src.s3_sync ensure "
            "(with AWS credentials), or set BUDGETVLM_S3_URI."
        )

    if args.action == "upload":
        # Re-ensure settings before upload so first-time runs are safe
        uri = ensure_bucket(uri, cfg=cfg, region=region, dry_run=args.dry_run)
        dest = upload_results(uri, run_id=args.run_id, region=region, dry_run=args.dry_run)
        print(dest)
    else:
        path = download_results(
            uri, run_id=args.run_id, dest_dir=args.dest, region=region, dry_run=args.dry_run
        )
        print(path)


if __name__ == "__main__":
    main()
