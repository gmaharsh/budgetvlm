# RunPod + S3 results (auto bucket)

The GPU pipeline **ensures the S3 bucket at start** and **uploads results at end**.

## What happens automatically

```text
bash scripts/run_gpu_pipeline.sh
        │
        ├─ python -m src.s3_sync ensure
        │     create bucket if missing
        │     update: Block Public Access + SSE-S3
        │
        ├─ run experiments → results/
        │
        └─ python -m src.s3_sync upload --run-id <id>
              → s3://budgetvlm-<account>/results/<run_id>/...
```

Default bucket name: `budgetvlm-<AWS_ACCOUNT_ID>`  
Example for account `704052814573`: `s3://budgetvlm-704052814573/results/`

## RunPod env

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export BUDGETVLM_AWS_ACCOUNT_ID=704052814573   # optional pin
```

Then:

```bash
git clone https://github.com/gmaharsh/budgetvlm.git
cd budgetvlm
bash scripts/run_gpu_pipeline.sh
```

## Manual commands

```bash
# create/update bucket only
python -m src.s3_sync ensure

# upload current results/
python -m src.s3_sync upload --run-id runpod_fixed20

# download later on your laptop
python -m src.s3_sync download --run-id runpod_fixed20 --dest ./from_s3
```

## IAM permissions needed

On the bucket (or account):

- `s3:CreateBucket`
- `s3:HeadBucket` / `s3:ListBucket`
- `s3:PutBucketPublicAccessBlock`
- `s3:PutEncryptionConfiguration`
- `s3:PutBucketOwnershipControls`
- `s3:PutObject`
- `s3:GetObject`
