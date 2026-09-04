# RunPod + S3 results

Use RunPod for GPU experiments and S3 so results survive pod shutdown.

## 1. Create an S3 bucket (once)

```bash
aws s3 mb s3://YOUR_BUCKET --region us-east-1
```

Suggested prefix layout:

```text
s3://YOUR_BUCKET/budgetvlm/
  runpod_20260904T021500Z/
    results/predictions/...
    results/metrics/...
    results/figures/...
```

## 2. Credentials on the RunPod

In the pod terminal (or RunPod Secrets / env):

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export BUDGETVLM_S3_URI=s3://YOUR_BUCKET/budgetvlm
```

Use an IAM user/role limited to this bucket (`s3:PutObject`, `s3:GetObject`, `s3:ListBucket`).

## 3. Run experiments

```bash
git clone https://github.com/gmaharsh/budgetvlm.git
cd budgetvlm
bash scripts/run_gpu_pipeline.sh
```

If `BUDGETVLM_S3_URI` is set, the script uploads `results/` at the end under a UTC `run_id`.

Manual upload / download:

```bash
python -m src.s3_sync upload --run-id runpod_fixed20
python -m src.s3_sync download --run-id runpod_fixed20 --dest ./from_s3
```

Dry-run (lists what would upload, no network writes beyond local scan):

```bash
python -m src.s3_sync upload --run-id test --dry-run
```

## 4. Pull results to your laptop

```bash
export BUDGETVLM_S3_URI=s3://YOUR_BUCKET/budgetvlm
python -m src.s3_sync download --run-id runpod_YYYYMMDDThhmmssZ --dest ./results_from_s3
```

Or:

```bash
aws s3 sync s3://YOUR_BUCKET/budgetvlm/runpod_YYYYMMDDThhmmssZ ./results_from_s3
```
