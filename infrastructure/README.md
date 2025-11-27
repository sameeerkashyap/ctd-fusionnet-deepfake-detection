# Infrastructure Setup

Terraform configuration provisions a Deep Learning VM on Google Cloud with a Tesla T4 GPU, CUDA, and the PyTorch m131 image release. It also creates an SSH key pair inside the repo-level `.ssh` directory so you can authenticate to the instance without manual key generation. No data-protection backup services are configured.

## Prerequisites

1. Terraform `>= 1.6`
2. GCP project with billing enabled
3. `gcloud auth application-default login` or a service account key exported as `GOOGLE_APPLICATION_CREDENTIALS`

## Usage

1. Duplicate the example tfvars file and edit the values:
   ```bash
   cd /Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/infrastructure
   cp terraform.tfvars.example terraform.tfvars
   # edit terraform.tfvars with your project, region, etc.
   ```
   Use `gcloud compute images list --project=deeplearning-platform-release --filter="family:pytorch-*"` if you need to discover the latest PyTorch Deep Learning VM family and update `dl_image_family` accordingly.
   > **Tip:** Ensure the `zone` you pick actually offers the `nvidia-tesla-t4` accelerator (e.g., `us-central1-b`, `us-west1-b`). Check availability with `gcloud compute accelerator-types list --zones <zone>`. Bucket names must be globally unique; update `bucket_name` accordingly.
2. Run Terraform pointing to that tfvars file (automatic when named `terraform.tfvars`):
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

Terraform also provisions a Cloud Storage bucket and grants the VM’s default service account `roles/storage.objectAdmin`, so the instance can read/write objects without extra credential files.

The SSH key pair is written to `../.ssh/<ssh_key_name>` (set in `terraform.tfvars`) and ignored by git. After apply, grab the ready-to-run SSH command:

```bash
terraform output -raw ssh_command
```

> **Note:** The default image family string corresponds to the PyTorch CUDA m131 release from the Deep Learning VM catalog. Change `dl_image_family` if Google updates the naming convention.

