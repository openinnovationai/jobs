# PyTorch FSDP Distributed Training Job on OICM+

This job demonstrates distributed training using PyTorch's Fully Sharded Data Parallel (FSDP) API. It is designed to run on a cluster with multiple GPUs and leverages a shared persistent volume to efficiently manage dataset downloads and sharing between worker processes.

## Key Features
- **Distributed Training:** Utilizes PyTorch FSDP for efficient multi-GPU training.
- **Configurable Resources:** Set GPU type/count, CPU, memory, and other hyperparameters in `config.yaml`.
- **Shared Persistent Volume:** Uses a Persistent Volume Claim (PVC) to share data between all worker pods.

## Why Use a PVC (Persistent Volume Claim)?
- The `shared_volume` section in `config.yaml` will create a PVC for your job and mount it under /pvc-home in all your job pods.
- **It is required to use a PVC** so that the dataset is downloaded only once by the master process (rank 0), and then shared with all other workers via the mounted volume.
- Without a PVC, each worker would download the dataset separately, leading to redundant downloads, wasted bandwidth, and potential data inconsistency.
- The shared volume is mounted at `/pvc-home` inside each container.

## Example `config.yaml`
```yaml
resources:
  gpu:
    accelerator_count: 8
    accelerator: H100 # change it to the available accelerator in your cluster
  memory: 128
  cpu: 4
replicas: 2
config_map:
  FOLDER_NAME: "test"
  EPOCHS: "5"
  BATCH_SIZE: "256"
  LEARNING_RATE: "0.01"
shared_volume:
  size: 20
```

## How It Works
- Only the master process (rank 0) downloads the dataset to the shared volume (`/pvc-home`).
- All other processes wait until the download is complete, then access the dataset from the same shared location.
- This ensures efficient data sharing and avoids redundant downloads.

## Notes
- Adjust the `accelerator` and resource values in `config.yaml` to match your environment.

---

For more details, see the code in `main.py` and the configuration in `config.yaml`.
