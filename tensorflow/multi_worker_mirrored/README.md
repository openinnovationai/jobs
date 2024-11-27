# Distributed TensorFlow Training Script with Multi-Worker Strategy

This repository contains a distributed training script leveraging TensorFlow's `tf.distribute.MultiWorkerMirroredStrategy` for efficient training on multiple GPUs or nodes. The script is designed for scalability and includes functionalities to handle multi-node setups

## Features

- **Multi-Worker Distributed Training**: Utilizes `MultiWorkerMirroredStrategy` for training across multiple GPUs/nodes.
- **Templated Resource Configuration**: Easily define resources, replicas, and other parameters.
- **Dependency Management**: Ensure reproducibility with pinned dependency versions.
- **Secret Management**: Secure integration with APIs and cloud storage using environment variables.

**Note**: You may need to Push checkpoints to services like:
  - [HuggingFace Hub](https://huggingface.co/)
  - AWS S3
  - Google Cloud Storage
  - Azure Blob Storage
  - Custom file servers using FTP or WebDAV.

## Prebuilt Images of our Platform and TensorFlow Dependencies
Platform's Prebuilt Images
The platform provides prebuilt Docker images optimized for specific hardware and use cases, ensuring seamless execution and improved performance. Depending on your setup, you can use these images without additional customization:

  - CPU Workloads: Images come pre-installed with tensorflow-cpu for environments without GPU acceleration.
  - ROCm Workloads: For AMD GPUs, images include tensorflow-rocm, optimized for ROCm-enabled devices such as MI210.


**Note:** Currently, we do not support `ParameterServerStrategy` due to its experimental status in TensorFlow, which includes limited stability, incomplete features, and potential unresolved bugs that can prevent training from initiating properly; we recommend using fully supported strategies like `MultiWorkerMirroredStrategy` for reliable performance.
