# Distributed Fine-tuning Job with LoRA

A distributed machine learning pipeline for fine-tuning BERT models using LoRA (Low-Rank Adaptation) on the GLUE SST-2 sentiment analysis task with multi-GPU support.

## Overview

This project implements an efficient distributed fine-tuning workflow that:
- Supports multi-GPU distributed training using PyTorch DDP (Distributed Data Parallel)
- Uses LoRA for parameter-efficient training
- Automatically handles quantization based on GPU count (8-bit for single GPU, FP32 for multi-GPU)
- Trains on the Stanford Sentiment Treebank (SST-2) dataset
- Automatically uploads results to Hugging Face Hub
- Includes process synchronization and shared volume support

## What This Script Does

This distributed training script provides a complete end-to-end fine-tuning pipeline that:

1. **Distributed Training Setup**: Automatically detects multi-GPU environments and configures PyTorch DDP (Distributed Data Parallel) across multiple processes/nodes
2. **Pulls model and dataset from Hugging Face**: Downloads the pre-trained BERT model (`bert-base-uncased`) and the GLUE SST-2 dataset from Hugging Face Hub
3. **Adaptive Fine-tuning**: Uses LoRA (Low-Rank Adaptation) with smart quantization - 8-bit for single GPU setups, FP32 for multi-GPU distributed training
4. **Process Synchronization**: Ensures all processes complete training before model upload using Accelerate's `wait_for_everyone()`
5. **Uploads the result model to Hugging Face**: Main process automatically pushes the fine-tuned model to your specified Hugging Face repository

## Features

- **Distributed Training**: Full support for multi-GPU training using PyTorch DDP with automatic process coordination
- **Adaptive Quantization**: Automatically switches between 8-bit quantization (single GPU) and FP32 (multi-GPU) based on available hardware
- **Parameter-Efficient Training**: Uses LoRA to fine-tune only a small subset of parameters across all processes
- **Process Synchronization**: Built-in synchronization barriers to ensure consistent training across all workers
- **Shared Volume Support**: Configured to work with shared storage across distributed nodes
- **Automatic Model Upload**: Only the main process handles model upload to prevent conflicts
- **Environment Variable Logging**: Displays distributed training environment variables for debugging

## Project Structure

```
distributed-finetuning-job/
├── main.py                 # Main distributed training script with process 
├── config.yaml            # Resource configuration for multi-GPU deployment
├── requirements.txt       # Python dependencies for distributed training
├── utils/
│   ├── __init__.py
│   └── training_utils.py  # Core training utilities with distributed support
└── README.md             # This file
```

## Configuration

### Environment Variables

The following environment variables must be set for all processes:

- `HF_TOKEN`: Your Hugging Face API token for model upload
- `RESULT_MODEL_ID`: Target Hugging Face model repository ID  
- `RUN_NAME`: Name for the training run (for tracking purposes)

**Distributed Training Environment Variables** (automatically set by the training infrastructure):
- `RANK`: Global rank of the current process
- `LOCAL_RANK`: Local rank within the current node
- `WORLD_SIZE`: Total number of processes across all nodes
- `NODE_RANK`: Rank of the current node

### Resource Configuration (`config.yaml`)

```yaml
resources:
  gpu:
    accelerator_count: 1      # GPUs per replica
    accelerator: H100 
  memory: 128                # Memory per replica in GB
  cpu: 4                     # CPU cores per replica
replicas: 4                  # Number of distributed workers/nodes
config_map:
  HF_TOKEN: <your hf token with write creds>
  RESULT_MODEL_ID: <your repo id>
  RUN_NAME: "run #1"
shared_volume:
  size: 10                   # Shared storage size in GB for model checkpoints
```

**Distributed Setup Notes:**
- Each replica gets 1 H100 GPU, so this configuration uses 4 H100s total
- Shared volume allows all processes to access the same model output directory
- Total compute: 4 × (1 H100 + 128GB RAM + 4 CPU cores)

### Training Configuration

The current distributed setup uses:
- **Base Model**: `bert-base-uncased`
- **Dataset**: GLUE SST-2 (Stanford Sentiment Treebank)
- **Task**: Binary sentiment classification
- **Epochs**: 1 (configurable in `main.py`)
- **Batch Size**: 64 per device (total effective batch size: 64 × 4 = 256)
- **Quantization**: Adaptive (8-bit for single GPU, FP32 for multi-GPU)
- **Distributed Backend**: PyTorch DDP via Accelerate

### LoRA Configuration

The LoRA setup includes:
- **Rank (r)**: 8
- **Alpha**: 16  
- **Target Modules**: Query and value projection layers
- **Dropout**: 0.05
- **Task Type**: Sequence classification
- **Gradient Checkpointing**: Enabled across all processes

## Model Details

### Architecture
- **Backbone**: BERT-base-uncased
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) 
- **Quantization**: Adaptive (8-bit for single GPU, FP32 for distributed training)
- **Task**: Binary sentiment classification (positive/negative)
- **Distribution**: PyTorch DDP across multiple GPUs/nodes

### Performance Optimizations
- **Distributed Training**: Scales across multiple GPUs with automatic gradient synchronization
- **Gradient Checkpointing**: Enabled across all processes to reduce memory usage
- **Adaptive Quantization**: Automatically disables quantization for multi-GPU to avoid compatibility issues
- **Parameter-Efficient Training**: Only ~0.3% of parameters are trainable with LoRA
- **Process Synchronization**: Built-in barriers prevent race conditions during training and model saving

## Output

After distributed training completion:
1. Model files are saved to `/pvc-home/output/` directory on shared volume
2. Only the main process (rank 0) uploads the model to the specified Hugging Face repository
3. All processes synchronize before model upload to ensure training completion
4. Training logs from all processes are available in the output directory
5. Repository URL is logged for reference

## Dependencies

- `transformers==4.56.2`: Hugging Face transformers library
- `datasets==4.1.1`: Dataset loading and processing
- `accelerate==1.10.1`: Distributed training support
- `peft==0.17.1`: Parameter-Efficient Fine-Tuning library
- `bitsandbytes==0.47.0`: 8-bit quantization
- `evaluate==0.4.6`: Model evaluation metrics

## Hardware Requirements

### Distributed Setup (Default Configuration)
- **GPUs**: 4× H100 (1 per replica)
- **Memory**: 4× 128GB RAM (512GB total)
- **CPU**: 4× 4 cores (16 cores total)
- **Storage**: 10GB shared volume + local storage per node
- **Network**: High-speed interconnect (InfiniBand recommended for multi-node)

### Single GPU Fallback
- **Minimum**: Any CUDA-compatible GPU with 8GB+ VRAM
- **Memory**: 16GB+ system RAM
- **CPU**: 4+ cores
- **Note**: Automatically enables 8-bit quantization for memory efficiency

### Network Requirements
- **Multi-node**: Low-latency network for gradient synchronization
- **Single-node multi-GPU**: PCIe or NVLink interconnect

## Troubleshooting

### Common Issues

#### Distributed Training Issues
1. **Process Synchronization Failures**: Ensure all processes can communicate and shared volume is accessible
2. **Rank/World Size Mismatch**: Verify environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`) are set correctly
3. **Network Connectivity**: Check inter-node communication for multi-node setups
4. **Shared Volume Access**: Ensure all processes can read/write to `/pvc-home/output/`

#### General Training Issues  
5. **Out of Memory**: Reduce batch size in `main.py` or enable gradient accumulation
6. **Hugging Face Upload Fails**: Verify `HF_TOKEN` and `RESULT_MODEL_ID` are correct
7. **Dataset Loading Issues**: Ensure internet connection for downloading GLUE dataset
8. **Quantization Errors**: For multi-GPU setups, quantization is automatically disabled

### Debugging

#### Environment Variable Debugging
The script automatically logs distributed training environment variables at startup:
```bash
RANK = 0
LOCAL_RANK = 0  
WORLD_SIZE = 4
NODE_RANK = 0
```

#### Enable Verbose Logging
Modify the training arguments in `utils/training_utils.py`:
```python
logging_steps=1  # Log every step instead of every 10
```

#### Process-Specific Debugging
Each process logs its own training progress. Check logs from all processes to identify issues:
- Main process (rank 0): Handles model upload
- Worker processes (rank 1-3): Handle distributed training only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
