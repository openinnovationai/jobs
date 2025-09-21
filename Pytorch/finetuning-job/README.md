# Fine-tuning Job with LoRA

A machine learning pipeline for fine-tuning BERT models using LoRA (Low-Rank Adaptation) on the GLUE SST-2 sentiment analysis task.

## Overview

This project implements an efficient fine-tuning workflow that:
- Uses LoRA for parameter-efficient training
- Applies 8-bit quantization to reduce memory usage
- Trains on the Stanford Sentiment Treebank (SST-2) dataset
- Automatically uploads results to Hugging Face Hub
- Tracks experiments using OIP tracking client

## What This Script Does

This script provides a complete end-to-end fine-tuning pipeline that:

1. **Pulls model and dataset from Hugging Face**: Automatically downloads the pre-trained BERT model (`bert-base-uncased`) and the GLUE SST-2 dataset from Hugging Face Hub
2. **Performs fine-tuning**: Uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the model on the sentiment analysis task with 8-bit quantization for memory optimization
3. **Uploads the result model to Hugging Face**: Automatically pushes the fine-tuned model to your specified Hugging Face repository for sharing and deployment

## Features

- **Parameter-Efficient Training**: Uses LoRA to fine-tune only a small subset of parameters
- **Memory Optimization**: 8-bit quantization reduces GPU memory requirements
- **Automatic Model Upload**: Trained models are automatically pushed to Hugging Face Hub
- **Experiment Tracking**: Built-in experiment tracking and logging
- **Configurable Resources**: GPU and compute resources defined in `config.yaml`

## Project Structure

```
finetuning-job/
├── main.py                 # Main training script and entry point
├── config.yaml            # Resource configuration for deployment
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   └── training_utils.py  # Core training utilities and LoRA setup
└── README.md             # This file
```

## Configuration

### Environment Variables

The following environment variables must be set:

- `HF_TOKEN`: Your Hugging Face API token for model upload
- `RESULT_MODEL_ID`: Target Hugging Face model repository ID
- `RUN_NAME`: Name for the training run (for tracking purposes)

### Resource Configuration (`config.yaml`)

```yaml
resources:
  gpu:
    accelerator_count: 1
    accelerator: H100 
  memory: 128
  cpu: 4
replicas: 1
config_map:
  HF_TOKEN: <your hf token with write creds>
  RESULT_MODEL_ID: <your repo id>
  RUN_NAME: <run name to be created in OICM>
```

### Training Configuration

The current setup uses:
- **Base Model**: `bert-base-uncased`
- **Dataset**: GLUE SST-2 (Stanford Sentiment Treebank)
- **Task**: Binary sentiment classification
- **Epochs**: 1 (configurable in `main.py`)
- **Batch Size**: 64 per device (configurable)

### LoRA Configuration

The LoRA setup includes:
- **Rank (r)**: 8
- **Alpha**: 16
- **Target Modules**: Query and value projection layers
- **Dropout**: 0.05
- **Task Type**: Sequence classification

## Model Details

### Architecture
- **Backbone**: BERT-base-uncased
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 8-bit with BitsAndBytesConfig
- **Task**: Binary sentiment classification (positive/negative)

### Performance Optimizations
- **Gradient Checkpointing**: Enabled to reduce memory usage
- **8-bit Quantization**: Reduces model size and memory requirements
- **Parameter-Efficient Training**: Only ~0.3% of parameters are trainable with LoRA

## Output

After training completion:
1. Model files are saved to `./output/` directory
2. Model is automatically uploaded to the specified Hugging Face repository
3. Training metrics and logs are available through the OIP tracking system
4. Repository URL is logged for reference

## Dependencies

- `transformers==4.56.2`: Hugging Face transformers library
- `datasets==4.1.1`: Dataset loading and processing
- `accelerate==1.10.1`: Distributed training support
- `peft==0.17.1`: Parameter-Efficient Fine-Tuning library
- `bitsandbytes==0.47.0`: 8-bit quantization
- `evaluate==0.4.6`: Model evaluation metrics

## Hardware Requirements

- **Recommended**: H100 GPU with 128GB memory
- **Minimum**: Any CUDA-compatible GPU with 8GB+ VRAM
- **CPU**: 4+ cores recommended
- **Memory**: 16GB+ system RAM

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `main.py` or enable gradient accumulation
2. **Hugging Face Upload Fails**: Verify `HF_TOKEN` and `RESULT_MODEL_ID` are correct
3. **Dataset Loading Issues**: Ensure internet connection for downloading GLUE dataset

### Debugging

Enable verbose logging by modifying the training arguments in `utils/training_utils.py`:
```python
logging_steps=1  # Log every step instead of every 10
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
