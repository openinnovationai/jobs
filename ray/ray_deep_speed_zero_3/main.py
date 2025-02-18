import os
from tempfile import TemporaryDirectory

import deepspeed
import ray
import ray.train
import torch
from datasets import load_dataset
from deepspeed.accelerator import get_accelerator
from ray.air import RunConfig
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def train_func(config):
    """Your training function that will be launched on each worker."""
    print("Starting train_func...")
    # Unpack training configs
    set_seed(config["seed"])
    num_epochs = config["num_epochs"]
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]

    # Instantiate the Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )

    # Prepare Ray Data Loaders
    # ====================================================
    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("validation")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def collate_fn(batch):
        outputs = tokenizer(
            list(batch["sentence1"]),
            list(batch["sentence2"]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        outputs["labels"] = torch.LongTensor(batch["label"])
        return outputs

    train_dataloader = train_ds.iter_torch_batches(
        batch_size=train_batch_size, collate_fn=collate_fn
    )
    eval_dataloader = eval_ds.iter_torch_batches(
        batch_size=eval_batch_size, collate_fn=collate_fn
    )
    # ====================================================

    # Initialize DeepSpeed Engine
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    device = get_accelerator().device_name(model.local_rank)

    # Initialize Evaluation Metrics
    f1 = BinaryF1Score().to(device)
    accuracy = BinaryAccuracy().to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            f1.update(predictions, batch["labels"])
            accuracy.update(predictions, batch["labels"])

        # torchmetrics will aggregate the metrics across all workers
        eval_metric = {
            "f1": f1.compute().item(),
            "accuracy": accuracy.compute().item(),
        }
        f1.reset()
        accuracy.reset()

        if model.global_rank == 0:
            print(f"epoch {epoch}:", eval_metric)

        # Report checkpoint and metrics to Ray Train
        # ==============================================================
        with TemporaryDirectory() as tmpdir:
            # Each worker saves its own checkpoint shard
            model.save_checkpoint(tmpdir)

            # Ensure all workers finished saving their checkpoint shard
            torch.distributed.barrier()

            # Report checkpoint shards from each worker in parallel
            ray.train.report(
                metrics=eval_metric, checkpoint=Checkpoint.from_directory(tmpdir)
            )
        # ==============================================================


if __name__ == "__main__":
    print("Preparing configs")
    deepspeed_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
            },
        },
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 100}},
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},  # Turn this on if using AMPERE GPUs.
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "none",
            },
            "offload_param": {
                "device": "none",
            },
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": True,
        "steps_per_print": 10,
        "train_micro_batch_size_per_gpu": 16,
        "wall_clock_breakdown": False,
    }

    training_config = {
        "seed": 42,
        "num_epochs": 10,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "deepspeed_config": deepspeed_config,
    }

    print("Downloading dataset")
    hf_datasets = load_dataset("glue", "mrpc")
    print("Downloading dataset - Done")

    print("Init Ray DS")
    ray_datasets = {
        "train": ray.data.from_huggingface(hf_datasets["train"]),
        "validation": ray.data.from_huggingface(hf_datasets["validation"]),
    }
    print("Init Ray DS - Done")
    print("Init script")

    bucket = os.environ.get("S3_BUCKET", "sandbox-ray")
    config = RunConfig(storage_path=f"s3://{bucket}/ray_mdp")

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_config,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        run_config=config,
    )
    print("Init script - Done")
    print("Running training script")
    result = trainer.fit()
    print("Running training script - Done")
    # Retrieve the best checkponints from results
    result.best_checkpoints
