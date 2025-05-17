import argparse
import logging
import os
import sys
from datetime import datetime

import boto3
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from botocore.exceptions import NoCredentialsError
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = None


def setup_logger(id_, file=None):
    """Set up a global logger.

    Args:
        id_ (str): id to show for each log file. It can be the rank if the
            current process for example.
        file (string, optional): file name to use for logging.

    Returns:
        logger: logger instance.
    """
    global LOGGER

    fmt = f"[%(asctime)s | {id_}] %(levelname)s : %(message)s"
    formatter = logging.Formatter(fmt)
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(handler)

    if file is not None:
        handler = logging.FileHandler(file, "w")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        LOGGER.addHandler(handler)
    return LOGGER


def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an s3 bucket. This function assumes
    that AWS credentials are set in environment variables.

    Args:
        file_name (str): path to the file to be uploaded.
        bucket (str): bucket name.
        object_name (str, optional): path to where to store the file on
            the bucket.

    Returns:
        bool: True if the upload succeeded, False otherwise.
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(file_name, bucket, object_name)
        return True
    except FileNotFoundError:
        LOGGER.debug(f"The file '{file_name}' was not found")
        return False
    except NoCredentialsError:
        LOGGER.debug("AWS credentials not available")
        return False


def _get_envvar_as_int(varname):
    """Convert environment variable to int.
    Return None if the variable is not set or set with
    a non-integer value.

    Args:
        varname (str): variable's name.

    Returns:
        int: variable as int or None.
    """
    value = os.environ.get(varname)

    if value is not None:
        try:
            return int(value)
        except ValueError:
            LOGGER.debug(f"'{varname}' should be an integer, found: {value}")
    return None


def setup(rank):
    """
    Initialize process group for distributed training. If launched with
    `torchrun` then the 'RANK' environment variable will be set
    automatically and should be used with `init_process_group`. In that
    case the `rank` argument will be ignored.
    if 'RANK' is not set, `rank` is used.

    'RANK' corresponds to the global rank of the process, taking into
    account all other processes, all nodes included.


    Importantly, if you run the script locally (using python your_script.py)
    environment variables WORLD_SIZE should be set to 1, MASTER_ADDR to
    "localhost" or any valid address on the machine, and MASTER_PORT to
    an available port (e.g., 12345).
    """
    env_rank = _get_envvar_as_int("RANK")
    if env_rank is not None:
        rank = env_rank

    torch.distributed.init_process_group(backend="nccl", rank=rank)
    return rank


def cleanup():
    """Destroy process group."""
    dist.destroy_process_group()


def get_data(val_size=0.2):
    """Load CIFAR-10 dataset, split train data into train and
    validation subsets.
    """

    # Data augmentation and normalization for train dataset
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    # Only apply to_tensor and normalization transforms for evaluation data
    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    # Use CIFAR-10 dataset
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    train_size = int(val_size * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_ds.transform = transform_train
    val_ds.transform = transform_val_test

    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val_test
    )
    return train_ds, val_ds, test_ds


def make_loader(dataset, batch_size=128, num_workers=1, sampler=None):
    """Make a data loader for the dataset"""
    if sampler is not None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def get_loaders(train_ds, val_ds, test_ds, batch_size=128):
    """Get loaders for train, validation and test data."""
    sampler = DistributedSampler(dataset=train_ds, shuffle=True)
    train_loader = make_loader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=8
    )
    val_loader = make_loader(val_ds, batch_size=batch_size)
    test_loader = make_loader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def get_device_id(model):
    """Get device id of the model. Returns None if th model on the CPU."""
    device = next(model.parameters()).device
    if device.type == "cuda":
        return device.index  # Get the device index
    else:
        return None  # CPU doesn't have a device index


def train(rank, train_loader, val_loader, learning_rate=0.001, epochs=1):

    device_id = torch.device("cuda:{}".format(rank))
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device_id)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10, device=device_id)

    model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for _i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device_id), labels.to(device_id)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Add some L2 regularization
            l2_lambda = 0.0001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= _i + 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device_id), labels.to(device_id)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                # Add L2 regularization
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm

                val_loss += loss.item()
        val_loss /= _i + 1
        val_acc = evaluate(model, val_loader)

        LOGGER.info(
            " ".join(
                [
                    f"Epoch [{epoch + 1}/{epochs}],",
                    f"Loss: {train_loss:.4f},",
                    f"Val loss: {val_loss:.4f},",
                    f"Val accuracy: {val_acc:.4f}%",
                ]
            )
        )
    return model


def evaluate(model, data_loader):
    device_id = get_device_id(model)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device_id), labels.to(device_id)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning a ResNet18 on CIFAR-10 using DDP"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        help="Local process rank. Must be >= 0 and < # GPUs (default=0)",
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (default=1).",
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training and evaluation (default=128).",
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (default=0.001).",
        default=0.001,
    )

    args = parser.parse_args()
    local_rank = _get_envvar_as_int("LOCAL_RANK")
    if local_rank is None:
        local_rank = args.local_rank

    rank = setup(local_rank)
    date = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    id_ = f"rank-{rank}"
    log_file = f"{date}_{id_}_logs.txt"
    setup_logger(id_=f"{id_}", file=log_file)
    msg = f"Running rank {rank}, world size: {os.environ.get('WORLD_SIZE')}"
    LOGGER.info(msg)

    batch_size = int(os.environ.get("BATCH_SIZE", args.batch_size))
    epochs = int(os.environ.get("EPOCHS", args.epochs))
    lr = float(os.environ.get("LEARNING_RATE", args.learning_rate))

    train_ds, val_ds, test_ds = get_data(val_size=0.2)
    train_loader, val_loader, test_loader = get_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    msg = f"Training for {epochs} epochs, lr: {lr}, batch_size: {batch_size}"
    LOGGER.info(msg)
    model = train(
        rank,
        train_loader,
        val_loader,
        learning_rate=lr,
        epochs=epochs,
    )
    test_acc = evaluate(model, test_loader)
    LOGGER.info(f"Test accuracy: {test_acc:.4f}%")

    bucket = os.environ.get("BUCKET_NAME")
    s3_folder = os.environ.get("FOLDER_NAME")
    if None not in (bucket, s3_folder):
        object_name = f"{s3_folder}/{log_file}"
        if upload_to_s3(log_file, bucket, object_name):
            msg = f"Log file '{log_file}' uploaded to "
            msg += f"'{bucket}/{object_name}'"
            LOGGER.info(msg)
        else:
            LOGGER.debug("Error uploading logs to S3 bucket")

        if rank == 0:
            model_name = f"{date}_{id_}_resnet18-chckpt-DDP.pt"
            torch.save(model.state_dict(), model_name)

            object_name = f"{s3_folder}/{model_name}"
            if upload_to_s3(model_name, bucket, object_name):
                msg = f"Model checkpoint '{model_name}' successfully"
                msg += f"uploaded to '{bucket}/{object_name}'"
                LOGGER.info(msg)
            else:
                LOGGER.debug("Error uploading model checkpoint to S3 bucket")
    cleanup()


if __name__ == "__main__":
    sys.exit(main())
