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
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from botocore.exceptions import NoCredentialsError
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data.distributed import DistributedSampler


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


def setup():
    """Set up process group for distributed training. This function uses
    'RANK' and 'WORLD_SIZE' that should be automatically set by `torchrun`.

    Returns:
        int: rank of the current process.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    LOGGER.info(f"Setting up rank {rank}, world size: {world_size}")
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.barrier()
    return rank


def cleanup():
    """Destroy process group."""
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 on CIFAR-10 using FSDP"
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="initial learning rate",
    )
    args = parser.parse_args()
    return args


def train(model, train_loader, loss_fn, optimizer, epoch):
    """Train model."""
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        msg = f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}"
        LOGGER.info(msg)


# Test function
def test(model, test_loader, loss_fn):
    """Evaluate model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss = test_loss / len(test_loader)
    acc = correct / len(test_loader.dataset)
    LOGGER.info(f"Test Loss: {test_loss}, Accuracy: {acc}")


def print_gpu_info():
    LOGGER.info("=" * 100)
    if torch.cuda.is_available():
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        LOGGER.info("No GPU found")
    
    LOGGER.info("=" * 100)


def main():


    date = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    id_ = f"rank-{os.environ.get('RANK')}"
    log_file = f"{date}_{id_}_logs.txt"
    setup_logger(id_=f"{id_}", file=log_file)

    print_gpu_info()

    args = parse_args()
    rank = setup()

    batch_size = int(os.environ.get("BATCH_SIZE", args.batch_size))
    epochs = int(os.environ.get("EPOCHS", args.epochs))
    lr = float(os.environ.get("LEARNING_RATE", args.lr))

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_sampler = DistributedSampler(trainset)
    train_loader = data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = data.DataLoader(testset, batch_size=100, shuffle=False)

    model = models.resnet18(num_classes=10)
    model = model.cuda()
    model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    msg = f"Training for {epochs} epochs, lr: {lr}, batch_size: {batch_size}"
    LOGGER.info(msg)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train(model, train_loader, loss_fn, optimizer, epoch)
        test(model, test_loader, loss_fn)

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
            model_name = f"{date}_{id_}_resnet18-chckpt-FSDP.pt"
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
    main()