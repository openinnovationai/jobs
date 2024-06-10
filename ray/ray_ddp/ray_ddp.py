import logging.config
import os

LOGGING_CONF = {"version": 1,
                "formatters": {
                    "simple": {"format": "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"}},
                "handlers": {
                    "console": {"class": "logging.StreamHandler", "formatter": "simple", "stream": "ext://sys.stdout"}},
                "root": {"level": "INFO", "handlers": ["console"]}

                }

logging.config.dictConfig(LOGGING_CONF)

import ray.train.torch

import torch
import torch.nn as nn
from ray.air import RunConfig
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig


def get_dataset():
    return datasets.FashionMNIST(root="/tmp/data", train=True, download=True, transform=ToTensor(), )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                               nn.Linear(512, 10), )

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits


def train_func_distributed():
    num_epochs = 50
    batch_size = 64
    print("Downloading DS")
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)
    print("Downloading DS - Done")

    model = NeuralNetwork()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)
        count = 0
        loss = 0
        for inputs, labels in dataloader:
            count += 1
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f"Total dataloader sise {count}")
        print(f"epoch: {epoch}, loss: {loss.item()}")
        print("Sleeping")


if __name__ == "__main__":
    info_abot_init = ray.init()
    bucket = os.environ.get("S3_BUCKET", "sandbox-ray")
    config = RunConfig(storage_path=f"s3://{bucket}/ray_ddp")
    trainer = TorchTrainer(train_func_distributed, run_config=config,
                           scaling_config=ScalingConfig(num_workers=2, use_gpu=True))

    results = trainer.fit()
