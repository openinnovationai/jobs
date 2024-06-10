import os

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.air import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

GROUP = 2
PROCESS_GROUP = 2


# Define a synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, data_size, image_size):
        self.data_size = data_size
        self.image_size = image_size
        self.data = torch.randn(data_size, 3, image_size, image_size)
        self.targets = torch.randint(0, 10, (data_size,))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main():
    image_size = 28
    batch_size = 64
    num_epochs = 150
    data_size = 5000
    print("Preparing datasets")
    dataset = SyntheticDataset(data_size, image_size)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    print("Preparing model")
    model = Net()

    mesh_2d = init_device_mesh("cuda", (GROUP, PROCESS_GROUP))

    model = ray.train.torch.prepare_model(model, parallel_strategy="fsdp",
                                          parallel_strategy_kwargs={"device_mesh": mesh_2d,
                                                                    "sharding_strategy": ShardingStrategy.HYBRID_SHARD})

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Training")
    for epoch in range(num_epochs):
        model.train()
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    print("Training - Done")


if __name__ == "__main__":
    print("Init")
    bucket = os.environ.get("S3_BUCKET", "sandbox-ray")
    config = RunConfig(storage_path=f"s3://{bucket}/ray_fsdp", checkpoint_config=CheckpointConfig(num_to_keep=1))
    trainer = TorchTrainer(main, run_config=config,
                           scaling_config=ScalingConfig(num_workers=GROUP * PROCESS_GROUP, use_gpu=True))

    print("Init - Done")
    print("Training")
    results = trainer.fit()
    print("Training - Done")
