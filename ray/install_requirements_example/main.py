import time
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import pandas


def train_func():
    print("=" * 100)
    print(torch.__version__)
    print(pandas.__version__)
    print("=" * 100)

    print("Running train_func")
    for i in range(100):
        print(f"Iteration {i}")
        time.sleep(2)
    print("Running train_func - Done")


if __name__ == "__main__":
    trainer = TorchTrainer(train_func, scaling_config=ScalingConfig(num_workers=1))
    trainer.fit()
