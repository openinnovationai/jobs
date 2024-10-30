import os
import time 

import torch
import torch.distributed as dist


SHARED_VOLUME_PATH = "/home"


def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, global_rank, world_size


def write_file(filename: str, content: str) -> None:
    """
    Writes content to a file in the shared volume path.

    Args:
        filename (str): Name of the file to write
        content (str): Content to write to the file
    """
    try:
        with open(f"{SHARED_VOLUME_PATH}/{filename}", "w") as file:
            file.write(content)
    except IOError as e:
        raise Exception(f"Error writing to file: {e}")


def read_file_content(filename: str) -> str:
    """
    Reads content from a file in the shared volume path.

    Args:
        filename (str): Name of the file to read

    Returns:
        str: Content of the file

    Raises:
        Exception: If file cannot be read
    """
    try:
        with open(f"{SHARED_VOLUME_PATH}/{filename}", "r") as file:
            return file.read()
    except IOError as e:
        raise Exception(f"Error reading file: {e}")


def train(global_rank):
    filename = "test.txt"
    # write a file by the master
    if global_rank == 0:
        write_file(filename=filename, content="This is written from the master")

    dist.barrier()  # workers wait for the master to finish

    # read the file content from the worker
    if global_rank != 0:
        file_content = read_file_content(filename=filename)
        print(f"GLOBAL_RANK: {global_rank}, file_content: {file_content}")


def main():
    rank, local_rank, global_rank, world_size = setup_distributed()

    print(
        f"RANK: {rank}, LOCAL_RANK: {local_rank}, GLOBAL_RANK: {global_rank}, WORLD_SIZE: {world_size}"
    )

    train(global_rank)

    time_before_remove = 60
    for i in range(time_before_remove):
        # log in the master
        if global_rank == 0:
            print(f"removing job in ({time_before_remove - i})s")
        time.sleep(1)


if __name__ == "__main__":
    main()
