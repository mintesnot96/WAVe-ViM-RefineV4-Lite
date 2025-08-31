# src/utils/dist.py
# (project root)/src/utils/dist.py

import os
import torch
import torch.distributed as dist

def init_distributed(ddp: bool):
    if ddp and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=dist_backend, init_method="env://")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return True, rank, world_size
    return False, 0, 1

def is_primary():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
