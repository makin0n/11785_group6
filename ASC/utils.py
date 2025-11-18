
import torch 
import os
import torch.distributed as dist


def setup_ddp():
    """Initialize DDP if running in distributed mode."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set CUDA device BEFORE initializing the process group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Ensure rendezvous address/port exist (torchrun usually provides these)
        os.environ.setdefault('MASTER_ADDR', os.environ.get('MASTER_ADDR', '127.0.0.1'))
        os.environ.setdefault('MASTER_PORT', os.environ.get('MASTER_PORT', '29500'))

        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        
        # Verify initialization was successful
        if dist.is_initialized():
            print(f"DDP initialized successfully: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        return True, rank, world_size, local_rank
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0