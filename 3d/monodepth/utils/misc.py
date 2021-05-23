
import torch
import torch.distributed as dist
import torch.utils.data as data

import os
import random
import numpy as np 

# https://pytorch.org/docs/stable/distributed.html#launch-utility
# python -m torch.distributed.launch --nproc_per_node=4 main.py --
# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --

def setup_distributed(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed(args):
    '''
    '''
    # --use_env
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(int(os.environ['WORLD_SIZE']))
        args.gpu = int(os.environ['LOCAL_RANK'])
        
        if hasattr(args, 'local_rank'):
            # --local_rank
            assert args.local_rank == int(os.environ['LOCAL_RANK']), ''
            
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        
    else:
        print('Not using distributed...')
        args.distributed = False
        return 

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    args.dist_backend = 'nccl'
    args.dist_url = 'env://'

    dist.init_process_group(args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    # setup_distributed(args.rank == 0)



def get_world_size():
    '''
    '''
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    '''
    '''
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_master_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_master_process():
        torch.save(*args, **kwargs)


def set_random_seed(seed):
    '''
    fix the seed for reproducibility
    '''
    seed = seed + get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed



def build_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=False, distribued=False):
    '''
    world_size = 4, batch_size = 3
    0 1 2 3, 4 5 6 7, 8 9 10 11, 12 13 14 15
    process-0 [0, 4, 8]
    process-1 [1, 5, 9]
    ...
    process-0 [12]
    process-1 [13]
    '''
    if distribued:
        sampler = data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)
    
    # sampler = data.DistributedSampler(dataset, shuffle=shuffle) if distribued else None
    if True:
        batch_sampler = data.BatchSampler(sampler, batch_size, drop_last=drop_last)
        dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    else:
        # sampler option is mutually exclusive with shuffle
        shuffle = shuffle if not distribued else None
        dataloader = data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)

    return dataloader, sampler

    