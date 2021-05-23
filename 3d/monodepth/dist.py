import argparse

import torch
import torch.nn as nn
import torch.optim as optim 

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, BatchSampler, SequentialSampler, RandomSampler


import utils.misc as misc


def build_dataloader():
    # https://pytorch.org/docs/stable/data.html?highlight=sampler#torch.utils.data.Sampler
    pass


def train_epoch(model, dataloader, optimizer, device):
    '''
    '''
    print('train.....')

    model.train()
    for _data in dataloader:

        data = torch.rand(1, 3, 100, 100).to(device)
        loss = model(data)
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        print(misc.get_rank(), _data)
        
    return 


def test_epoch(model, dataloader, device):
    print('eval.....')

    model.eval()
    for _data in dataloader:
        print(misc.get_rank(), _data)

    return 


def main(args):

    misc.init_distributed(args)
    misc.set_random_seed(args.seed)
    print(args)
    
    device = torch.device(args.device)
    
    model = nn.Conv2d(3, 8, 3)
    model.to(device)
    print(model)

    params_dicts = [{'params': model.parameters(), 'lr': args.lr * 0.1}]
    optimizer = optim.SGD(params_dicts, lr=args.lr, )
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10)

    train_dataset = range(16)
    val_dataset = range(16)

    train_dataloader, train_sampler = misc.build_dataloader(train_dataset, args.batch_size, True, args.num_workers, True, args.distributed)
    val_dataloader, _ = misc.build_dataloader(val_dataset, args.batch_size, False, args.num_workers, False, args.distributed)
    
    for e in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(e)

        train_epoch(model, train_dataloader, optimizer, device)
        
        test_epoch(model, val_dataloader, device)
        
        scheduler.step()
        
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=str, default='cuda')

    
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--lr', default=0.1, type=float)


    # distributed
    # parser.add_argument('--world_size', default=1, type=int,)
    # parser.add_argument('--local_rank', type=int,) # or --use_env
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    
    main(args)
    