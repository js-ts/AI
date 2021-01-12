import torch
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer

import os
import random

NUM = 1000
DIM = 64


class EmbeddingPS(torch.nn.Module):
    def __init__(self, num, dim):
        super().__init__()
        self.embed = torch.nn.EmbeddingBag(num_embeddings=num,embedding_dim=dim, mode='sum')
        
    def forward(self, indics, offset):
        ''' '''
        return self.embed(indics, offset).cpu()


def _retrieve_embedding_parameters(emb_rref):
    param_rrefs = []
    for param in emb_rref.local_value().parameters():
        param_rrefs.append(rpc.RRef(param))
    return param_rrefs



class Model(torch.nn.Module):
    def __init__(self, embed_rref, device_id):
        super().__init__()
        self.embed_rref = embed_rref
        self.device_id = device_id
        
        fc = torch.nn.Linear(DIM, DIM).cuda(device_id)
        self.fc = torch.nn.parallel.DistributedDataParallel(fc, device_ids=[device_id])

    def forward(self, indics, offset):
        emb_lookup = self.emb_rref.rpc_sync().forward(indics, offset)
        return self.fc(emb_lookup.to(self.device_id))
        



def _worker(rref, rank):
    '''
    '''
    num_indices = random.randint(20, 50)
    indices = torch.LongTensor(num_indices).random_(0, NUM)

    offsets = []
    start = 0
    while start < num_indices:
        offsets.append(start)
        start += random.randint(1, 10)
    offsets = torch.LongTensor(offsets)
    
    print(rank, indices.shape, offsets.shape)

    # model = Model(rref, rank)
    output = rref.rpc_sync().forward(indices, offsets) 
    print(rank, output.shape)

    params_rref = rpc.rpc_sync(to='ps', func=_retrieve_embedding_parameters, args=(rref, ))
    # params_rref.extend([dist.rpc.RRef(p) for p in model.parameters()])
    
    opt = dist.optim.DistributedOptimizer(torch.optim.Adam, params_rref, lr=0.01)
    # criterion = torch.nn.CrossEntropyLoss()

    for e in range(5):
        for indices, offsets in [(indices, offsets), ]:
            with dist.autograd.context() as context_id:
                output = rref.rpc_sync().forward(indices, offsets)
                # output = model(indices, offsets)
                loss = output.sum()
                dist.autograd.backward(context_id, [loss])
                opt.step(context_id)
        
        print(rank, e, loss.item())


def run_worker(rank, world_size, ):
    '''
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '50000'
    
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    # rpc_backend_options.num_worker_threads = 8
    rpc_backend_options.init_method = 'tcp://localhost:50001'
    # rpc_backend_options.rpc_timeout = 2

    if rank == 3:
        '''master''' 
        rpc.init_rpc('master', rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc_backend_options) # ,

        embed_rref = rpc.remote(to='ps', func=EmbeddingPS, args=(NUM, DIM), )

        _ps = []
        for trainer_rank in [0, 1]:
            trainer_name = f'trainer{trainer_rank}'
            _p = rpc.rpc_async(to=trainer_name, func=_worker, args=(embed_rref, trainer_rank))
            _ps.append(_p)
        
        for _p in _ps:
            _p.wait()
        
    elif rank in [0, 1]:
        '''ddp worker'''
        dist.init_process_group('gloo', rank=rank, world_size=2)
        rpc.init_rpc(f'trainer{rank}', rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc_backend_options)
        pass 

    else:
        '''ps'''
        rpc.init_rpc('ps', rank=rank, world_size=world_size, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc_backend_options)
        pass 
        
    # dist.destroy_process_group()
    rpc.shutdown()


if __name__ == '__main__':
    '''
    '''
    world_size = 4
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
