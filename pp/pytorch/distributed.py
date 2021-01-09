import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 

import os 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, ) # torch.distributed.launch
parser.add_argument('--world_size', type=int, default=8) 
args = parser.parse_args()

torch.set_printoptions(precision=10)
# torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python xx.py
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


def setup(word_size, rank, baclend='gloo', init_method='env://'):
    '''setup 
    '''
    if init_method.startswith('env://'):
        '''env://'''
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'

        dist.init_process_group(backend='gloo', 
                                init_method='env://', 
                                world_size=word_size,
                                rank=rank)

    elif init_method.startswith('file://'):
        '''fil:///'''
        dist.init_process_group(backend='gloo', 
                                # init_method='file:///C:/Users/wenylv/sharedfile', 
                                init_method='file:///mnt/c/Users/wenylv/Desktop/sharedfile',
                                world_size=word_size,
                                rank=rank)

    elif init_method.startswith('tcp://'):
        '''tcp'''
        dist.init_process_group(backend='gloo', 
                                init_method='tcp://127.0.0.1:12345', 
                                world_size=word_size,
                                rank=rank)


def cleanup():
    '''cleanup
    '''
    dist.destroy_process_group()


def run(rank, world_size):
    '''process
    '''
    setup(world_size, rank)
    
    print(f'{dist.get_world_size()}, {dist.get_rank()} \n')
    print(f'args.local_rank: {args.local_rank}, rank: {rank}')

    # if dist.get_rank() == 0: pass
    # model_to_save = model.module if hasattr(model, 'module') else model
 
    cleanup()



def SpawnMain():
    '''SpawnMain
    '''
    world_size = args.world_size
    mp.spawn(fn=run, args=(world_size, ), nprocs=world_size, join=True)
    

def ProcessMain():
    '''ProcessMain
    '''
    world_size = args.world_size
    process = []
    
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size, ), )
        p.start()
        process.append(p)

    for p in process:
        p.join()
    

if __name__ == '__main__':
    # SpawnMain()
    # ProcessMain()

    # torch.distributed.launch
    # python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 pytorch_dist_mp.py --world_size=8  
    run(args.local_rank, args.world_size)



"""
# torch.set_num_threads(1)
# export OMP_NUM_THREADS=1
'''Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, 
to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.'''

if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
    current_env["OMP_NUM_THREADS"] = str(1)
    print("*****************************************\n"
          "Setting OMP_NUM_THREADS environment variable for each process "
          "to be {} in default, to avoid your system being overloaded, "
          "please further tune the variable for optimal performance in "
          "your application as needed. \n"
          "*****************************************".format(current_env["OMP_NUM_THREADS"]))
During the few hours when the job is running, did the DistributedDataParallel training making
"""
