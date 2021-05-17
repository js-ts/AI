import torch
import torch.nn as nn 


from options import OptionsV1

class Solver(object):

    def __init__(self, options) -> None:
        super().__init__()

        pass



if __name__ == '__main__':
    
    opt = OptionsV1().parse()
    print(opt.name)

