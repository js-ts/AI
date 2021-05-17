import torch


def disp_to_depth(disp, min_depth, max_depth):
    '''
    '''
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth

    _disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / _disp

    return _disp, depth
    

