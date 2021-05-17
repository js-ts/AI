import torch


def disp_to_depth(disp, min_depth, max_depth):
    '''
    '''
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth

    _disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / _disp

    return _disp, depth
    

def axisangle_to_matrix(vec):
    '''
    vec [n, 3], x * theta, y * theta, z * theta
    -> 
    4 x 4 transformation matrix

    R = I + A * sin + A^2 * (1 - cos)
    A
        0 -a3 a2
        a3 0 -a1
        -a2 a1 0
    '''
    angle = torch.norm(vec, p=2, dim=-1, keepdim=True)
    axis = vec / (angle + 1e-7)

    A = torch.zeros(vec.shape[0], 4, 4).to(dtype=vec.dtype, device=vec.device)
    I = torch.eye(4).to(dtype=vec.dtype, device=vec.device)

    for k, (j, i) in enumerate([(0, 1), (0, 2), (1, 2)]):
        A[:, j, i] = axis[:, 2-k] * -1 * (-1) ** k
        A[:, i, j] = axis[:, 2-k] * (-1) ** k

    R = I  + A * torch.sin(angle) + torch.matmul(A, A) * (1 - torch.cos(angle))

    return R


def translate_to_matrix(vec):
    '''
    vec [n, 3], 
    -> 
    4 x 4 transformation matrix
    '''
    T = torch.eye(4, 4).repeat(vec.shape[0], 1, 1).to(dtype=vec.dtype, device=vec.device)
    T[:, :-1, -1] = vec.contiguous()

    return T 


def params_to_matrix(axisangle, translate, invert=False):
    '''
    [n, 3], [n, 3] -> [n, 4, 4]
    '''
    R = axisangle_to_matrix(axisangle)

    if invert:
        R = R.permute(0, 2, 1)
        translate *= -1

    T = translate_to_matrix(translate)

    if invert:
        return torch.matmul(R, T)
    else:
        return torch.matmul(T, R)

