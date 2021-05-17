import torch
import torch.nn.functional as F 


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
    
    R = I  + A * torch.sin(angle).unsqueeze(-1) + torch.matmul(A, A) * (1 - torch.cos(angle)).unsqueeze(-1)

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
        translate = -1 * translate # here dont using in-space operation

    T = translate_to_matrix(translate)

    if invert:
        return torch.matmul(R, T)
    else:
        return torch.matmul(T, R)


def reprojection(im, proj, padding_mode="border"):
    '''
    im [n, 3, h, w]
    proj [n, 2, h, w],  (0, 0) left-top (w-1, h-1) right-bottom
    '''
    grid = proj.permute(0, 2, 3, 1)
    grid[..., 0] / proj.shape[3] - 1
    grid[..., 0] / proj.shape[2] - 1
    grid = (grid - 0.5) * 2 # (-1, -1) left-top (1, 1) right-bottom
    
    return F.grid_sample(im, grid, padding_mode=padding_mode, align_corners=False)


def depth_metrics(pred, gt):
    '''
    '''
    # pred *= torch.median(gt) / torch.median(pred)
    # pred = torch.clamp(pred, min=1e-3, max=80)

    thr = torch.max(pred / gt, gt / pred)
    a1 = (thr < 1.25 ** 1).float().mean()
    a2 = (thr < 1.25 ** 2).float().mean()
    a3 = (thr < 1.25 ** 3).float().mean()

    rms = ((gt - pred) ** 2).mean().sqrt()
    rms_log = ((gt.log() - pred.log()) ** 2).mean().sqrt()

    rel_abs = (torch.abs(gt - pred) / gt).mean()
    rel_sq = ((gt - pred) ** 2 / gt).mean()

    return rel_abs, rel_sq, rms, rms_log, a1, a2, a3 

