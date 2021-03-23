

import paddle


def assert_image(img, ):
    '''make sure img is image
    '''
    pass


def normalize(img, mean, std, inspace=True, channel_last=False):
    '''normalize image
    '''
    _mean = paddle.to_tensor(mean, place=img.place).astype(img.dtype)
    _std = paddle.to_tensor(std, place=img.place).astype(img.dtype)

    if not channel_last:
        _mean = _mean.reshape((-1, 1, 1))
        _std = _std.reshape((-1, 1, 1))
    
    return (img - _mean) / _std 


"""
mean = [1,2,3]
std = [3,2,1]

def normalize(img, mean, std, inspace=True, channel_last=False):
    '''normalize image
    '''
    _mean = paddle.to_tensor(mean, place=img.place).astype(img.dtype)
    _std = paddle.to_tensor(std, place=img.place).astype(img.dtype)

    if not channel_last:
        _mean = _mean.reshape((-1, 1, 1))
        _std = _std.reshape((-1, 1, 1))
    
    return (img - _mean) / _std 

data = paddle.randn((1, 3, 640, 640))

%%timeit
_ = normalize(data, mean, std)





mean = [1,2,3]
std = [3,2,1]

def normalize_numpy(img, mean, std, inspace=True, channel_last=False):
    '''normalize image
    '''
    _mean = np.array(mean)
    _std = np.array(std)

    if not channel_last:
        _mean = _mean.reshape((-1, 1, 1))
        _std = _std.reshape((-1, 1, 1))
    
    return (img - _mean) / _std

arr = np.random.randn(1, 3, 640, 640)

%%timeit
normalize_numpy(arr, mean, std)

"""


def rgb_to_grayscale(img, num_output_channels=1):
    '''
    '''
    assert img.shape[-3] == 3, 'image should have 3 channels.'

    rgb_weights = [0.2989, 0.5870, 0.1140]
    rgb_weights = paddle.to_tensor(rgb_weights, place=img.place).astype(img.dtype)

    img = (img * rgb_weights.reshape((-1, 1, 1))).sum(axis=-3, keepdim=True)

    if num_output_channels > 1:
        _shape = img.shape
        _shape[-3] = num_output_channels
        return img.expand(_shape)
    
    return img



"""

def rgb_to_grayscale(img, num_output_channels=1):
    '''
    '''
    assert img.shape[-3] == 3, 'image should have 3 channels.'

    rgb_weights = [0.2989, 0.5870, 0.1140]
    rgb_weights = paddle.to_tensor(rgb_weights, place=img.place).astype(img.dtype)

    img = (img * rgb_weights.reshape((-1, 1, 1))).sum(axis=-3, keepdim=True)

    if num_output_channels > 1:
        _shape = img.shape
        _shape[-3] = num_output_channels
        return img.expand(_shape)
    
    return img

data = paddle.randn((1, 3, 640, 640))

%%timeit
_ = rgb_to_grayscale(data)




"""



# ------------
import math
import paddle
from paddle.nn.functional import affine_grid, grid_sample
import time


def _affine_grid(theta, w, h, ow, oh):
    '''
    '''
    d = 0.5
    # tic = time.time()
    base_grid = paddle.ones((1, oh, ow, 3), dtype=theta.dtype)
    # print(time.time() - tic)
    
    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)

    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid
    
    scaled_theta = theta.transpose((0, 2, 1)) / paddle.to_tensor([0.5 * w, 0.5 * h])
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta)
    
    return output_grid.reshape((1, oh, ow, 2))



def rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=None, translate=None,):
    '''
    '''
    angle = -angle % 360

    n, c, h, w = img.shape

    # image center is (0, 0) in matrix
    if translate is None:
        post_trans = [0, 0]
    else:
        post_trans = translate
    
    if center is None:
        rotn_center = [0, 0]
    else:
        rotn_center = [(p - s * 0.5) for p, s in zip(center, [w, h])]
    
    angle = -math.radians(angle)
    matrix = [math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0,]
    
    matrix = paddle.to_tensor(matrix, place=img.place)
    
    matrix[2] += matrix[0]*(-rotn_center[0]-post_trans[0]) + matrix[1]*(-rotn_center[1]-post_trans[1])
    matrix[5] += matrix[3]*(-rotn_center[0]-post_trans[0]) + matrix[4]*(-rotn_center[1]-post_trans[1])
    
    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]
    
    if expand:
        # calculate output size
        corners = paddle.to_tensor([[-0.5 * w, -0.5 * h, 1.0],
                                    [-0.5 * w, 0.5 * h, 1.0],
                                    [0.5 * w, 0.5 * h, 1.0],
                                    [0.5 * w, -0.5 * h, 1.0]], dtype=matrix.dtype, place=img.place)
        
        _pos = corners.reshape((1, 4, 3)).bmm(matrix.reshape((1, 2, 3)).transpose((0, 2, 1))).reshape((1, 4, 2))
        min_val = _pos.min(axis=-2).floor()
        max_val = _pos.max(axis=-2).ceil()
        
        npos = max_val - min_val
        nw = npos[0][0]
        nh = npos[0][1]
        
        ow, oh = int(nw.numpy()[0]), int(nh.numpy()[0])
        
    else:
        ow, oh = w, h
    
    m = matrix.reshape((1, 2, 3))
    # tic = time.time()
    # grid = affine_grid(m, (n, c, h, w))
    grid = _affine_grid(m, w, h, ow, oh)

    out = grid_sample(img, grid, mode=interpolation)
    
    return out




# -------------

import math
from paddle.nn.functional import grid_sample, affine_grid


def _compute_inverse_affine_matrix(center, angle, translate, scale, shear):
    '''
    '''
    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def _compute_output_size(matrix, w: int, h: int):
    '''
    '''
    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    pts = paddle.to_tensor([
            [-0.5 * w, -0.5 * h, 1.0],
            [-0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, -0.5 * h, 1.0],])
    theta = paddle.to_tensor(matrix).reshape((1, 2, 3))
    new_pts = pts.reshape((1, 4, 3)).bmm(theta.transpose((0, 2, 1))).reshape((4, 2))
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = paddle.ceil((max_vals / tol).trunc_() * tol)
    cmin = paddle.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def _affine_grid(theta, w: int, h: int, ow: int, oh: int) :
    '''
    '''
    d = 0.5
    base_grid = paddle.empty((1, oh, ow, 3), dtype=theta.dtype)
    
    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow,)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh,).unsqueeze_(-1)
    base_grid[..., 1] = y_grid 
    base_grid[..., 2] = 1

    rescaled_theta = theta.transpose((0, 2, 1)) / paddle.to_tensor([0.5 * w, 0.5 * h])
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(rescaled_theta)
    return output_grid.reshape((1, oh, ow, 2))



def _apply_grid_transform(img, grid, mode, fill):
    '''
    '''
    if img.shape[0] > 1:
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = paddle.ones((img.shape[0], 1, img.shape[2], img.shape[3]))
        img = paddle.concat((img, dummy), dim=1)

    img = grid_sample(img, grid, mode='nearest', padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = paddle.to_tensor(fill).reshape((1, len_fill, 1, 1)).expand_as(img)
        if mode == 'nearest':
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img
    
    return img


def rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=None, resample=None):
    '''
    theta -> affine_grid -> grid_sample
    '''

    h, w = img.shape[-2:]
    
    rt_center = [0.0, 0.0]
    if center is not None:
        rt_center = [1.0 * (c - s * 0.5) for c, s in zip(center, (w, h))]
    
    matrix = _compute_inverse_affine_matrix(rt_center, -angle, (0., 0.), 1., (0., 0.))

    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    ow, oh = w, h 
    matrix = paddle.to_tensor(matrix, place=img.place).astype(img.dtype).reshape((1, 2, 3))

    grid = _affine_grid(matrix, w=w, h=h, ow=ow, oh=oh)

    return _apply_grid_transform(img, grid, interpolation, fill=fill)

    


"""



In [97]: a = paddle.rand((2, 3))

In [98]: b = np.random.rand(2, 3)

In [99]: paddle.to_tensor(b, dtype=a.dtype, place=a.place)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-99-a21a932d8b3f> in <module>
----> 1 paddle.to_tensor(b, dtype=a.dtype, place=a.place)

<decorator-gen-226> in to_tensor(data, dtype, place, stop_gradient)

~/opt/anaconda3/lib/python3.8/site-packages/paddle/fluid/wrapped_decorator.py in __impl__(func, *args, **kwargs)
     23     def __impl__(func, *args, **kwargs):
     24         wrapped_func = decorator_func(func)
---> 25         return wrapped_func(*args, **kwargs)
     26 
     27     return __impl__

~/opt/anaconda3/lib/python3.8/site-packages/paddle/fluid/framework.py in __impl__(*args, **kwargs)
    223         assert in_dygraph_mode(
    224         ), "We only support '%s()' in dynamic graph mode, please call 'paddle.disable_static()' to enter dynamic graph mode." % func.__name__
--> 225         return func(*args, **kwargs)
    226 
    227     return __impl__

~/opt/anaconda3/lib/python3.8/site-packages/paddle/tensor/creation.py in to_tensor(data, dtype, place, stop_gradient)
    169 
    170     if dtype and convert_dtype(dtype) != data.dtype:
--> 171         data = data.astype(dtype)
    172 
    173     return paddle.Tensor(

TypeError: Cannot interpret 'VarType.FP32' as a data type

In [100]: paddle.to_tensor(b, place=a.place).astype(a.dtype)
Out[100]: 
Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[0.81749469, 0.64073175, 0.31098202],
        [0.37226987, 0.52068907, 0.28416878]])


"""
