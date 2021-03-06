
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from .. import initializer as int



class HardSigmoid(nn.Layer):
    def __init__(self, ):
        super().__init__()
        pass
    
    def forward(self, x):
        x = paddle.minimum(paddle.ones_like(x), (x + 1) / 2)
        x = paddle.maximum(paddle.zeros_like(x), x)
        return x
    
    
class ShiftedSigmoid(nn.Layer):
    '''shift [-1, 1]
    '''
    def __init__(self, ):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)
        return 2 * x - 1
        
        
class DynamicHeadBlock(nn.Layer):
    def __init__(self, levels=3, channels=8, dim=128):
        super().__init__()
        
        L = levels
        C = channels
        dim = dim
        k = 3
        
        self.k = k
        self.mid_idx = L // 2
        
        self.l_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                        nn.Conv2D(L, L, 1, 1), 
                                        nn.ReLU(), 
                                        HardSigmoid(), )
        
        # self.offset_conv = nn.Conv2D(C, 2 * 3 * 3, 3, 1, 1)
        # self.weight_conv = nn.Sequential(nn.Conv2D(C, 3 * 3, 3, 1, 1), nn.Sigmoid())
        self.offset_conv = nn.Conv2D(C, 2 * k * k + k * k, 3, 1, 1)
        
        self.deform_convs = nn.LayerList([paddle.vision.ops.DeformConv2D(C, C, 3, 1, 1) for i in range(L)])
        
        self.c_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1), 
                                         nn.Flatten(),  
                                         nn.Linear(C, dim), 
                                         nn.ReLU(), 
                                         nn.Linear(dim, C),
                                         nn.LayerNorm(C),
                                         ShiftedSigmoid(), )
        
        # init.reset_initialized_parameter(self)
        
    def forward(self, feat):
        '''
        feat [N, L, C, H, W]
        '''
        n, l, c, h, w = feat.shape
        
        # layer
        feat = feat.reshape([n, l, c, -1])
        feat = self.l_attention(feat) * feat
        feat = feat.reshape([n, l, c, h, w])
        
        
        # spatial
        # offset = self.offset_conv(feat[:, self.mid_idx])
        # weight = self.weight_conv(feat[:, self.mid_idx])
        
        _offset = self.offset_conv(feat[:, self.mid_idx])
        weight = F.sigmoid(_offset[:, :self.k * self.k])
        offset = _offset[:, self.k * self.k:]
                           
        sptials = []
        for i in range(l):
            sptials.append(self.deform_convs[i](feat[:, i], offset, mask=weight))

        feat = paddle.concat([s.unsqueeze(0) for s in sptials], axis=0).mean(axis=0)
        

        # channel 
        feat = self.c_attention(feat).unsqueeze([2, 3]) * feat
        print(feat.shape)
        
        return feat
        
        
class DynamicHead(nn.Layer):
    pass



if __name__ == '__main__':
    
    m = DynamicHeadBlock()
    data = paddle.rand([1, 3, 8, 3, 3])
    m(data).sum().backward()
    