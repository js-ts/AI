
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
    
        
class DynamicHeadBlock(nn.Layer):
    def __init__(self, ):
        super().__init__()
        
        l = 3
        c = 8
        
        self.mid_idx = l // 3
        
        self.l_attention = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                        nn.Conv2D(l, l, 1, 1), 
                                        nn.ReLU(), 
                                        HardSigmoid(), )
        
        self.offset_conv = nn.Conv2D(c, 2 * 3 * 3, 3, 1, 1)
        self.weight_conv = nn.Sequential(nn.Conv2D(c, 3 * 3, 3, 1, 1), nn.Sigmoid())
        self.deform_conv = paddle.vision.ops.DeformConv2D(c, c, 3, 1, 1)
        
        # init.reset_initialized_parameter(self)
        
    def forward(self, feat):
        '''
        feat [N, L, C, H, W]
        '''
        n, l, c, h, w = feat.shape
        
        feat = feat.reshape([n, l, c, -1])
        feat = self.l_attention(feat) * feat
        feat = feat.reshape([n, l, c, h, w])
        
        offset = self.offset_conv(feat[:, self.mid_idx])
        weight = self.weight_conv(feat[:, self.mid_idx])
        
        sptials = []
        for i in range(l):
            sptials.append(self.deform_conv(feat[:, i], offset, mask=weight))
            
        sptials = paddle.concat([s.unsqueeze(0) for s in sptials], axis=0)
        sptials = sptials.mean(axis=0)
        
        print(sptials.shape)
        
        return feat
        
        
class DynamicHead(nn.Layer):
    pass



if __name__ == '__main__':
    
    m = DynamicHeadBlock()
    data = paddle.rand([1, 3, 8, 3, 3])
    m(data).sum().backward()
    