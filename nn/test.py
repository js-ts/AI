
import torch
import torch.nn as nn

import pyecharts
from pyecharts.charts import Timeline



model = nn.Sequential(nn.Conv2d(1,20,5), nn.ReLU(), nn.Conv2d(20,64,5), nn.ReLU())


a = 0
print(a << 1)


print(model[0])