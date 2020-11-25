import numpy as np 
from sklearn.decomposition import PCA


a = np.random.rand(10, 100)

b = PCA().fit_transform(a)[:, :2]

print(b.shape)
