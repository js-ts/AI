import numpy as np 
from PIL import Image
import matplotlib as plt
import matplotlib.cm 


def save_color_depth(disp, path):
    # Saving colormapped depth image
    disp = disp if isinstance(disp, np.ndarray) else disp.squeeze().cpu().detach().numpy()
    
    vmax = np.percentile(disp, 95)
    normalizer = plt.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)

    im.save(path)

