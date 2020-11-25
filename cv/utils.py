import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image


def fig2data(fig):
    '''matplotlib figure to data
    '''
    fig.canvas.draw() # draw the renderer
    w, h = fig.canvas.get_width_height() # Get the RGBA buffer from the figure
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())

    return image.convert('RGB')

def test_fig2data(output='test.jpg'):
    '''test'''
    fig = plt.figure()
    x = np.random.rand(100)
    y = np.cos(x)
    plt.scatter(x, y)

    image = fig2data(fig)
    with open(output, 'w') as f:
        image.save(f, format='JPEG')
     
     
if __name__ == '__main__':

    test_fig2data()
    
    
