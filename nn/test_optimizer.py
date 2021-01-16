import numpy as np 
import sympy
import matplotlib.pyplot as plt


def func(x, w, b):
    '''linear
    '''
    return x * w + b

X = sympy.Symbol('X')
W = sympy.Symbol('W')
B = sympy.Symbol('B')
T = sympy.Symbol('T')
Y = func(X, W, B)
L = (Y - T) ** 2
W_Grad = L.diff(W)
B_Grad = L.diff(B)
Grads = [L.diff(p) for p in [W, B]]

n_samples = 1000
data = (np.random.rand(n_samples) - 0.5) * 10
labl = func(data, 2, 10)

w = 1.
b = 0.
params = [w, b]
lr = 1e-2
n_inters = 1

for _ in range(n_inters):
    for i in range(n_samples):

        # L = (Y - labl[i]) ** 2
        # w_grad = L.diff(W).subs({X: data[i], W: w, B: b})
        # b_grad = L.diff(B).subs({X: data[i], W: w, B: b})
        # loss = L.subs({X: data[i], W: w, B: b})

        loss = L.subs({X: data[i], T: labl[i], W: w, B: b})
        w_grad = W_Grad.subs({X: data[i], T: labl[i], W: w, B: b})
        b_grad = B_Grad.subs({X: data[i], T: labl[i], W: w, B: b})
        w -= lr * w_grad 
        b -= lr * b_grad

        # grads = [Grad.subs({X: data[i], T: labl[i], W: w, B: b}) for Grad in Grads]
        # for i in range(len(params)):
        #     params[i] -= lr * grads[i]
        
        print(loss, w, b)   


plt.figure(figsize=(10, 5), dpi=100)
plt.subplot(121)
plt.title('origin')
plt.scatter(data, labl, c='b')
plt.subplot(122)
plt.title('new')
plt.scatter(data, func(data, w, b), c='r')
plt.show()



"""
def func(w):
    '''any function
    '''
    return w ** 2

def derive(w):
    '''
    '''
    return 2 * w

start = 6.5
lr = 0.01


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot()
ax.set_xlabel('w')
ax.set_ylabel('L')

ln, = ax.plot([], [], 'ro', animated=True)
ws, ls = [], []

def init():
    '''
    '''
    x = np.linspace(-7, 7, 100)
    y = func(x)
    ax.plot(x, y)
    ax.scatter([0], [func(0)], c='k', )
    return ln, 

def update(frame):
    '''
    '''
    ws.append(frame)
    ls.append(func(frame))
    ln.set_data(ws, ls)
    return ln, 


frames = [start]
for i in range(500):
    w = frames[-1] - lr * derive(frames[-1])
    if abs(w - frames[-1]) < 1e-5:
        print(w)
        break
    frames.append(w)

anim = FuncAnimation(fig, update, frames=frames, interval=150, init_func=init, blit=True)
anim.save('test_animation.gif', )
"""
