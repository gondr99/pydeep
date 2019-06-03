import sys, os;
import numpy as np;
from dataset.mnist import load_mnist;
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size);
        y = y.reshape(1, y.size);
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size;

def numerical_diff(f, x):
    #0.0001
    h = 1e-4; 
    return ( f(x+h) - f(x-h) ) / ( 2*h );

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad

def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)
        
        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
    
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True);

#랜덤하게 배치사이즈만큼 뽑아낸다.
train_size = x_train.shape[0];
batch_size = 10; 
batch_mask = np.random.choice(train_size, batch_size);
x_batch = x_train[batch_mask];
t_batch = t_train[batch_mask];
