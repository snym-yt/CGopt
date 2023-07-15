# -*- coding: utf-8 -*-
from os import path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave,imread
from skimage.transform import resize
import torch
from torch import nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
mse = nn.MSELoss()


## 画像変換器 ##
img_path = path.join(path.dirname(__file__), "test/input/8_256.jpg")
img_load = cv2.imread(img_path)
I_t = cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)
[X_N,Y_N] = np.shape(I_t)

def Gauss_Saidel(u, d_x, d_y, b_x, b_y, I_t, MU, LAMBDA):
    U = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) + np.hstack([np.reshape(u[0,:],[Y_N,1]), u[:,0:Y_N-1]]) + np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) + np.vstack([np.reshape(u[:,0],[1,X_N] ), u[0:X_N-1,:]])
    D = np.vstack([np.reshape(d_x[:,0],[1,X_N] ), d_x[0:Y_N-1,:]]) - d_x + np.hstack([np.reshape(d_y[0,:],[Y_N,1] ), d_y[:,0:X_N-1]]) - d_y
    B = -np.vstack([np.reshape(b_x[:,0],[1,X_N] ), b_x[0:Y_N-1,:]]) + b_x - np.hstack([np.reshape(b_y[0,:],[Y_N,1] ), b_y[:,0:X_N-1]]) + b_y
    G = LAMBDA/(MU + 4*LAMBDA)*(U+D+B) + MU/(MU + 4*LAMBDA)*I_t
    return G

def shrink(x,y):
    t = np.abs(x) - y
    S = np.sign(x)*(t > 0) * t
    return S

def main():
    ## Load Image

    CYCLE = 100
    MU = 5.0*10**(-2)
    LAMBDA = 1.0*10**(-2)
    TOL = 5.0*10**(-1)

    ## Initialization
    u = I_t
    d_x = np.zeros([X_N,Y_N])
    d_y = np.zeros([X_N,Y_N])
    b_x = np.zeros([X_N,Y_N])
    b_y = np.zeros([X_N,Y_N])

    for cyc in range(CYCLE):
        u_n = Gauss_Saidel(u,d_x,d_y, b_x ,b_y,I_t, MU,LAMBDA)
        Err = np.max(np.abs(u_n[2:X_N-2,2:Y_N-2] - u[2:X_N-2,2:Y_N-2]))
        if np.mod(cyc,10)==0:
            print([cyc,Err])
        if Err < TOL:
            break
        else:
            u = u_n
            nablax_u = np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) - u 
            nablay_u = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) - u 
            d_x = shrink(nablax_u + b_x, 1/LAMBDA)
            d_y = shrink(nablay_u + b_y, 1/LAMBDA)
            b_x = b_x + (nablax_u - d_x)
            b_y = b_y + (nablay_u - d_y)

    teach_img = cv2.imread(path.join(path.dirname(__file__), "test/teach/8_256.jpg"))
    teach_img = cv2.cvtColor(teach_img, cv2.COLOR_RGB2GRAY)
    loss = mse(torch.tensor(teach_img),torch.tensor(u)).item()/(255**2)
    psnr = 10*np.log10((1**2)/loss).item()
    print(loss)
    print(psnr)
    ## plot figure
    plt.figure(figsize=[5,4])
    plt.gray()
    plt.imshow(np.round(u))
    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), "save/denoised_image_TV.png"))
    plt.close()

if __name__ == "__main__":
    main()
