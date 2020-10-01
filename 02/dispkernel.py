import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

# Function to display, in gray scale, the weights in a grid
# the parameter 'kernel' contains ksize*ksize weights to display
# isize is the number of pixels on one dimentions of the square image to be displayed
# that is, the image to be displayed is isize * isize pixels
# NOTE THAT isize *must be divisable* by ksize

def dispKernel(kernel, ksize, isize):
    # for normalizing
    kmax = max(kernel)
    kmin = min(kernel)
    spread = kmax - kmin
    # print("max,min",kmax,kmin)

    dsize = int(isize / ksize)
    # print("dsize:",dsize)

    a = np.full((isize, isize), 0.0)

    # loop through each element of kernel
    for i in range(ksize):
        for j in range(ksize):
            # fill in the image for this kernel element
            basei = i * dsize
            basej = j * dsize
            for k in range(dsize):
                for l in range(dsize):
                    a[basei + k][basej + l] = (kernel[(i * ksize) + j] - kmin) / spread

    # print(a)

    x = np.uint8(a * 255)

    # print(x)
    img = Image.fromarray(x, mode='P')
    imshow(img, cmap='Greys_r')
    plt.show()