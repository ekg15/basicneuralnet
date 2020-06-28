import numpy as np
from Node import *
from Layer import *
# error = upsample(Wkl * error(l+1)) * f'(zkl)

# per weight:
# dError/d_wlmn = sum across feature map of: change in error w.r.t. x (elem of feat map) times change in x wrt w_lmn
# change in error w.r.t. x_mn in feat map is normal grad if densely connected, else if filtered again, like this.
# change in x_mn w.r.t. w_lmn is additive, so it's the activation of l-1, likely a component of the image.


def convolve():
    # Needs to:
    # take in array of values (image)
    # perform convolution
    # output feature map
    layer1 = Layer()
    layer1.generateInputNodes(16)
    k1 = np.random.rand(5, 5)
    f1 = Filter(5, k1)
    f1.applyGradient(layer1)
    pass


class ConvolutionLayer:
    def __init__(self, previousLayer=None, nextLayer=None, filters=None):
        # list of all kernels for feature finding that will be applied to input
        # array of filter class
        if filters is None:
            filters = []
        self.filters = filters
        # array of numpy matricies
        self.maps = []
        self.pooledmaps = []
        # either convolution layer or image (None)
        self.previousLayer = previousLayer
        # either densely connected or convolution
        self.nextlayer = nextLayer

    def runFilters(self, image):
        # create feature maps
        pass

    def poolMaps(self):
        # apply pooling method to feature maps
        pass

    def backprop(self):
        # all filters apply grads, give proper info
        pass


class Filter:
    def __init__(self, n, kernel=None):
        # matrix of weights
        if kernel is None:
            kernel = []
        self.kernel = kernel
        self.currentFeatMap = []
        self.n = n

    def convolveToFeatMap(self, image):
        # return a i-nxi-n feature map
        # i = len image, n = len featmap
        pass

    def applyGradient(self, nextLayer):
        # calculate gradient for all n^2 weights in a kernel
        # delta_L1 = nextLayer.calculatePartialsInner()
        with np.nditer(self.kernel, flags=['multi_index']) as itr:
            for i in itr:
                pass
                print(i, itr.multi_index)
                # dE/dx * dx/dw
                # dE/dx = grad of next layer (delta_L1)
                # dx/dw = value input to w
                # value input is given by coords of weight + coords of featmap
                # so the weight at (1,3) for x_8,9 of featmap is 9,12 in the image
        pass


if __name__ == '__main__':
    convolve()
