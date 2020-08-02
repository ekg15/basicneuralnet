import numpy as np
from scipy.signal import convolve2d
from numpy.fft import *
from ImageFormatting import *
from Node import *
from Layer import *
# error = upsample(Wkl * error(l+1)) * f'(zkl)

# per weight:
# dError/d_wlmn = sum across feature map of: change in error w.r.t. x (elem of feat map) times change in x wrt w_lmn
# change in error w.r.t. x_mn in feat map is normal grad if densely connected, else if filtered again, like this.
# change in x_mn w.r.t. w_lmn is additive, so it's the activation of l-1, likely a component of the image.

# plan:
# each weight in a filter's gradient is input weight (of image) times gradients of weights it "touches"
# ex: for a weight in kernel 3 per say, all gradients of weights of nodes taking input from third map
# (if densely connected second layer, this is all of them) are summed (change in ans w resp to layer)
# and multiplied by activation touching the weight for a given map position. So for 5th input in a map,
# multiply gradient by the corresponding weight of the input image. Sum this over all elements of the
# feature map. Adjust for pooling (dirac or average) and apply to weight w.


def convolve():
    # Needs to:
    # take in array of values (image)
    # perform convolution
    # output feature map
    # BUG SOL: need to actually output a result and receive cost to generate gradients lol
    layer1 = Layer()
    layer2 = Layer()
    layer3 = Layer()
    layer1.generateInputNodes(529)
    layer2.generateInnerNodes(32, layer1)
    layer3.generateInnerNodes(32, layer2)
    layer1.dummyInputNodes()
    layer2.createWeightMatrix(layer1)
    layer3.createWeightMatrix(layer2)
    k1 = np.ones((3, 3), dtype='float')
    f1 = Filter(3, k1)
    cl = ConvolutionLayer(filters=[f1])
    cl.maps = [np.random.random((3, 3))]
    print("================================Convolution operation================================")
    # f1.convolveToFeatMap(loadMNISTData('./train-images-idx3-ubyte', './train-labels-idx1-ubyte'))
    print(f1.convolveToFeatMap(np.arange(16) + 1))
    layer1.costOverActivationPartials = f1.convolveToFeatMap(np.arange(16) + 1).flatten()
    f1.applyGradient(layer1, np.arange(16) + 1, lambda x: x)
    # print(cl.maps[0])
    # TODO: below
    # print(cl.poolMapsRedux(2))
    # print(cl.poolMaps(2, mode="max"))


class ConvolutionLayer:
    def __init__(self, inputlayer=None, nextLayer=None, filters=None):
        # list of all kernels for feature finding that will be applied to input
        # array of filter class
        if filters is None:
            filters = []
        self.filters = filters
        # array of numpy matricies
        self.maps = []
        self.pooledmaps = []
        # needs: pool, filter
        # use grabsize and knowledge of method to backprop grad to weights evenly or dirac
        self.upsample = lambda x: x
        # either densely connected or convolution
        self.nextlayer = nextLayer
        self.inputlayer = inputlayer

    # def applyGradient(self, inputLayer, nextLayer, layerLplus2, image, upsample):

    def backprop(self, image):
        # all filters apply grads, give proper info
        kernels = reduce(lambda x, y: np.vstack((x, y)), list(map(lambda x: x.kernel, self.filters)))
        self.inputlayer.calculatePartialsInputLinear(None, self.nextlayer)
        c = 0
        for f in self.filters:
            # print("=========================new filter=========================")
            # print("position: ", f.position)
            f.position = c
            f.applyGradient(self.inputlayer, image, lambda x: x)
            c += 1

    def runFilters(self, image):
        # create feature maps
        maps = []
        for f in self.filters:
            # print(f.kernel)
            # print(image)
            # x = f.convolveToFeatMap(image)
            # print(x)
            maps.append(f.convolveToFeatMap(image))
        self.maps = maps
        return maps

    def poolMaps(self, n, mode="average"):
        # apply pooling method to feature maps
        # returns an nxn pooled feature map
        # a superior method might be dividing into ceil(k^2/n^2) different groups
        pooled = []
        for fm in self.maps:
            # how many features should be combined
            grabsize = np.shape(fm)[0] - n + 1
            pm = np.ndarray((n, n))
            for i in range(n):
                for j in range(n):
                    if mode == "average":
                        print(fm[i:i + grabsize, j:j + grabsize])
                        pm[i][j] = np.average(fm[i:i + grabsize, j:j + grabsize])
                    else:
                        pm[i][j] = np.max(fm[i:i + grabsize, j:j + grabsize])
            pooled.append(pm)
        self.pooledmaps = pooled
        return pooled

    def poolMapsRedux(self, n, mode="average"):
        # apply pooling method to feature maps
        # returns an nxn pooled feature map
        # a superior method might be dividing into ceil(k^2/n^2) different groups
        # also a nice way to establish map from pooled to raw feat
        # need to make that map. Fortunately, all feat/pooled maps can use this one
        # map from: coord -> list of coords
        # (i, j) -> (i,j), ... (i + grabsize, j + grabsize)
        # for last one per row: (i, j) -> (i + grabsize, j to end of row)
        # for last one row and column: (i, j) -> (i to end of cols, j to end of row)
        # TODO: Missing case: the last featmaplen%grabsize rows are not accounted for
        # TODO: o and p are incorrect for building the pool -> raw map
        pooled = []
        poolToFeatEntryMap = {}
        c = 0
        for fm in self.maps:
            # how many features should be combined
            # ceil(n/k) x ceil(n/k) areas
            print(fm)
            featmaplen = np.shape(fm)[0]
            grabsize = math.ceil(featmaplen/n)
            # print(np.shape(fm)[0])
            # print(n)
            pm = np.ndarray((n, n))
            o = 0
            p = 0
            for i in range(0, n, grabsize):
                p = 0
                for j in range(0, n, grabsize):
                    if mode == "average":
                        print(fm[i:i+grabsize, j:j+grabsize])
                        pm[o][p] = np.average(fm[i:i+grabsize, j:j+grabsize])
                    else:
                        pm[o][p] = np.max(fm[i:i+grabsize, j:j+grabsize])
                    # a map from this pooled entry to the corresponding entries in the raw feat map
                    if c == 0:
                        poolToFeatEntryMap[(i, j)] = [(i + q, j + s) for q in range(grabsize) for s in range(grabsize)]
                    p += 1
                if mode == "average":
                    if featmaplen % grabsize != 0:
                        pm[o][p] = np.average(fm[i:i+grabsize, -(featmaplen % grabsize):])
                else:
                    if featmaplen % grabsize != 0:
                        pm[o][p] = np.max(fm[i:i+grabsize, -(featmaplen % grabsize):])
                if c == 0:
                    poolToFeatEntryMap[(o, p)] = [(o + q, p + s) for q in range(grabsize) for s in range(0, featmaplen % grabsize)]
                o += 1
            if mode == "average":
                if featmaplen % grabsize != 0:
                    pm[np.shape(pm)[0] - 1][np.shape(pm)[0] - 1] = np.average(fm[-(featmaplen % grabsize):, -(featmaplen % grabsize):])
            else:
                if featmaplen % grabsize != 0:
                    pm[np.shape(pm)[0] - 1][np.shape(pm)[0] - 1] = np.max(fm[-(featmaplen % grabsize):, -(featmaplen % grabsize):])
            if c == 0:
                poolToFeatEntryMap[(np.shape(pm)[0] - 1, np.shape(pm)[0] - 1)] = [(np.shape(pm)[0] - 1 + q, np.shape(pm)[0] - 1 + s) for q in range(featmaplen % grabsize) for s in range(0, featmaplen % grabsize)]
            pooled.append(pm)
            c += 1
        self.pooledmaps = pooled
        # print(poolToFeatEntryMap)
        return pooled



class Filter:
    def __init__(self, n, kernel=None):
        # matrix of weights
        if kernel is None:
            kernel = []
        self.kernel = kernel
        self.currGrad = []
        self.currentFeatMap = []
        self.n = n
        self.position = 0


    def convolveToFeatMap(self, image):
        # return a i-nxi-n feature map
        # i = len image, n = len featmap
        # image is 1d array
        imglength = int(math.sqrt(len(image)))
        featmapLength = imglength - self.n + 1
        featmap = np.ndarray((featmapLength, featmapLength))
        imgreshaped = np.reshape(np.array(image), (imglength, imglength))/255
        # technically correct, but the image is ill-formatted
        # not going to do FFT.
        # print("kernel: ", self.kernel)
        convkernel = np.flip(self.kernel.flatten(), 0)
        # print("convkernel: ", convkernel)
        # print(imgreshaped)
        for i in range(featmapLength):
            for j in range(featmapLength):
                featmap[i, j] = np.convolve(convkernel, imgreshaped[i:i+self.n, j:j+self.n].flatten(), mode='valid')

        # returns a convolution of the two of dimension (imglen - n + 1) x (imglen - n + 1)
        return featmap

    # Tabled: Need to get gradient to work on pooled features
    # Could use a pooled object? Perhaps a pooled feature map can be associated with a mapping of a pooled feature
    # to raw features, which can then be easily translated back to weights.
    # need to do proper testing and see if this even vaguely works

    # do we really need layerLplus2?

    def applyGradient(self, inputLayer, image, upsample):
        # calculate gradient for all n^2 weights in a kernel
        # gradient of next layer, dE/dx
        # organized as: d_l1[0] for node 0
        # organized as: d_l1[0][0] for weight b/t node 0
        # add a dummy costOverActivationPartials array to layerLplus2
        # array of floats, should match (each float is partial derivative of the cost function relative to this node's activation)
        # should be the length of layerLplus2
        # by the time this is called, this should actually exist
        delta_L0_maybe = inputLayer.costOverActivationPartials
        print(np.average(np.array(delta_L0_maybe)))
        # print(delta_L0_maybe)
        imglength = int(math.sqrt(len(image)))
        imgreshaped = np.reshape(np.array(image), (imglength, imglength))/255
        featmapLength = imglength - self.n + 1
        # print(featmapLength)
        gradArr = np.ndarray((self.n, self.n))
        with np.nditer(self.kernel, flags=['multi_index']) as itr:
            for q in itr:
                # print("q ", q, ", index ", itr.multi_index)
                sum_q = 0.0
                for i in range(featmapLength):
                    for j in range(featmapLength):
                        # d_xij/dw_q
                        weight_activ = imgreshaped[i + (self.n - 1 - itr.multi_index[0]), j + (self.n - 1 - itr.multi_index[1])]
                        # upsample goes here if pooled
                        # split weight activation via dirac or avg
                        # upsample is to return an array of grads to be split amongst weights
                        # for average pooling, I've determined nothing will be done at the moment.
                        # TODO: at the moment, all filters must be of the same dimension
                        # print("pos in activs: ", (self.position * featmapLength ** 2) + i * featmapLength + j)
                        elem_influence = delta_L0_maybe[(self.position * featmapLength ** 2) + i * featmapLength + j]
                        # print("pos in image: ", i + (self.n - 1 - itr.multi_index[0]), ", ",  j + (self.n - 1 - itr.multi_index[1]))
                        # dE/d_xij
                        # print(delta_L0_maybe[i * featmapLength + j])
                        sum_q += weight_activ * elem_influence
                        # print(sum_q)
                        # print(featmapLength)
                        # print(i)
                        # print(j)
                # gradient for indiv weight
                gradArr[itr.multi_index] = sum_q
                # may need to include activation function??
                # grad = sum over i,j in featmap of dE/dx_ij * dx_ij/dw
                # dE/dx_ij = delta_L1_maybe[i * 5 + j]
                # dx_ij/dW = image[(itr.multi_index[0] + i) * imglength + itr.multi_index[1] + j]
                # coord of weight + coord of featmap elem
                # dx/dw is input that was mult by w to form x_ij in featmap
                # dE/dx = grad of next layer (delta_L1_maybe)
                # dx/dw = value input to w
                # value input is given by coords of weight + coords of featmap
                # so the weight at (1,3) for x_8,9 of featmap is 9,12 in the image
        self.currGrad = -1 * gradArr
        # print("gradient for kernel:", self.currGrad)
        print("Avg val of gradient conv: ", np.average(self.currGrad))
        # print("before:", self.kernel)
        self.kernel = self.kernel + self.currGrad
        # print("after:", self.kernel)
        return gradArr


if __name__ == '__main__':
    convolve()
