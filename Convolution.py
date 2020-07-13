import numpy as np
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
    k1 = np.random.rand(5, 5)
    f1 = Filter(5, k1)
    f1.applyGradient(layer1, layer2, layer3, loadMNISTData('./train-images-idx3-ubyte', './train-labels-idx1-ubyte'), lambda x: x)
    cl = ConvolutionLayer(filters=[k1])
    cl.maps = [np.random.random((3, 3))]
    print("================================Convolution operation================================")
    f1.convolveToFeatMap(loadMNISTData('./train-images-idx3-ubyte', './train-labels-idx1-ubyte'))
    # print(cl.maps[0])
    # TODO: below
    # print(cl.poolMapsRedux(2))
    # print(cl.poolMaps(2, mode="max"))


class ConvolutionLayer:
    def __init__(self, inputlayer=None, previousLayer=None, nextLayer=None, filters=None):
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
        # either convolution layer or image (None)
        self.previousLayer = previousLayer
        # either densely connected or convolution
        self.nextlayer = nextLayer
        self.inputlayer = inputlayer

    # def applyGradient(self, inputLayer, nextLayer, layerLplus2, image, upsample):

    def backprop(self, image):
        # all filters apply grads, give proper info
        for f in self.filters:
            f.applyGradient(self.inputlayer, self.nextlayer, image, lambda x: x)

    def runFilters(self, image):
        # create feature maps
        maps = []
        for f in self.filters:
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
        print(poolToFeatEntryMap)
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


    def convolveToFeatMap(self, image):
        # return a i-nxi-n feature map
        # i = len image, n = len featmap
        # image is 1d array
        imglength = int(math.sqrt(len(image)))
        featmapLength = imglength - self.n + 1
        featmap = np.ndarray((featmapLength, featmapLength))
        imgreshaped = np.reshape(np.array(image), (imglength, imglength))
        # technically correct, but the image is ill-formatted
        # maybe = np.convolve(self.kernel.flatten(), np.array(image), 'valid')
        # maybe2 = np.convolve(self.kernel, imgreshaped, 'valid')
        convres = np.real(ifft2(fft2(imgreshaped) * fft2(self.kernel, s=imgreshaped.shape)))
        print(convres)
        print(convres.flatten())
        print(convres.shape)
        # returns a flattened convolution of the two, equal to the original dimension of the image
        return convres.flatten()

    # Tabled: Need to get gradient to work on pooled features
    # Could use a pooled object? Perhaps a pooled feature map can be associated with a mapping of a pooled feature
    # to raw features, which can then be easily translated back to weights.
    # need to do proper testing and see if this even vaguely works

    # do we really need layerLplus2?

    def applyGradient(self, inputLayer, nextLayer, layerLplus2, image, upsample):
        # calculate gradient for all n^2 weights in a kernel
        # gradient of next layer, dE/dx
        # organized as: d_l1[0] for node 0
        # organized as: d_l1[0][0] for weight b/t node 0
        # add a dummy costOverActivationPartials array to layerLplus2
        # array of floats, should match (each float is partial derivative of the cost function relative to this node's activation)
        # should be the length of layerLplus2
        dummypartials = [1 for i in range(len(layerLplus2.nodeList))]
        # print(dummypartials)
        layerLplus2.costOverActivationPartials = dummypartials
        delta_L1 = nextLayer.calculatePartialsInner(None, layerLplus2, inputLayer)
        delta_L0 = inputLayer.calculatePartialsInner(None, nextLayer, Layer(nodeList=list(map(lambda x: Node(activationValue=x), self.kernel.flatten()))))
        delta_L0_maybe = inputLayer.costOverActivationPartials
        # print(delta_L1)
        print(delta_L0_maybe)
        print(len(delta_L0_maybe))
        # print(image)
        imglength = int(math.sqrt(len(image)))
        featmapLength = imglength - self.n
        # print(featmapLength)
        # print(len(image))
        # 2d or flattened convolution??
        # we going flattened boys
        # Flattened rule: kernel_ij touches base + (i * len) + j
        gradArr = np.ndarray((self.n, self.n))
        with np.nditer(self.kernel, flags=['multi_index']) as itr:
            for q in itr:
                print(q, itr.multi_index)
                sum_q = 0.0
                for i in range(featmapLength):
                    for j in range(featmapLength):
                        # d_xij/dw_q
                        weight_activ = image[(itr.multi_index[0] + i) * imglength + itr.multi_index[1] + j]
                        # upsample goes here if pooled
                        # split weight activation via dirac or avg
                        # upsample is to return an array of grads to be split amongst weights
                        # for average pooling, I've determined nothing will be done at the moment.
                        elem_influence = delta_L0_maybe[i * featmapLength + j]
                        # print(image[(itr.multi_index[0] + i) * imglength + itr.multi_index[1] + j])
                        # dE/d_xij
                        # print(delta_L0_maybe[i * featmapLength + j])
                        sum_q += weight_activ * elem_influence
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
        print(gradArr)
        self.currGrad = -1 * gradArr
        print(self.kernel)
        self.kernel = self.kernel + self.currGrad
        print(self.kernel)
        return gradArr


if __name__ == '__main__':
    convolve()
