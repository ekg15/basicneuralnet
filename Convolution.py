import numpy as np
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
    # print(cl.poolMaps(2))
    # print(cl.poolMaps(2, mode="max"))


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
        # needs: pool, filter
        # use grabsize and knowledge of method to backprop grad to weights evenly or dirac
        self.upsample = lambda x: x
        # either convolution layer or image (None)
        self.previousLayer = previousLayer
        # either densely connected or convolution
        self.nextlayer = nextLayer

    def runFilters(self, image):
        # create feature maps
        pass

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
        pooled = []
        for fm in self.maps:
            # how many features should be combined
            # ceil(k/n) x ceil(k/n) areas
            grabsize = np.shape(fm)[0] - n + 1
            pm = np.ndarray((n, n))
            for i in range(n):
                for j in range(n):
                    if mode == "average":
                        print(fm[i:i+grabsize, j:j+grabsize])
                        pm[i][j] = np.average(fm[i:i+grabsize, j:j+grabsize])
                    else:
                        pm[i][j] = np.max(fm[i:i+grabsize, j:j+grabsize])
            pooled.append(pm)
        self.pooledmaps = pooled
        return pooled

    def backprop(self):
        # all filters apply grads, give proper info
        pass


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
        maybe = np.convolve(self.kernel.flatten(), np.array(image), 'valid')
        print(imglength)
        print(self.kernel)
        print(np.shape(maybe))
        print(maybe)
        # try 2d
        # print(imgreshaped)
        for i in range(featmapLength):
            for j in range(featmapLength):
                # convolve kernel and image[i:i+self.n, j:j+self.n]
                # 7/12: Some sort of weird bug here in regards to array formatting
                # I should probably just do convolution manually
                featmap[i][j] = np.convolve(self.kernel, imgreshaped[i:i+self.n, j:j+self.n], 'valid')
        print(featmap)
        return featmap

    # TODO: Need to get gradient to work on pooled features
    # Could use a pooled object? Perhaps a pooled feature map can be associated with a mapping of a pooled feature
    # to raw features, which can then be easily translated back to weights.

    def applyGradient(self, inputLayer, nextLayer, layerLplus2, image, upsample):
        # calculate gradient for all n^2 weights in a kernel
        # gradient of next layer, dE/dx
        # organized as: d_l1[0] for node 0
        # organized as: d_l1[0][0] for weight b/t node 0
        # add a dummy costOverActivationPartials array to layerLplus2
        # array of floats, should match (each float is partial derivative of the cost function relative to this node's activation)
        # should be the length of layerLplus2
        dummypartials = [np.random.random() * 5 for i in range(len(layerLplus2.nodeList))]
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
                        pass
                        # d_xij/dw_q
                        weight_activ = image[(itr.multi_index[0] + i) * imglength + itr.multi_index[1] + j]
                        # upsample goes here if pooled
                        # split weight activation via dirac or avg
                        # upsample is to return an array of grads to be split amongst weights
                        # need to implement backprop for pooling, annoying
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
