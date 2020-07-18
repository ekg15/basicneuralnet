import math
from Convolution import *
from Node import *
from Layer import *
from Network import *
from ImageFormatting import *
from random import *


def convtest():
    image = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    expectedResultsArray = [0, 1]
    k1 = np.ones((2, 2), dtype='float')
    k2 = np.ones((2, 2), dtype='float')
    f1 = Filter(2, k1)
    f2 = Filter(2, k2)
    filters = [f1, f2]
    layer1 = Layer()
    layer2 = Layer()
    layer3 = Layer()
    layer1.generateInputNodes(8)
    layer2.generateInnerNodes(3, layer1)
    layer3.generateInnerNodes(2, layer2)
    convlayer = ConvolutionLayer(inputlayer=layer1, nextLayer=layer2, filters=filters)
    convlayer.runFilters(image)

    layer1.inputFromFeatMaps(convlayer.maps)
    for n in layer1.nodeList:
        print(n.activationValue)

    layer2.createWeightMatrix(layer1)
    layer3.createWeightMatrix(layer2)

    # layer2.checkWeights()
    # layer3.checkWeights()

    layer2.activateNodes()
    # for n in layer2.nodeList:
    #    print(n.activationValue)

    print("L3 activ")
    layer3.activateNodes()
    for n in layer3.nodeList:
        print(n.activationValue)

    wGF = layer3.calculatePartialsLast(expectedResultsArray, layer2)
    # print(wGF)
    wGI1 = layer2.calculatePartialsInner(wGF, layer3, layer1)
    # print(wGI1)
    layer3.applyGradient(wGF)
    layer2.applyGradient(wGI1)
    convlayer.backprop(image)





def main():
    # imageToValueArray('./wbgm.jpg')
    image = loadMNISTData('./train-images-idx3-ubyte', './train-labels-idx1-ubyte')
    seed(2)
    expectedResultsArray = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    layer1 = Layer()
    layer2 = Layer()
    layer3 = Layer()
    layer4 = Layer()

    layer1.generateInputNodes(784)
    layer2.generateInnerNodes(32, layer1)
    layer3.generateInnerNodes(32, layer2)
    layer4.generateInnerNodes(10, layer3)

    layer1.inputFromImageArray(image)

    layer2.createWeightMatrix(layer1)
    layer3.createWeightMatrix(layer2)
    layer4.createWeightMatrix(layer3)

    layer2.checkWeights()
    layer3.checkWeights()
    layer4.checkWeights()

    layer2.activateNodes()
    layer3.activateNodes()
    layer4.activateNodes()

    for n in layer2.nodeList:
        print(n.activationValue)
    for n in layer3.nodeList:
        print(n.activationValue)
    cost = layer4.calculateCost(expectedResultsArray)
    print("cost", cost)
    wGF = layer4.calculatePartialsLast(expectedResultsArray, layer3)
    # wGI1 = layer3.calculatePartialsInner(wGF, layer4, layer2)
    wGI1 = layer3.calculatePartialsInner(wGF, layer4, layer2)
    print("Layer 2 Partial Calculation")
    wGI = layer2.calculatePartialsInner(wGI1, layer3, layer1)

    print("l4w")
    # layer4.checkWeights()
    # print(wGF)
    layer4.applyGradient(wGF)
    # layer4.checkWeights()

    print("l3w")
    # layer3.checkWeights()
    # print(wGI1)
    layer3.applyGradient(wGI1)
    # layer3.checkWeights()


    # print("l2w")
    # layer2.checkWeights()
    # print(wGI)
    layer2.applyGradient(wGI)
    # layer2.checkWeights()
    # print(image)

    # print(layer4.findPartialDerivative(1, 1, 1))

    inputArray, valueArray = loadMNISTDataArray('./train-images-idx3-ubyte', './train-labels-idx1-ubyte')
    network = Network()
    network.addInputLayer(784)
    network.addInnerLayer(32)
    network.addInnerLayer(16)
    network.addInnerLayer(10)
    network.runOnTrainingSet(inputArray[0:10000], valueArray[0:10000], 10)
    # showImage(inputArray[4999])





if __name__ == '__main__':
    convtest()
