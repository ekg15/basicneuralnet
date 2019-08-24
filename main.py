import math
from Node import *
from Layer import *
from ImageFormatting import *
from random import *

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

    layer2.activateNodes()
    layer3.activateNodes()
    layer4.activateNodes()

    for n in layer2.nodeList:
        print(n.activationValue)
    for n in layer3.nodeList:
        print(n.activationValue)
    cost = layer4.calculateCost(expectedResultsArray)
    print("cost", cost)
    wGF = layer4.calculatePartialsLast(expectedResultsArray, layer2)
    wGI1 = layer3.calculatePartialsInner(wGF, layer4, layer2)
    wGI1 = layer3.calculatePartialsInner(wGF, layer4, layer2)
    wGI = layer2.calculatePartialsInner(wGF, layer3, layer1)

    print("l4wm", layer4.weightMatrix)
    print("wGF", wGF)
    layer4.applyGradient(wGF)
    print("l4wm", len(layer4.weightMatrix), len(layer4.weightMatrix[0]))
    print(layer4.costOverActivationPartials)

    layer3.applyGradient(wGI1)
    print("l3wm", len(layer3.weightMatrix), len(layer3.weightMatrix[0]))
    print(layer3.costOverActivationPartials)

    layer2.applyGradient(wGI)
    print("l2wm", len(layer2.weightMatrix), len(layer2.weightMatrix[0]))




if __name__ == '__main__':
    main()
