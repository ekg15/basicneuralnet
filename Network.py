import math
from Layer import *


class Network:
    def __init__(self):
        self.layers = []

    def addInputLayer(self, numOfNodes):
        eLL = []
        layer = Layer()
        layer.generateInputNodes(numOfNodes)
        eLL.append(layer)
        self.layers = eLL

    def addInnerLayer(self, numOfNodes):
        eLL = []
        for l in self.layers:
            eLL.append(l)
        layer = Layer()
        layer.generateInnerNodes(numOfNodes, self.layers[-1])
        layer.createWeightMatrix(self.layers[-1])
        eLL.append(layer)
        self.layers = eLL

    def feedInputLayer(self, inputArray):
        self.layers[0].inputFromImageArray(inputArray)

    def activateLayers(self):
        for i in range(1, len(self.layers)):
            self.layers[i].activateNodes()

    def checkWeightMatrix(self):
        for l in self.layers:
            l.checkWeights()

    def calculatePartials(self, expectedValueArray):
        gradient = []
        currWeightGradient = self.layers[-1].calculatePartialsLast(expectedValueArray, self.layers[-2])
        gradient.append(currWeightGradient)
        for i in range(len(self.layers) - 2, 0, -1):
            # i + 1 is next layer, i - 1 is previous. Remember that partials are calculated from last layer back
            currWeightGradient = self.layers[i].calculatePartialsInner(currWeightGradient, self.layers[i+1], self.layers[i-1])
            gradient.append(currWeightGradient)
        for i in range(1, len(gradient) + 1):
            # print(gradient[i-1])
            self.layers[-i].applyGradient(gradient[i-1])
        return gradient

    def getResultsArray(self, expectedResult, numOfResultsPossible):
        expectedValueArray = [0] * numOfResultsPossible
        expectedValueArray[expectedResult] = 1
        return expectedValueArray

    def runOnTrainingSet(self, inputArrays, expectedValueArray, numofClassifications):
        resList = []
        for i in range(0, len(inputArrays)):
            expValArr = self.getResultsArray(expectedValueArray[i], numofClassifications)
            self.feedInputLayer(inputArrays[i])
            self.activateLayers()
            print(self.layers[-1].getHighestActivation(), expectedValueArray[i])
            resList.append((self.layers[-1].getHighestActivation(), expectedValueArray[i]))
            self.calculatePartials(expValArr)
        print(resList)
        fail = 0
        numpass = 0
        print(sum(1 for x, y in resList if x == y) / len(resList))
        print(sum(1 for x, y in resList[-50:] if x == y) / len(resList[-50:]))
