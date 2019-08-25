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
        for i in range(1, self.layers):
            self.layers[i].activateNodes()

    def checkWeightMatrix(self):
        for l in self.layers:
            l.checkWeights()

    def calculatePartials(self, expectedValueArray):
        currWeightGradient = self.layers[-1].calculatePartialsLast(expectedValueArray, self.layers[-2])
        for i in range(len(self.layers) - 1, 1, -1):
            # i + 1 is next layer, i - 1 is previous. Remember that partials are calculated from last layer back
            currWeightGradient = self.layers[i].calculatePartialsInner(currWeightGradient, self.layers[i+1], self.layers[i-1])
            self.layers[i].applyGradient(currWeightGradient)

    def calculateResultsArray(self, expectedResult, numOfResultsPossible):
        expectedValueArray = [0] * numOfResultsPossible
        expectedValueArray[expectedResult] = 1
        return expectedValueArray

    def runOnTrainingSet(self, inputArrays, expectedValueArray, numofClassifications):
        for i in range(0, len(inputArrays)):
            expValArr = self.calculateResultsArray(expectedValueArray[i], numofClassifications)
            self.feedInputLayer(inputArrays[i])
            self.activateLayers()
            print(self.layers[-1].getHighestActivation())
