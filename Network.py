import math
from Layer import *


class Network:
    def __init__(self):
        self.layers = []
        self.convLayer = None

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


    def feedInputLayerConv(self):
        self.layers[0].inputFromFeatMaps(self.convLayer.maps)


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

    def calcAndApplyGradient(self, gradients):
        # all gradients compiled by running network
        print(len(gradients))
        print(len(gradients[0]))
        print(len(gradients[0][0]))
        print(gradients[0][0])
        print(len(gradients[0][0][0]))
        for k in range(1, len(gradients)):
            # each layer's set of weight lists
            for m in range(0, len(gradients[k])):
                # each node's list of weights
                for l in range(0, len(gradients[k][m])):
                    # each weight in the weight list
                    for u in range(0, len(gradients[k][m][l])):
                        gradients[0][m][l][u] += gradients[k][m][l][u]
        for o in range(0, len(gradients[0])):
            for x in range(0, len(gradients[0][o])):
                for b in range(0, len(gradients[0][o][x])):
                    gradients[0][o][x][b] = gradients[0][o][x][b]/len(gradients)
        print(gradients[0][0])
        for i in range(1, len(gradients[0]) + 1):
            self.layers[-i].applyGradient(gradients[0][i-1])


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
            # if i % 1000 == 0 and i != 0:
            #     self.calcAndApplyGradient(sgdGradient)
        print(resList)
        fail = 0
        numpass = 0
        print(sum(1 for x, y in resList[:-50] if x == y) / len(resList[:-50]))
        print(sum(1 for x, y in resList[-50:] if x == y) / len(resList[-50:]))


    def runOnTrainingSetConvolution(self, inputArrays, expectedValueArray, numofClassifications):
        resList = []
        sgdGradient = []
        for i in range(0, len(inputArrays)):
            expValArr = self.getResultsArray(expectedValueArray[i], numofClassifications)
            self.convLayer.runFilters(inputArrays[i])
            self.feedInputLayerConv()
            self.activateLayers()
            # for n in self.layers[-1].nodeList:
            #     print(n.activationValue, end=",")
            print(self.layers[-1].getHighestActivation(), expectedValueArray[i])
            resList.append((self.layers[-1].getHighestActivation(), expectedValueArray[i]))
            sgdGradient.append(self.calculatePartials(expValArr))
            self.convLayer.backprop(inputArrays[i])
            # if i % 1000 == 0 and i != 0:
            #     self.calcAndApplyGradient(sgdGradient)
        print(resList)
        fail = 0
        numpass = 0
        print(sum(1 for x, y in resList[:-50] if x == y) / len(resList[:-50]))
        print(sum(1 for x, y in resList[-50:] if x == y) / len(resList[-50:]))
