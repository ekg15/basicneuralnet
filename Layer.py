import math
from functools import reduce
from Node import *
from random import *

class Layer:
    def __init__(self, nodeList=[], weightMatrix=[]):
        self.nodeList = nodeList
        self.weightMatrix = weightMatrix
        self.costOverActivationPartials = []

    def generateInputNodes(self, numOfNodes):
        eL = []
        for i in range(0, numOfNodes):
            eL.append(Node())
        self.nodeList = eL

    def generateInnerNodes(self, numOfNodes, previousLayer):
        eL = []
        for i in range(0, numOfNodes):
            eL.append(Node(previousLayer.nodeList))
        self.nodeList = eL

    def createWeightMatrix(self, previousLayer):
        lmao = []
        for i in range(0, len(self.nodeList)):
            nodeWeightList = []
            print(i)
            for j in range(0, len(previousLayer.nodeList)):
                nodeWeightList.append(uniform(-1, 1))
            # list at index 0 in weightMatrix corresponds to node 0 in nodeList
            lmao.append(nodeWeightList)
        self.weightMatrix = lmao
        print(self.weightMatrix)

    def inputFromImageArray(self, imageArray):
        index = 0
        for n in self.nodeList:
            n.calculateActivationValueFromInput(imageArray[index])
            index += 1

    def dummyInputNodes(self):
        for n in self.nodeList:
            n.calculateActivationValueFromInput(1)

    def activateNodes(self):
        index = 0
        for n in self.nodeList:
            n.calculateActivationValue(self.weightMatrix[index])
            index += 1

    def calculateCost(self, expectedResults):
        index = 0
        cost = 0.0
        for n in self.nodeList:
            cost += math.pow(n.activationValue - expectedResults[index], 2)
            print("current", cost)
            index += 1
        return cost

    def calculatePartialsInner(self, nextLayerWeightGradient, nextLayer, previousLayer):
        eL = []
        weightGradient = []
        # each node in this layer. We access the necessary weights by iterating over this and the next layer,
        # which will provide the matching coordinates in out weight matrix
        for i in range(0, len(self.nodeList)):
            print("starting")
            partialList = []
            # [l][i] is how each node l in the next layer is affected by node i in current layer
            costOverActivationPartial = 0.0
            for l in range(0, len(nextLayer.nodeList)):
                sigmoidOfZNext = 1.0 / (1 + math.pow(math.e, -1 * nextLayer.nodeList[l].z))
                # for inner layer partial we need to sum effects of this partial on each of next layer's partials
                costOverActivationPartial += nextLayer.costOverActivationPartials[l] * (sigmoidOfZNext * (1 - sigmoidOfZNext)) \
                    * nextLayer.weightMatrix[l][i]
                print(nextLayer.weightMatrix[l][i], l, i)
                print(nextLayer.costOverActivationPartials[l])
            # now iterating over the weights (between current[i] and prev[j]) in our matrix and calculating partials
            for j in range(0, len(previousLayer.nodeList)):
                sigmoidOfZ = 1.0/(1 + math.pow(math.e, -1 * self.nodeList[i].z))
                partialij = costOverActivationPartial * (sigmoidOfZ * (1-sigmoidOfZ)) * previousLayer.nodeList[j].activationValue
                partialList.append(partialij)
            eL.append(costOverActivationPartial)
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        return weightGradient

    def calculatePartialsLast(self, expectedResults, previousLayer):
        eL = []
        weightGradient = []
        # for each node in current layer
        for i in range(0, len(self.nodeList)):
            partialList = []
            # j is for each node in the next layer, aka weight [i][j] between the previous layer's node j and node i
            for j in range(0, len(previousLayer.nodeList)):
                sigmoidOfZ = 1.0/(1 + math.pow(math.e, -1 * self.nodeList[i].z))
                partialij = 2 * (self.nodeList[i].activationValue - expectedResults[i]) * (sigmoidOfZ * (1-sigmoidOfZ))\
                    * previousLayer.nodeList[j].activationValue
                partialList.append(partialij)  # the value for weight ij
            eL.append(2 * (self.nodeList[i].activationValue - expectedResults[i]))
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        return weightGradient  # the portion of the gradient for this layer's weights

    def applyGradient(self, partialMatrix):
        # i is for each node in current layer
        for i in range(0, len(partialMatrix)):
            # j is for each node the next layer, aka weight [i][j] between the previous layer's node j and node i
            for j in range(0, len(partialMatrix[i])):
                self.weightMatrix[i][j] += -1 * partialMatrix[i][j]
