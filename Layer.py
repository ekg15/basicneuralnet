import math
from functools import reduce
from Node import *
from random import *
import numpy as np


class Layer:
    def __init__(self, nodeList=[]):
        self.nodeList = nodeList
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

    def generateInputNodesConv(self, numOfNodes):
        eL = []
        for i in range(0, numOfNodes):
            eL.append(Node(activationFunction=lambda x: .02 * x if x < 0 else x))
        self.nodeList = eL

    def generateInnerNodesConv(self, numOfNodes, previousLayer):
        eL = []
        for i in range(0, numOfNodes):
            eL.append(Node(inputLayer=previousLayer.nodeList, activationFunction=lambda x: .02 * x if x < 0 else x))
        self.nodeList = eL

    def createWeightMatrix(self, previousLayer):
        lmao = []
        for i in range(0, len(self.nodeList)):
            nodeWeightList = []
            print(i)
            for j in range(0, len(previousLayer.nodeList)):
                nodeWeightList.append(uniform(-1, 1))
            # list at index 0 in weightMatrix corresponds to node 0 in nodeList
            self.nodeList[i].weights = nodeWeightList
            lmao.append(nodeWeightList)

    def checkWeights(self):
        for n in self.nodeList:
            print("weights:", n.weights, len(n.weights), end=' ')

    def inputFromImageArray(self, imageArray):
        index = 0
        # print(imageArray)
        for n in self.nodeList:
            n.calculateActivationValueFromInput(imageArray[index]/255)
            index += 1

    def inputFromFeatMaps(self, maps):
        maplen = maps[0].flatten().size
        for i in range(len(self.nodeList)):
            for f in maps[i//maplen].flatten():
                self.nodeList[i].calculateActivationValueFromInput(f)


    def dummyInputNodes(self):
        for n in self.nodeList:
            n.calculateActivationValueFromInput(1)

    def activateNodes(self):
        for n in self.nodeList:
            n.calculateActivationValue()

    def calculateCost(self, expectedResults):
        index = 0
        cost = 0.0
        for n in self.nodeList:
            cost += math.pow(n.activationValue - expectedResults[index], 2)
            print("current", cost)
            index += 1
        return cost

    def getHighestActivation(self):
        activs = list(map(lambda x: x.activationValue, self.nodeList))
        return max(list(range(len(activs))), key=lambda x: activs[x])


    # going to need new Partial Derivative function to work with Convolutional layers
    # or maybe not


    # partial derivatives of the cost with respect to an inner layer of weights


    def calculatePartialsInner(self, nextLayerWeightGradient, nextLayer, previousLayer):
        eL = []
        weightGradient = []
        # each node in this layer. We access the necessary weights by iterating over this and the next layer,
        # which will provide the matching coordinates in out weight matrix
        for i in range(0, len(self.nodeList)):
            # print("starting")
            partialList = []
            # [l][i] is how each node l in the next layer is affected by node i in current layer
            costOverActivationPartial = 0.0
            for l in range(0, len(nextLayer.nodeList)):
                sigmoidOfZNext = 1.0 / (1 + math.pow(math.e, -1 * nextLayer.nodeList[l].z))
                # for inner layer partial we need to sum effects of this partial on each of next layer's partials
                # print(nextLayer.nodeList[l].weights)
                costOverActivationPartial += nextLayer.costOverActivationPartials[l] * (sigmoidOfZNext * (1 - sigmoidOfZNext)) \
                    * nextLayer.nodeList[l].weights[i]
                # print(nextLayer.nodeList[l].weights[i], l, i)
                # print(nextLayer.costOverActivationPartials[l])
            # now iterating over the weights (between current[i] and prev[j]) in our matrix and calculating partials
            for j in range(0, len(previousLayer.nodeList)):
                sigmoidOfZ = 1.0/(1 + math.pow(math.e, -1 * self.nodeList[i].z))
                partialij = costOverActivationPartial * (sigmoidOfZ * (1-sigmoidOfZ)) * previousLayer.nodeList[j].activationValue
                partialList.append(partialij)
            # append partial derivative of the cost function relative to this node's activation
            eL.append(costOverActivationPartial)
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        return weightGradient


    def calculatePartialsInnerRelu(self, nextLayerWeightGradient, nextLayer, previousLayer):
        eL = []
        weightGradient = []
        # each node in this layer. We access the necessary weights by iterating over this and the next layer,
        # which will provide the matching coordinates in out weight matrix
        for i in range(0, len(self.nodeList)):
            # print("starting")
            partialList = []
            # [l][i] is how each node l in the next layer is affected by node i in current layer
            costOverActivationPartial = 0.0
            for l in range(0, len(nextLayer.nodeList)):
                # print(nextLayer.nodeList[l].z)
                reLuOfZNext = .02 * nextLayer.nodeList[l].z if nextLayer.nodeList[l].z < 0 else nextLayer.nodeList[l].z
                derivativeOfZNext = .02 if nextLayer.nodeList[l].z < 0 else 0
                # for inner layer partial we need to sum effects of this partial on each of next layer's partials
                # print(nextLayer.nodeList[l].weights)
                costOverActivationPartial += nextLayer.costOverActivationPartials[l] * derivativeOfZNext \
                    * nextLayer.nodeList[l].weights[i]
                # print(nextLayer.nodeList[l].weights[i], l, i)
                # print(nextLayer.costOverActivationPartials[l])
            # now iterating over the weights (between current[i] and prev[j]) in our matrix and calculating partials
            for j in range(0, len(previousLayer.nodeList)):
                reLuOfZ = .02 * self.nodeList[i].z if self.nodeList[i].z < 0 else self.nodeList[i].z
                derivativeOfZ = .02 if self.nodeList[i].z < 0 else 0
                partialij = costOverActivationPartial * derivativeOfZ * previousLayer.nodeList[j].activationValue
                partialList.append(partialij)
            # append partial derivative of the cost function relative to this node's activation
            eL.append(costOverActivationPartial)
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        return weightGradient


    def calculatePartialsInputLinear(self, nextLayerWeightGradient, nextLayer):
        eL = []
        weightGradient = []
        # each node in this layer. We access the necessary weights by iterating over this and the next layer,
        # which will provide the matching coordinates in out weight matrix
        for i in range(0, len(self.nodeList)):
            # print("starting")
            partialList = []
            # [l][i] is how each node l in the next layer is affected by node i in current layer
            costOverActivationPartial = 0.0
            for l in range(0, len(nextLayer.nodeList)):
                # print(nextLayer.nodeList[l].z)
                reLuOfZNext = .02 * nextLayer.nodeList[l].z if nextLayer.nodeList[l].z < 0 else nextLayer.nodeList[l].z
                derivativeOfZNext = .02 if nextLayer.nodeList[l].z < 0 else 1
                # for inner layer partial we need to sum effects of this partial on each of next layer's partials
                # print(nextLayer.nodeList[l].weights)
                # print(nextLayer.costOverActivationPartials[l])
                # print(derivativeOfZNext)
                costOverActivationPartial += nextLayer.costOverActivationPartials[l] * derivativeOfZNext \
                    * nextLayer.nodeList[l].weights[i]
            # print(costOverActivationPartial)
            # append partial derivative of the cost function relative to this node's activation
            eL.append(costOverActivationPartial)
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        # print(self.costOverActivationPartials)
        # print("max input partial: ", max(self.costOverActivationPartials))
        return weightGradient


    # partial derivatives of the cost with respect to the last layer of weights


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
            # append partial derivative of the cost function relative to this node's activation
            eL.append(2 * (self.nodeList[i].activationValue - expectedResults[i]))
            weightGradient.append(partialList)
        self.costOverActivationPartials = eL
        return weightGradient  # the portion of the gradient for this layer's weights

    def findPartialDerivative(self, activationPrevious, zCurrent, costOverActivationPartial):
        sigmoidOfZ = 1.0 / (1 + math.pow(math.e, -1 * zCurrent))
        print(sigmoidOfZ)
        return costOverActivationPartial * (sigmoidOfZ * (1 - sigmoidOfZ)) * activationPrevious

    def applyGradient(self, partialMatrix):
        # i is for each node in current layer
        for i in range(0, len(partialMatrix)):
            # j is for each node the next layer, aka weight [i][j] between the previous layer's node j and node i
            for j in range(0, len(partialMatrix[i])):
                self.nodeList[i].weights[j] += -1 * partialMatrix[i][j]
        # print("average val of gradient ", len(self.nodeList), ": ", np.average(np.array(partialMatrix)))

