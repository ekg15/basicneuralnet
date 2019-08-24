import math


class Node:
    def __init__(self, inputLayer=[], activationValue=0):
        self.inputLayer = inputLayer
        self.z = 0
        self.activationValue = activationValue

    def calculateActivationValue(self, weightvector):
        z = 0.0
        index = 0
        for node in self.inputLayer:
            print("wv[index]", weightvector[index])
            print("node.aV", node.activationValue)
            z += node.activationValue * weightvector[index]
            index += 1
        print("z", z)
        self.z = z
        self.activationValue = 1.0/(1.0 + math.pow(math.e, -1.0 * z))

    def calculateActivationValueFromInput(self, inputValue):
        self.activationValue = inputValue

