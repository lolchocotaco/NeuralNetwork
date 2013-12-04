import math
from node import node


class nnet:
    # Input : Index 0
    # Hidden: Index 1
    # Output: Index 2

    def __init__(self, netLoc):
        self.fileName = netLoc
        self.layers = [0]*3
        self.layerSize = [0]*3
        self.nodesIn = 0
        self.nodesH = 0
        self.nodesOut = 0
        self.loadFile()

    def loadFile(self):
        with open(self.fileName, 'r') as f:
            fLine = (f.readline()).split(" ")
            for layerNum, val in enumerate(fLine):
                self.layerSize[layerNum] = int(fLine[layerNum])
                self.layers[layerNum] = [node(y) for y in range(self.layerSize[layerNum])] # initialize a list of nodes

            for layer in range(1,3):
                for nodeNum in range(len(self.layers[layer])):
                    line = f.readline()
                    weights = map(float,line.split(" "))
                    self.layers[layer][nodeNum].biasWeight = weights.pop(0)
                    self.layers[layer][nodeNum].weights = weights

    def writeFile(self, fileName):
        with open(fileName,'wr') as f:
            f.write(" ".join(map(str,  self.layerSize)) + "\n")
            for layer in range(1, 3):
                for node in self.layers[layer]:
                    f.write(" ".join(map(str,[round(x, 3) for x in node.weights]))+ "\n")

    def train(self, fileName, epoch, alpha):
        currEpoch = 0
        while currEpoch < epoch:
            with open(fileName, "r") as f:
                fLine = (f.readline().strip()).split(" ")
                print(fLine)

                #For each training case
                for line in f:
                    trainingRow = map(float, line.split(" "))
                    truth = int(trainingRow.pop(len(trainingRow)-1))

                    #Initialize input Layer
                    for node in self.layers[0]:
                        node.activation = trainingRow[node.nodeNum]

                    #Propogate forward
                    for layer in range(1,3):
                        for node in self.layers[layer]:
                            activations = [prevNode.activation for prevNode in self.layers[layer-1]]
                            node.inval = sum([a*b for a, b in zip(node.weights, activations)]) + node.inputBias*node.biasWeight
                            node.activation = self.sig(node.inval)

                     #Initialize for output later
                    for node in self.layers[2]:
                        node.delta = self.dSig(node.inval)*(truth - node.activation)

                    # Back propogation
                    for node in self.layers[1]:
                        deltas = [nextNode.delta for nextNode in self.layers[2]]
                        weights = [nextNode.weights[node.nodeNum] for nextNode in self.layers[2]]
                        node.delta = self.dSig(node.inval) * sum([a*b for a, b in zip(deltas, weights)])

                    # For all weights
                    for layer in range(1,3):
                        for node in self.layers[layer]:
                            actDelAlpha = [alpha * prevNode.activation * node.delta for prevNode in self.layers[layer-1]]
                            node.biasWeight += alpha*node.inputBias*node.delta
                            node.weights = [a+b for a,b in zip(node.weights, actDelAlpha)]


            currEpoch +=1

    def fowardProp(self, fileName, epoch, alpha):




    def sig(self,val):
        return 1/(1+math.e**(-val))

    def dSig(self, val):
        return self.sig(val)*(1-self.sig(val))