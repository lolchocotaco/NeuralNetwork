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
                self.layers[layerNum] = [node(y) for y in range(self.layerSize[layerNum])]

            for layer in range(1,3):
                for nodeNum in range(len(self.layers[layer])):
                    line = f.readline()
                    weights = map(float,line.split(" "))
                    bias = weights.pop(0)
                    self.layers[layer][nodeNum].weights = weights
                    self.layers[layer][nodeNum].bias = bias

    def writeFile(self, fileName):
        with open(fileName,'wr') as f:
            f.write(" ".join(map(str,self.layerSize)))


