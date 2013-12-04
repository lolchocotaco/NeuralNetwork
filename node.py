class node:

    def __init__(self, nodeNum):
        self.nodeNum = nodeNum
        self.biasWeight = 1
        self.inputBias = -1
        self.weights = []
        self.inval = 0
        self.activation = -1
        self.delta = 0