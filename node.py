class node:

    def __init__(self, nodeNum):
        self.nodeNum = nodeNum
        self.bias = 1
        self.inputBias = -1
        self.weights = []
        self.inval = []
        self.activation = []
        self.delta = []