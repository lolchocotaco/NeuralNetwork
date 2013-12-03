


class nnet:

    def __init__(self, netLoc):
        self.fileName = netLoc
        self.nodesIn = 0
        self.nodesH = 0
        self.nodesOut = 0
        self.loadFile()

    def loadFile(self):
        with open(self.fileName,'r') as f:
            fLine = (f.readline()).split(" ")
            self.nodesIn = fLine[0]
            self.nodesH = fLine[1]
            self.nodesOut = fLine[2]



