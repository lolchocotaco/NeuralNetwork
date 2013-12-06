import math

class node:
    def __init__(self, nodeNum):
        self.nodeNum = nodeNum
        self.biasWeight = 1
        self.inputBias = -1
        self.weights = []
        self.inval = 0
        self.activation = -1
        self.delta = 0

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
                    weights = map(float, line.split(" "))
                    self.layers[layer][nodeNum].biasWeight = weights.pop(0)
                    self.layers[layer][nodeNum].weights = weights

    def writeFile(self, fileName):
        print("Writing to '{0}'".format(fileName))
        with open(fileName,'wr') as f:
            f.write(" ".join(map(str,  self.layerSize)) + "\n")
            for layer in range(1, 3):
                for node in self.layers[layer]:
                    weightVec = [node.biasWeight]
                    weightVec.extend(node.weights)
                    f.write(" ".join([format(val,'.3f') for val in weightVec])+ "\n")

    def train(self, fileName, epoch, alpha):
        print("=====BEGIN TRAINING======")
        print("Reading from '{0}'".format(fileName))
        currEpoch = 0
        while currEpoch < epoch:
            with open(fileName, "r") as f:
                fLine = (f.readline().strip()).split(" ")

                #For each training case
                for line in f:
                    trainingRow = map(float, line.split(" "))
                    truths = [int(trainingRow.pop(len(trainingRow)-1)) for node in self.layers[-1]]
                    truths.reverse()
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
                    for node in self.layers[-1]:
                        node.delta = self.dSig(node.inval)*(truths[node.nodeNum] - node.activation)

                    # Back propogation
                    for layer in range(1, 2):
                        for node in self.layers[layer]:
                            deltas = [nextNode.delta for nextNode in self.layers[layer+1]]
                            weights = [nextNode.weights[node.nodeNum] for nextNode in self.layers[layer+1]]
                            node.delta = self.dSig(node.inval) * sum([a*b for a, b in zip(deltas, weights)])

                    # For all weights
                    for layer in range(1,3):
                        for node in self.layers[layer]:
                            actDelAlpha = [alpha * prevNode.activation * node.delta for prevNode in self.layers[layer-1]]
                            node.biasWeight += alpha*node.inputBias*node.delta
                            pastWeights = node.weights
                            node.weights = [a+b for a, b in zip(pastWeights, actDelAlpha)]

            currEpoch += 1

    def test(self, fileName,outName):
        print("=====BEGIN TESTING======")
        print("Writing to {0}".format(outName))
        result = []
        truthVec = []
        with open(fileName,"r")as r:
                fLine = (r.readline().strip()).split(" ")
                for line in r:
                    trainingRow = map(float, line.split(" "))
                    truths = [int(trainingRow.pop(len(trainingRow)-1)) for node in self.layers[-1]]
                    truths.reverse()

                    # print(truths)
                    # truth = int(trainingRow.pop(len(trainingRow)-1))
                    truthVec.append(truths)

                    #Initialize input Layer
                    for node in self.layers[0]:
                        node.activation = trainingRow[node.nodeNum]

                    #Propogate forward
                    for layer in range(1,3):
                        for node in self.layers[layer]:
                            activations = [prevNode.activation for prevNode in self.layers[layer-1]]
                            node.inval = sum([a*b for a, b in zip(node.weights, activations)]) + node.inputBias*node.biasWeight
                            node.activation = self.sig(node.inval)

                    # create vector of node classification results
                    testRes = []
                    for node in self.layers[-1]:
                        if node.activation >= 0.5:
                            testRes.append(1)
                        else:
                            testRes.append(0)
                    result.append(testRes) # append the result from each test

        # Testing is now done.  write results
        with open(outName, "w") as w:
            A, B, C, D = 0, 0, 0, 0
            avgAcc, avgPre, avgRec, avgF1 = 0.0, 0.0, 0.0, 0.0
            for node in self.layers[-1]:
                nodeTruth = [truth[node.nodeNum] for truth in truthVec]
                nodeGuess = [res[node.nodeNum] for res in result]
                truthAndGuess = zip(nodeGuess, nodeTruth)
                # Getting confusion matrix elements
                a = sum([x & y for x, y in truthAndGuess])
                A += a
                b = sum([x & ~y for x, y in truthAndGuess])
                B += b
                c = sum([~x & y for x, y in truthAndGuess])
                C += c
                d = sum([(~x & ~y) + 2 for x, y in truthAndGuess])
                D += d

                acc = (a + d)/float((a + b + c + d))
                avgAcc += acc
                pre = a/float((a + b))
                avgPre += pre
                rec = a/float((a + c))
                avgRec += rec
                f1  = (2*pre*rec)/float((pre + rec))
                avgF1  += f1
                w.write("{0} {1} {2} {3} {4} {5} {6} {7}\n"
                        .format(a, b, c, d, format(acc, '.3f'), format(pre, '.3f'), format(rec, '.3f'), format(f1, '.3f') ))

            # Microaveraging
            acc = (A + D)/float((A + B + C + D))
            pre = A/float((A + B))
            rec = A/float((A + C))
            f1  = (2*pre*rec)/(pre + rec)
            w.write("{0} {1} {2} {3}\n".format(format(acc, '.3f'), format(pre, '.3f'), format(rec, '.3f'), format(f1, '.3f') ))

            #Macroaveraging:
            avgAcc /= len(self.layers[-1])
            avgPre /= len(self.layers[-1])
            avgRec /= len(self.layers[-1])
            avgF1  = (2*avgPre*avgRec)/(avgPre+ avgRec)
            w.write("{0} {1} {2} {3}\n".format(format(avgAcc,'.3f'), format(avgPre, '.3f'), format(avgRec, '.3f'), format(avgF1, '.3f') ))



    def sig(self,val):
        return 1/(1+math.e**(-val))

    def dSig(self, val):
        return self.sig(val)*(1-self.sig(val))