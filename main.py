import os.path
from nnet import nnet

def getFile(prompt):
    fileName = raw_input(prompt)
    while not os.path.isfile(fileName):
        print("\tPlease input a valid file")
        fileName = raw_input(prompt)
    return fileName


def runNet():
    pass

if __name__ == "__main__":
    # netLoc = getFile("Initial net location: ")
    # trainLoc = getFile("Training Set: ")
    # outName = raw_input("Output Name: ")

    # epoch = raw_input("Epoch: ")
    # while not epoch.isdigit():
    #     epoch = raw_input("\tEnter Integer value for Epoch: ")
    # epoch = int(epoch)

    # rate = float(raw_input("Rate: "))

    netLoc = "res/sample.NNWDBC.init"
    trainLoc = "res/wdbc.train"
    outName = "res/test.out"
    epoch = "10"
    rate = "2.1"

    net = nnet(netLoc)
    print(net.nodesIn)
    print(net.nodesH)
    print(net.nodesOut)
    runNet()
