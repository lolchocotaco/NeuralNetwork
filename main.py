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

    if raw_input("Test or Train? ").lower() == "train":
        netLoc = "res/sample.NNGrades.init"
        trainLoc = "res/grades.train"
        outName = "res/gradeTrain.out"
        epoch = 100
        rate = 0.05

        net = nnet(netLoc)
        net.train(trainLoc, epoch, rate)
        net.writeFile(outName)
    else:
        netLoc = "res/gradeTrain.out"
        testLoc = "res/grades.test"
        outName = "test.out"

        net = nnet(netLoc)
        net.test(testLoc,outName)
