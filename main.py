import os.path
from nnet import nnet
import random

def getFile(prompt):
    fileName = raw_input(prompt)
    while not os.path.isfile(fileName):
        print("\tPlease input a valid file")
        fileName = raw_input(prompt)
    return fileName




def genInit():
    inNodes = raw_input("Num input: ")
    hNodes = raw_input("Num Hidden: ")
    oNodes = raw_input("Num Out: ")

    with open("data/nn.init","w") as w:
        w.write("{0} {1} {2}\n".format(inNodes,hNodes,oNodes))

        for hNodeNum in range(int(hNodes)):
            line = [random.random() for i in range(int(inNodes)+1)]
            w.write(" ".join([format(val,'.3f') for val in line])+ '\n')

        for oNodeNum in range(int(oNodes)):
            line = [random.random() for j in range(int(hNodes)+1)]
            w.write(" ".join([format(val,'.3f') for val in line])+'\n')


if __name__ == "__main__":

    print("Sameer Chauhan's Basic Neural Network\n=====================")
    print("This Neural Network can work with one hidden layer")
    print("What would you like to do?")

    inPutz = raw_input("Train or Test or Gen (network) ? ").lower()
    if inPutz == "train":
        # netLoc = getFile("Initial net location: ")
        # trainLoc = getFile("Training Set: ")
        # outName = raw_input("Output Name: ")

        # epoch = raw_input("Epoch: ")
        # while not epoch.isdigit():
        # epoch = raw_input("\tEnter Integer value for Epoch: ")
        # epoch = int(epoch)

        # rate = float(raw_input("Rate: "))

        netLoc = "data/nn.init"
        trainLoc = "data/training.csv"
        outName = "data/sahearts.trained"
        epoch = 500
        rate = 0.1

        net = nnet(netLoc)
        net.train(trainLoc, epoch, rate)
        net.writeFile(outName)
    elif inPutz == "gen":
        genInit()
        exit()
    elif inPutz == "test":
        # netLoc = getFile("Trained net location: ")
        # testLoc = getFile("Testing Set: ")
        # outName = raw_input("Output Filename: ")

        netLoc = "data/sahearts.trained"
        testLoc = "data/test.csv"
        outName = "data/sahearts.res"

        net = nnet(netLoc)
        net.test(testLoc,outName)
    else:
        print("I don't know what you want from me")


