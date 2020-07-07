from PIL import Image, ImageOps
from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pylab as plot


def imageToValueArray(imagePath):
    img = Image.open(imagePath, 'r').convert("L")
    croppedImg = ImageOps.fit(img, (32, 32))
    # croppedImg.show()
    dataArray = list(croppedImg.getdata())
    dataArrayFloats = []
    for n in dataArray:
        dataArrayFloats.append(n/255.0)
    print(dataArray)
    print(dataArrayFloats)


def loadMNISTData(imageFilePath, labelFilePath):
    x, y = loadlocal_mnist(
        images_path=imageFilePath,
        labels_path=labelFilePath)
    print(len(x[0]))
    print(len(y))
    print((x[0]))
    '''eL = []
    for n in range(0, 28):
        pRL = []
        for j in range(0, 28):
            print("%3d" % (x[0][(n * 28) + j]), end=' ')
            pRL.append(x[0][(n * 28) + j])
        print("")
        eL.append(pRL)'''
    floatArr = []
    for n in x[0]:
        floatArr.append(n/255.0)
    return floatArr


def loadMNISTDataConv(imageFilePath, labelFilePath):
    x, y = loadlocal_mnist(
        images_path=imageFilePath,
        labels_path=labelFilePath)
    print(len(x[0]))
    print(len(y))
    print((x[0]))
    eL = []
    for n in range(0, 28):
        pRL = []
        for j in range(0, 28):
            pRL.append(x[0][(n * 28) + j])
        eL.append(pRL)
    floatArr = []
    for n in x[0]:
        floatArr.append(n/255.0)
    return floatArr


def showImage(image):
    eL = []
    for n in range(0, 28):
        pRL = []
        for j in range(0, 28):
            pRL.append(image[(n * 28) + j])
        eL.append(pRL)
    plot.imshow(eL, cmap=plot.cm.binary)
    plot.show()


def loadMNISTDataArray(imageFilePath, labelFilePath):
    x, y = loadlocal_mnist(
        images_path=imageFilePath,
        labels_path=labelFilePath)
    return x, y


# not used, worth a shot


def readNISTLabels(labelFilePath):
    file = open(labelFilePath, "rb")
    c = 0
    labelList = []
    with file as f:
        byte = f.read(1)
        while byte:
            # Do stuff with byte.
            num = str(byte)[4:6]
            labelList.append(num)
            c += 1
            byte = f.read(1)
    print(len(labelList))
    print(labelList[0:10])

# not used, worth a shot


def readNISTImages(imageFilePath):
    file = open(imageFilePath, "rb")
    c = 0
    pixelList = []
    with file as f:
        byte = f.read(1)
        while byte:
            # Do stuff with byte.
            num = str(byte)[4:6]
            pixelList.append(num)
            c += 1
            byte = f.read(1)
    pixelList = pixelList[16:]
    print(len(pixelList))
    print(pixelList[0:784])
