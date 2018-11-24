import numpy as np
import cv2
import math
import subprocess
import random
from skimage.transform import resize
from scipy.misc import imresize

IMG_PATH = "./Images/SIRE-"
OUTPUT_PATH = "./output/"
TEMP_OUTPUT = "./temp/"

def displayImg(img, windowname = "window"):
    cv2.imshow(windowname, img)
    cv2.waitKey(0)

def saveImgsTemp(imgArr):
    for i, (img) in enumerate(imgArr):
    # displayImg(cv2.resize(bubbledImg, (int(width/2), int(height/2))))
        cv2.imwrite(TEMP_OUTPUT + str(i) + ".png", img)
        cv2.imwrite(TEMP_OUTPUT + str(i) + ".min.png", cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))

def applyGamma(img):
    gamma = 1.5
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return np.uint8(0.75 * cv2.LUT(img, table))

def applyGaussFilter(img):
    filterSizes = [3, 5, 7, 9]
    return [(cv2.GaussianBlur(img, (fSize, fSize), cv2.BORDER_DEFAULT), fSize/(2 * math.sqrt(2 * math.log(2)))) for fSize in filterSizes]

def saveGauss(imgsWithGauss):
    # imagem onde se calculara a media das imagens binarizadas
    imgMean = np.zeros(shape = imgsWithGauss[0][0].shape)
    for i, (item) in enumerate(imgsWithGauss):
        IMG_NAME = "gauss-"+str(i)
        IMG_FULL_NAME = OUTPUT_PATH+IMG_NAME

        # cria o diretorio que ira armazenar os arquivos do valor corrente do filtro gaussiano
        bashCommand = "mkdir "+IMG_FULL_NAME
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # salva a imagem filtrada
        cv2.imwrite(IMG_FULL_NAME+"/"+IMG_NAME+".png", item[0])

        # executa o software que faz a binariazacao
        bashCommand = "mindtct " + IMG_FULL_NAME+"/"+IMG_NAME+".png " + IMG_FULL_NAME+"/"+IMG_NAME
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # tranforma de binary raw para png
        result = saveBinary(IMG_FULL_NAME+"/"+IMG_NAME + ".brw", item[0].shape)
        imgMean += result
    # retorna a media das imagens binarizadas
    return np.uint8(imgMean / 4)

def saveBinary(imgPath, imgShape):
    fd = open(imgPath, 'rb')
    rows = imgShape[0]
    cols = imgShape[1]
    f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
    im = f.reshape((rows, cols)) #notice row, column format
    fd.close()
    cv2.imwrite(imgPath[:-4]+"_binary.png", im)
    return im

def localHistogramEqualization(img):
    newImage = np.zeros(shape = img.shape)
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            newImage[i: i+8, j:j+8] = cv2.equalizeHist(img[i:i+8, j:j+8])
    return np.uint8(newImage)

def cropImage(img):
    mascara = (img<255)*1
    i, j = np.where(mascara == 0)
    maxH = max(i);
    minH = min(i);
    maxW = max(j);
    minW = min(j);
    return img[minH:maxH,minW:maxW]

def geomDistortion(img, N):
    H, W = img.shape
    new_vecw = detectPolyn(img, N)
    TamanhoNovaImagem = np.ceil(W+2*(max(new_vecw)-min(new_vecw)))+1
    ip = np.ones(shape = [H, int(round(TamanhoNovaImagem))])*255
    last_vecw = new_vecw.shape[0]

    if (last_vecw < H-1):
        for i in range(last_vecw, H-1):
            new_vecw = np.append(new_vecw, new_vecw[last_vecw-1])
    for i in range(0, H-1):
        li = resize(np.array([img[i,:]]).T, (int(W+2*(max(new_vecw) - new_vecw[i])),1))
        lisize = li.shape[0]
        ip[i,int(round((TamanhoNovaImagem-lisize)/2)):int(round((TamanhoNovaImagem-lisize)/2)+lisize)] = li.T;

    offset = int((ip.shape[1]-(W))/2)

    return (ip[:, offset:offset + W])

def applyTreshHold(img, ths = 127):
    return (np.array([np.array([np.uint8(pixel) if pixel <= ths else np.uint8(255) for pixel in line]) for line in img]),
            np.array([np.array([np.uint8(pixel) if pixel >= ths else np.uint8(255) for pixel in line]) for line in img]))

def getProbability(currentPlace, center, ths = 0.5):
    scaleFactor = 1 / (1 + abs(currentPlace - center)/300)
    # print(currentPlace, scaleFactor)
    randNum = random.random() * scaleFactor
    return randNum < ths

def getRandomElipse(template):
    x, y = template.shape
    center = (np.ceil(x/2), np.ceil(y/2))
    majorAxis = random.randint(1, 5)
    minorAxis = random.randint(1, 3)
    angulo = random.randint(0, 180)
    # color = random.randint(0, 255)
    color = 255
    cv2.ellipse(template, (math.ceil(x/2), math.ceil(y/2)), (majorAxis, minorAxis), angulo, 0,360, color, -1)
    return np.uint8(template)

def applyRandomPatherns(img):
    height, width = img.shape
    center = int(width/2)
    newImage = img.copy()
    windowSize = 5
    probArray = []
    for i in range(0, height, windowSize):
        for j in range(0, width, windowSize):
            probabilitie = getProbability(j, center)
            if probabilitie:
                probArray.append((i,j))
                elipse = getRandomElipse(img[i: i+windowSize, j:j+windowSize])
                newImage[i:i+windowSize, j:j+windowSize] += elipse

    # print(probArray)
    return np.uint8(newImage)

def customSum(img1, img2):
    height, width = img1.shape
    newImage = np.zeros(shape=img1.shape)
    for i in range(0, height):
        for j in range(0, width):
            newImage[i, j] = min(img1[i,j], img2[i, j])
    return newImage

def detectPolyn(img, N):
    H, W = img.shape
    imf = np.uint8(np.ones(shape = img.shape)*255)
    vech = []
    vecw = []
    for i in range(0, H):
        for j in range(0, W):
            if (img[i,j] != 255):
                imf[i,j] = 0
                vech.append(i)
                vecw.append(j)
                break
    vech = np.asarray(vech)
    vecw = np.asarray(vecw)
    pol = np.polyfit(vech, -1*(vecw)+H,N)
    return np.polyval(pol, vech)

def scoreImages(imgsPath):
    print("Cheguei aqui")

def tranformFingerprint():
    imgpath = IMG_PATH + "1.bmp"
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    equalizedImg = localHistogramEqualization(img)
    imgWithGamma = applyGamma(equalizedImg)
    imgsWithGauss = applyGaussFilter(imgWithGamma)
    binaryMean = saveGauss(imgsWithGauss)
    height, width = binaryMean.shape

    cropImg = cropImage(binaryMean)
    imgDistortion = np.uint8(geomDistortion(cropImg, 5) * 255)
    thresholdLowImg, thresholdHighImg = applyTreshHold(imgDistortion, 80)
    bubbledImg = applyRandomPatherns(thresholdLowImg)

    # result = customSum(bubbledImg, thresholdHighImg)
    result = cv2.addWeighted(bubbledImg, 0.9, thresholdHighImg, 0.2, 0)
    saveImgsTemp([bubbledImg, thresholdHighImg, result])
    # displayImg(cv2.resize(bubbledImg, (int(width/2), int(height/2))))

def main():
    tranformFingerprint()
    # eli = np.zeros(shape=(512, 512))
    # cv2.ellipse(eli,(256,256),(50,30),0,0,360,255,4)
    # displayImg(eli)

main()
