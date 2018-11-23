import numpy as np
import cv2
import math
import subprocess
from skimage.transform import resize
from scipy.misc import imresize

IMG_PATH = "./Images/SIRE-"
OUTPUT_PATH = "./output/"

def displayImg(img, windowname = "window"):
    cv2.imshow(windowname, img)
    cv2.waitKey(0)

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

def resizeImage(img, width, height):
    newImage = np.zeros(shape=img.shape)
    count = 0
    currentMin = width
    for i in range(0, height, 5):
        for j in range(0, width):
            if (img[i,j] < 255 or j > currentMin):
                currentMin = min(currentMin, j)
                newImage[i:i+5] = cv2.resize(img[i: i+5, j: width - j], (width, 5), cv2.INTER_CUBIC)
                break
    print(count)
    displayImg(img)
    displayImg(newImage)

def cropImage(img):
    mascara = (img<255)*1
#     print(mascara)
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
    print(TamanhoNovaImagem)
    if (last_vecw < H-1):
        for i in range(last_vecw, H-1):
            new_vecw = np.append(new_vecw, new_vecw[last_vecw-1])
    for i in range(0, H-1):   
        li = resize(img[i,:], (int(W+2*(max(new_vecw) - new_vecw[i])),1))
        lisize = li.shape[0]
        ip[i,int(round((TamanhoNovaImagem-lisize)/2)):int(round((TamanhoNovaImagem-lisize)/2)+lisize)] = li.T; 
    return (ip)

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

def tranformFingerprint():
    imgpath = IMG_PATH + "1.bmp"
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    equalizedImg = localHistogramEqualization(img)
    imgWithGamma = applyGamma(equalizedImg)
    imgsWithGauss = applyGaussFilter(imgWithGamma)
    binaryMean = saveGauss(imgsWithGauss)
    height, width = binaryMean.shape
    cropImg = cropImage(binaryMean)
    imgDistortion = geomDistortion(cropImg, 5)
    displayImg(imgDistortion)
    # res = cv2.resize(binaryMean,(2*width, 2*height), interpolation = cv2._INPUT_ARRAY_STD_VECTOR_MAT)
    #resizeImage(binaryMean, width, height)
    # displayImg(res)
    # result = np.polyfit([i for i in range(0, binaryMean.shape[1])], binaryMean[0], 5)
    # displayImg(binaryMean)

def main():
    tranformFingerprint()
    # displayBinatyImages()

main()
