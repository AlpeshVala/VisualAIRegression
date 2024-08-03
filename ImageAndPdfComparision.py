import cv2
from pytesseract import pytesseract
from skimage.metrics import structurl_similarity as compare_ssim
from pdf2image import convert_from_path
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pytest
from jproperties import Properties
import pytest_html

#Function to denoise the image
def denoiseImage(imageToBeDenoised):
    denoisedImage = cv2.fastNlMeansDenoisingColored(imageToBeDenoised,None,10,10,7,15)
    plt.subplot(121),plt.imshow(imageToBeDenoised)
    plt.subplot(122),plt.imshow(denoisedImage)
    plt.show()
    return denoisedImage

def extractTextFromImage(testImage):
    # In double quotes below add the path of tesseract.exe
    path_to_tesseract = r""
    pytesseract.tesseract_cmd = path_to_tesseract
    text =pytesseract.image_to_string(testImage,lang="eng")
    print(text[:-1])
    return text

def resizeImage(resizeImage):
    resizedImage = cv2.resize(resizeImage,
                              (int(resizeImage.shape[1] + (resizeImage.shape[1] * .4)),
                               int(resizeImage.shape[0] + (resizeImage.shape[0] * .25))),
                              interpolation = cv2.INTER_AREA)
    updatedImage = cv2.cvtColor(resizedImage,cv2.COLOR_BGR2RGB)
    return updatedImage

def updateImageDimensions(imageToBeResized):
    resizedImage = cv2.resize(imageToBeResized,(1100,550))
    cv2.imwrite('resizedImage.png',resizedImage)
    print('Image is Resized')

def imageMse(imageA,imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imgeA.shape[1])
    return err

def compareImages(imageA,imageB,title):
    mseValue= imageMse(imageA,imageB)
    ssimValue = compare_ssim(imageA,imageB)
    #Pyplot figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f,SSIM: %.2f" % (mseValue,ssimValue))

    #Display the first image
    ax = fig.add_subplot(1,2,1)
    plt.imshow(imageA,cmap = plt.cm.gray)
    plt.axis("off")

    #Display the second image
    ax =fig.add_subplot(1,2,2)
    plt.imshow(imageB,cmap=plt.cm.gray)
    plt.axis("off")

    #Show the images
    plt.show()

def imageDifferences(imageA,imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #Compute the Structural Similarity Index
    (score,diff) = compare_ssim(grayA,grayB,full=True)
    diff = (diff * 255).astype("unit8")
    print("SSIM: {}".format(score))

    #Calculating Threshold and draw the contours
    thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Loop over the contours
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("FirstImage",imageA)
    cv2.imshow("SecondImage",imageB)
    cv2.imshow("Diff",diff)
    cv2.imshow("Thresh",thresh)
    cv2.waitKey(0)

def hide_text_area(original_img, keyword):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,13,3)
    black =np.zeros([gray.shape[0] + 2, gray.shape[1] + 2],np.uint8)
    mask =cv2.floodFill(th.copy(), black, (0,0), 0, 0 , 0, flags=8)[1]
    kernel_length = 15
    horizontal_kernel = cv2.getStructuralElement(cv2.MORPH_RECT,(kernel_length,1))
    dilate = cv2.dilate(mask,horizontal_kernel,iterations=1)
    img2 = original_img.copy()
    contours = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    flag = 0
    for c in contours:
        x, y, w, h =  cv2.boundingRect(c)
        roi = th[y:y+h,x:x + w]
        roi = cv2.fastNlMeansDenoising(roi,None,10,21,7)
        text = pytesseract.image_to_string(roi)
        print(text)
        if keyword.lower() in text.lower():
            flag = 1
            img2 = cv2.rectangle(original_img,(x,y),(x+w , y+h),(0,0,0),-1)
    if flag == 0:
        print("Word not found")
    cv2.imshow("image",cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY))
    cv2.waitKey(0)
    return img2

#Sample Test

@pytest.mark.order1
def validateImagesAndContent():
    #Read the image
    firstImage = cv2.imread("<Give the path of your image>")
    secondImage =  cv2.imread("<Give the path of your image>")

    #compare the images
    compareImages(firstImage,secondImage,"FirstImage vs SecondImage")

    #Highlight the differences using contours
    imageDifferences(firstImage,secondImage)

    #Readthe content of first image
    textFromFirstImage = extractTextFromImage(firstImage)

    # Readthe content of Second image
    textFromSecondImage = extractTextFromImage(secondImage)


