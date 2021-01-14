import easyocr
import cv2
import numpy as np
#preporcessing image before OCR

def denoise(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 32)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize(image):
   return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
image = cv2.imread('test.png')
resi = resize(image)
dst = cv2.fastNlMeansDenoisingColored(resi, None, 10, 10, 7, 15) 
gray = get_grayscale(dst)
smoo = denoise(gray)
reader = easyocr.Reader(['fr']) 
result = reader.readtext( smoo, detail = 0)
print(result)

