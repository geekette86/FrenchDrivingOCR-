import easyocr
import cv2
import numpy as np
import re
#preporcessing image before OCR 3;1/3,3/3;4$/312/
import pytesseract
custom_config = r'--oem 1 --psm 4'
def denoise(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 32)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    or_image = cv2.bitwise_or(img, closing)
    return or_image
def replace_chars(text):
    pattern = '[^a-zA-Z0-9 \n\.]'
    return re.sub(pattern, '', text)
def date_finder(dateme):
    return re.search('\d{2}.\d{2}.\d{4}', dateme)
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize(image):
   return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
image = cv2.imread('test.png')
resi = resize(image)
dst = cv2.fastNlMeansDenoisingColored(resi, None, 10, 10, 7, 15) 
gray = get_grayscale(dst)
smoo = denoise(gray)
result = pytesseract.image_to_string(smoo, config=custom_config)
print(date_finder(result))
print (replace_chars(result))

