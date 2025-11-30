import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def detect_text(image, lang='rus+eng'):
    processed_image = preprocess_image(image)
    
    custom_config = r'''
        --oem 3 
        --psm 8 
    '''
    
    text = pytesseract.image_to_string(processed_image, lang=lang, config=custom_config)

    #matches = re.findall(r'\d', text)
    #return matches[0] if matches else None
    return text