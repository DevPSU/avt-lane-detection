import cv2
import numpy as np
import os


def canny(image):
    # create image variable
    img = cv2.imread(image)
    # convert image to grayscale color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny grayscale image 
    canny = cv2.Canny(gray, 50,150)
    return canny


# Show image
cv2.imshow('image', canny('avt-lane-detection/talos-nd01 copy/1568087426_1388_-9.600000000000001_12.28v_30.0c_frontlower.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()
