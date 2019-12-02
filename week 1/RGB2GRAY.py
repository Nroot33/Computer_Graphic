import cv2
import numpy as np

def bgr2gray(bgr):
    b, g, r = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]
    gray =  (0.1140 * b) + (0.5870 * g) + (0.2989 * r)
    return gray

image = cv2.imread('./naeun.jpg')
image = bgr2gray(image)
image = image.astype(np.uint8)
cv2.imshow('picture',image)
cv2.waitKey(0)
cv2.destroyAllWindows()



