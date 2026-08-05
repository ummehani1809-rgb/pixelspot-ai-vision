import cv2
import numpy as np 

img = np.zeros((512,512,3),np.uint8) #creates a black canvas in matrix like
print(img)

cv2.rectangle(img,(0,0),(250,250),(0,255,0),3)
# rectangle(image variable, point 1, point 2, color,thickness )
cv2.imshow("image",img)
cv2.waitKey(0)