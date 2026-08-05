import cv2 
img = cv2.imread(r"C:\Users\ummeh\Downloads\cat4.jpg")
print(img.shape)

imgcropped = img[0:100,0:500]

cv2.imshow("output",img)
cv2.imshow("cropped output", imgcropped)
cv2.waitKey(0)

