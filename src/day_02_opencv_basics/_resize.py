import cv2 
img = cv2.imread(r"C:\Users\ummeh\Downloads\cat4.jpg")

''' In order to resize, we should know the size of the image
 so I have to use the shape method'''
print(img.shape)

''' the return values are (height, width,no. of channels( eg:3 = BGR)'''

imgresized = cv2.resize(img, (400,200))
print(imgresized.shape)

#here in resize function, width is the first paramter and height is second

cv2.imshow("original size", img)
cv2.imshow("resized image", imgresized)
cv2.waitKey(0)