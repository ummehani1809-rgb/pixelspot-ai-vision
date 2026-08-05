import cv2 
img = cv2.imread(r"C:\Users\ummeh\Downloads\cat4.jpg")
cv2.imshow("cat",img)
cv2.waitKey(0)

# we use cv2.imread () method to capture images adn the image path is passed to it 
# imshow() method is used to display the images and the variable in which the image is stored is passed to it 
#waitkey is to delay, 0 gives infinite delay and any other value is passed is milliseconds 
