import cv2

vid = cv2.VideoCapture(0) # here instead of passig the path of video, we just pass the id of the webcam, if it a default webcam or laptop webcam
# , pass 0, or pass 1,2 and so on according to the webcams connected 

#we can also defne few paramteres using .set() method to define the size of the output window 
# id 3 for width and id 4 for height , 10 for brightness 
vid.set(3,540)
vid.set(4,500)
vid.set(10,600)

while True:
    success, img = vid.read()
    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

    #the rest of the program is same as video capturing 