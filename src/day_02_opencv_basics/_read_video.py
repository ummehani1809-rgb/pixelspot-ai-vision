import cv2 

vid = cv2.VideoCapture(r"C:\Users\ummeh\OneDrive\Pictures\Camera Roll\WIN_20260727_19_31_43_Pro.mp4")

while True:
    correct, img = vid.read()
    cv2.imshow("video",img)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break 


'''note for myself:
ord function ives back the ASCII values of characters passed to it. 
0xFF gives the last 8 bits of binary value which is most significant
correct variable stores boolean values if the image has been successfully captured in that particular loop'''
