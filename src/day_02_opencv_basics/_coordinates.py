import cv2

def show_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")

        text = f"({x}, {y})"

        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

        cv2.imshow("Image", img)


img = cv2.imread(r"C:\Users\ummeh\Downloads\cat4.jpg")
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_coordinates)
while True:
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

''' namedwindow is used to create a window for oupput, 
we can also skip it as imshow creates a window by default
by changing the order of code '''