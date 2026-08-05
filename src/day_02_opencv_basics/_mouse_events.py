import cv2

img = cv2.imread(r"C:\Users\ummeh\Downloads\cat4.jpg")

def mouse_event(event, x, y, flags, params):

    if event == cv2.EVENT_MOUSEMOVE:
        print("Mouse Moving:", x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        print("Left Button Pressed:", x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        print("Left Button Released:", x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right Button Pressed:", x, y)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_event)

while True:
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()