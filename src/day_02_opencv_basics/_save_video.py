import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Codec (video format)
fourcc = cv2.VideoWriter_fourcc(*'XVID') #fourcc stands for four character code and * unpacks the format passed as expected by fourcc function

# Create VideoWriter object
out = cv2.VideoWriter(
    "output.avi", #name of the video file
    fourcc, #compressor
    20.0, #fps
    (640, 480) #window ka size 
)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Save frame
    out.write(frame)

    # Display frame
    cv2.imshow("Recording", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()