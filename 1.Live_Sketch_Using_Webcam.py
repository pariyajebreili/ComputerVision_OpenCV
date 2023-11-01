import cv2
import numpy as np

#sketch generating function
def sketch(image):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean up image using Guassian Blur
    # smooths the image by averaging pixel values in a local neighborhood.
    # helps in reducing high-frequency noise and small variations in the image, which can be seen as small, unwanted details or fluctuations.
    # obtain more stable and accurate edge maps.
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    
    # Do an invert binarize the image 
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)


if not cap.isOpened() and cap2.isOpened():
    print("Error: Could not open the webcam.")
    exit()



while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from the webcam.")
        break

    cv2.imshow('Live_image', (frame))
    cv2.imshow('Live_Sketch', sketch(frame))

    key = cv2.waitKey(1)
    if key==ord('q'):      # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to save the sketch as an image
        cv2.imwrite("sketch.png", sketch(frame))
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()      