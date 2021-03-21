# OpenCV program to perform Edge detection in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2

# np is an alias pointing to numpy library
import numpy as np


# capture frames from a camera
cap = cv2.VideoCapture(1)


# loop runs if capturing has been initialized
while(1):

    # reads frames from a camera
    ret, frame = cap.read()

    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    lower = np.array([12, 40, 60])
    upper = np.array([30, 255, 255])
    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    amask = cv2.inRange(hsv,lower,upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    rest = cv2.bitwise_and(frame,frame, mask=amask)

    # Display an original image
    cv2.imshow('Original',frame)

    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(frame,100,200)
    edge = cv2.Canny(rest,100,200)

    # Display edges in a frame
    cv2.imshow('Edges',edges)
    cv2.imshow('Edes',rest)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


        # Close the window
        cap.release()

        # De-allocate any associated memory usage
        cv2.destroyAllWindows()
