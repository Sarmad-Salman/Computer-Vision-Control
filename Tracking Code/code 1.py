'''
Using real-time streaming video from your built-in webcam, this program:
  - Creates a bounding box around a moving object
  - Calculates the coordinates of the centroid of the object
  - Tracks the centroid of the object
'''
import cv2
import numpy as np


def main():

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)
    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((10, 10), np.uint8)
    while(True):
        # Capture frame-by-frame
        # This method returns True/False as well as the video frame.
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Use every frame to calculate the foreground mask and update the background
        fg_mask = back_sub.apply(frame)
        # Close dark gaps in foreground object using closing
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # Remove salt and pepper noise with a median filter
        fg_mask = cv2.medianBlur(fg_mask, 5)
        # Threshold the image to make it either black or white
        _ , fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(
            fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        # If there are no countours
        if len(areas) < 1:
            # Display the resulting frame
            cv2.imshow('frame', frame)
            # If "q" is pressed on the keyboard, exit this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Go to the top of the while loop
            continue
        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)
        # Draw the bounding box
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # Draw circle in the center of the bounding box
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame, (x2, y2), 4, (0, 255, 0), -1)
        # Print the centroid coordinates (we'll use the center of the bounding box) on the image
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the resulting frame
       # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
     #set your desired size
        #cv2.resizeWindow('frame', 1920, 1080)
        cv2.imshow('frame', frame)
        # If "q" is pressed on the keyboard, exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #print(_doc_)
    main()
