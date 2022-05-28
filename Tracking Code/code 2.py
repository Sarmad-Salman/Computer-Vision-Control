import cv2
import numpy as np
video = cv2.VideoCapture(0)
ok, frame = video.read()
frame = cv2.flip(frame, 1)
cv2.imshow("Video Feed", frame)
roi_extract = cv2.selectROI("Select ROI", frame)
x, y, w, h = (roi_extract[0], roi_extract[1], roi_extract[2], roi_extract[3])
roi = frame[y: y + h, x: x + w]
cv2.imshow("ROI", roi)
roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hist_roiHSV = cv2.calcHist([roiHSV], [0], None, [180], [0, 180])
hist_roiHSV = cv2.normalize(hist_roiHSV, hist_roiHSV, 0, 255, cv2.NORM_MINMAX)
termCriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while video.isOpened():
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([frameHSV], [0], hist_roiHSV, [0, 180], 1)
    _, trackWindow = cv2.meanShift(mask, (x, y, w, h), termCriteria)
    x, y, w, h = trackWindow
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(frame, (x2, y2), 3, (255, 200, 50), -1)
    text = "x: " + str(x2) + ", y: " + str(y2)
    cv2.putText(frame, text, (x2 - 10, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
    cv2.imshow("Video Playback", frame)
    stopKey = cv2.waitKey(60)
    if stopKey == 27:
        break
video.release()
cv2.destroyAllWindows()