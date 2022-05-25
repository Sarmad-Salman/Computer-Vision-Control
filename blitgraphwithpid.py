import time
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
mtx = ([3315.65934, 0.00000000, 2218.92013], [0.00000000, 3188.38035, 1322.60141],[0.00000000, 0.00000000, 1.00000000])
fx = mtx[0][0]
cx = mtx[0][2]
fy = mtx[1][1]
cy = mtx[1][2]
z_world = 25

np.set_printoptions(suppress=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
rotation_matrix = np.eye(3)
roi_extract = cv2.selectROI("Select ROI", frame)
cv2.destroyWindow('Select ROI')
x, y, w, h = (roi_extract[0], roi_extract[1], roi_extract[2], roi_extract[3])
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# xx = np.linspace(0, 100, num=100)
# XX, YY = np.meshgrid(xx, xx)
XX, YY = [], []
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

line1, = ax1.plot([], lw=3)
text1 = ax1.text(0.5, 0.9, "")
line2, = ax2.plot([], lw=3)
line3, = ax3.plot([], lw=3)
line4, = ax4.plot([], lw=3)

# ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim([-620, 620])
ax1.set_title('X-Error')
# ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim([-360, 360])
ax2.set_title('Y-Error')

ax3.set_ylim([-420, 420])
ax3.set_title('X-PID')
ax4.set_ylim([-420, 420])
ax4.set_title('Y-PID')

fig.canvas.draw()   # note that the first draw comes before setting data

ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
ax2background = fig.canvas.copy_from_bbox(ax2.bbox)
ax3background = fig.canvas.copy_from_bbox(ax3.bbox)
ax4background = fig.canvas.copy_from_bbox(ax4.bbox)

plt.show(block=False)

t_start = time.time()
i = 1
x_ei = []
y_ei = []
ii = []

x_kp = 3.55
y_kp = 3.55
x_ki = 0.02
y_ki = 0.00
x_kd = 3.00
y_kd = 3.00
x_pid_i = 0.00
y_pid_i = 0.00
x_pid_t = []
y_pid_t = []
x_e = 0.00
y_e = 0.00
prev = 0

while(1):

    print(i)
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret == True:
        ret, frame1 = cv2.threshold(frame, 180, 155, cv2.THRESH_TOZERO_INV)
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, criteria)
        x, y, w, h = track_window
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        s = 1
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        u = x + int(w/2)
        v = y + int(h/2)
        cv2.circle(frame, (u, v), 3, (255, 200, 50), -1)
        text = "x: " + str(u) + ", y: " + str(v)
        cv2.putText(frame, text, (u - 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)
        cv2.imshow('Tracking Video', img2)
        x_world = round((u - cx) * z_world / fx, 2)
        y_world = round((v - cy) * z_world / fy, 2)
        world_coordinates = [x_world, y_world, z_world]

        ############################################################################################################################
        # x rightwards  y downwards

        x_prev_e = x_e
        y_prev_e = y_e
        
        now = time.time()

        x_e = float(u - 620)
        y_e = float(360-v)
        x_ei.append(x_e)
        y_ei.append(y_e)
        ii.append(i)

        x_pid_p = x_kp * x_e
        y_pid_p = y_kp * y_e
        # x_pid_p = 0
        # y_pid_p = 0

        if -3 < x_e < 3:
            x_pid_i = x_pid_i + (x_ki * x_e * 0)
        if -3 < y_e < 3:
            y_pid_i = y_pid_i + (y_ki * y_e)

        x_pid_d = x_kd * (x_e - x_prev_e) / (now-prev)
        y_pid_d = y_kd * (y_e - y_prev_e) / (now-prev)

        x_pid = x_pid_p + x_pid_i - x_pid_d
        y_pid = y_pid_p + y_pid_i - y_pid_d

        if x_pid > 400:
            x_pid = 400
        elif x_pid < -400:
            x_pid = -400

        if y_pid > 400:
            y_pid = 400
        elif y_pid < -400:
            y_pid = -400

        x_pid_t.append(x_pid)
        y_pid_t.append(y_pid)

        if i <= 100:
            ax1.set_xlim(1, i)
            ax2.set_xlim(1, i)
            ax3.set_xlim(1, i)
            ax4.set_xlim(1, i)
        else:
            ax1.set_xlim(i-100, i)
            ax2.set_xlim(i-100, i)
            ax3.set_xlim(i-100, i)
            ax4.set_xlim(i-100, i)
  
        line1.set_data(ii, x_ei)
        line2.set_data(ii, y_ei)
        line3.set_data(ii, x_pid_t)
        line4.set_data(ii, y_pid_t)

        tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format( fps=((i+1) / (time.time() - t_start)))
        text1.set_text(tx)
            
        # restore background
        fig.canvas.restore_region(ax1background)
        fig.canvas.restore_region(ax2background)
        fig.canvas.restore_region(ax3background)
        fig.canvas.restore_region(ax4background)

        # redraw just the points
        ax1.draw_artist(line1)
        ax2.draw_artist(line2)
        ax3.draw_artist(line3)
        ax4.draw_artist(line4)
        ax1.draw_artist(text1)

        # fill in the axes rectangle
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)
        fig.canvas.blit(ax3.bbox)
        fig.canvas.blit(ax4.bbox)

        prev = now

        fig.canvas.flush_events()
        i = i+1

        k = cv2.waitKey(30)
        if k == 27:
            break
    else:
        break

plt.show()
# cap.release()
# cv2.destroyAllWindows()
