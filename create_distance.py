import os
import cv2
import numpy as np
import csv


target_lines = [0, 10, -10, 20, -20, 30, -30, 40, -40]

global _x
global _y
global _touch_flag
_x = 500
_y = 500
_touch_flag = False

# マウスイベント
def __mouse_event(event, x, y, flag, params):
    global _x
    global _y
    global _touch_flag
    _x = x
    _y = y
    if event == cv2.EVENT_LBUTTONDOWN:
        _touch_flag = True

def CreateDistanceImage(input_file, output_file, W = 1920, H = 1080):

    global _x
    global _y
    global _touch_flag

    img = cv2.imread(input_file)
    img = cv2.resize(img, (W, H))

    mask = np.zeros((H, W, 3), np.uint8)

    for aa, axis in enumerate(["y", "x"]):

        tmp = np.ones((H, W), np.uint8) * 255

        for v in target_lines:
            window_name = f"select {axis} = {v} [cm]"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 0, 0)
            cv2.setMouseCallback(window_name, __mouse_event)
            line_points = []

            new_point_flag = False

            while True:
                dst = img.copy()
                if not new_point_flag:
                    line_points.append((_x, _y))
                    new_point_flag = True
                else:
                    line_points[-1] = (_x, _y)

                for xx, yy in line_points:
                    cv2.circle(dst, (xx, yy), 5, (255, 0, 0), -1)
                for i in range(len(line_points) - 1):
                    x1, y1 = line_points[i]
                    x2, y2 = line_points[i + 1]
                    cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 3)

                cv2.imshow(window_name, dst)
                key = cv2.waitKey(3)
                if key == ord('q'):
                    line_points = line_points[:-1]
                    break
                if _touch_flag or key == ord('y'):
                    new_point_flag = False
                    _touch_flag = False

            vv = v + 100
            for i in range(len(line_points) - 1):
                x1, y1 = line_points[i]
                x2, y2 = line_points[i + 1]
                cv2.line(tmp, (x1, y1), (x2, y2), vv, 3)
            img = dst.copy()
            cv2.destroyWindow(window_name)
        
        mask[:, :, aa] = tmp[:, :]
    cv2.imwrite(output_file, mask)


if __name__ == "__main__":
    input_file = "./src/0_line.png"
    output_file = "./src/0_distance.bmp"
    CreateDistanceImage(input_file, output_file)