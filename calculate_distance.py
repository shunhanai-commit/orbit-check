import os
import cv2
import numpy as np


class MyDistance:
    def __init__(self, mask_filename = "hoge"):
        self.mask = None
        self.mask = cv2.imread(mask_filename)

    def __estimate_value(self, v):
        # 10刻みになるように四捨五入
        return (int)(v / 10 + 0.5) * 10 - 100
    
    def get_x_distance(self, x, y):
        right = x
        left = x
        while self.mask[y, right, 1] == 255:
            right += 1
        v1 = self.__estimate_value(self.mask[y, right, 1])
        while self.mask[y, left, 1] == 255:
            left -= 1
        v2 = self.__estimate_value(self.mask[y, left, 1])
        if v1 == v2:
            xcm = v1
        elif (v1 - v2) == 10:
            xcm = v2 + (v1 - v2) * (x - left) / (right - left)
        else:
            print("[Error]")
            xcm = 9999
        return xcm

    def get_y_distance(self, x, y):
        top = y
        bottom = y
        while self.mask[bottom, x, 0] == 255:
            bottom += 1
        v1 = self.__estimate_value(self.mask[bottom, x, 0])
        while self.mask[top, x, 0] == 255:
            top -= 1
        v2 = self.__estimate_value(self.mask[top, x, 0])
        if v1 == v2:
            ycm = v1
        elif (v2 - v1) == 10:
            ycm = v1 + (v2 - v1) * (y - bottom) / (top - bottom)
        else:
            print("[Error]")
            ycm = 9999
        return ycm

    def get_distance(self, x, y):
        xcm = self.get_x_distance(x, y)
        ycm = self.get_y_distance(x, y)
        return xcm, ycm

f_input = "./src/"
f_output = "./dst/"
visualize_color = (255, 0, 0)
THRESH = 70   # 問題なかったので使用
files = [f for f in os.listdir(f_input) if '.mp4' in f]
mask_files = ["./src/0_distance.bmp"]

for ff, file in enumerate(files):

    md = MyDistance(mask_files[ff])

    if "なし" in file:
        visualize_color = (255, 0, 0)
    else:
        visualize_color = (0, 0, 255)

    cap = cv2.VideoCapture(f_input + file)
    width = (int)(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = (int)(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (int)(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')   
    total_count = (int)(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(f_output + file[:-4] + "_dst.mp4", fourcc, fps, (width, height))
    traj = np.zeros((height, width), np.uint8)  # 手の軌跡の画像
    cpos_history = []  # フレームiのレーザポインタ中心の座標

    datas = []

    # 初期の点を探す
    ret, init = cap.read()
    r0 = init[:, :, 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, tmp = cap.read()
    r1 = tmp[:, :, 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, tmp = cap.read()
    r2 = tmp[:, :, 2]
    # 差分抽出
    diff1 = cv2.absdiff(r0, r1)
    diff2 = cv2.absdiff(r0, r2)
    init_diff = cv2.bitwise_and(diff1, diff2)
    # init_diff = cv2.dilate(init_diff, np.ones((3,3),np.uint8), iterations = 1)
    red_poss = np.where(init_diff > THRESH)
    tmp = np.median(red_poss, 1)
    cpos = ((int)(tmp[1]), (int)(tmp[0]))
    cpos_history.append(cpos)
    
    i = 0
    while True:
        i += 1
        if i % 100 == 0:
            print(f"{i:5} / {total_count}")


        ret, img = cap.read()
        if not ret:
            break
        r = img[:,:,2]

        # 差分抽出
        diff1 = cv2.absdiff(r, r0)
        diff = cv2.bitwise_xor(diff1, init_diff)

        red_poss = np.where(diff > THRESH)
        if len(red_poss[0]) == 0:
            cpos = cpos_history[-1]
        else:
            tmp = np.median(red_poss, 1)
            cpos = ((int)(tmp[1]), (int)(tmp[0]))
            if cpos_history[-1][1] < 400 and np.abs(cpos[1] - cpos_history[0][1]) < 10:
                cpos = cpos_history[-1]

        cpos_history.append(cpos)
        # calc phisical distance
        xcm, ycm = md.get_distance(cpos[0], cpos[1])


        
        if i % 10 == 0:
            print([i, cpos[0], cpos[1], xcm, ycm])

        p1 = cpos_history[-2]
        p2 = cpos_history[-1]
        cv2.line(traj, p1, p2, 255, 1, 1)

        imbool = np.where(traj > 254)
        img[imbool] = visualize_color

        cv2.imshow("traj", cv2.resize(traj, (300, 300)))
        cv2.imshow("dst", cv2.resize(img, (300, 300)))
        cv2.waitKey(1)

        writer.write(img)

    writer.release()
    cap.release()

    import csv
    with open(f_output + file[:-4] + "_log_time.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "xpx", "ypx", "xcm", "ycm"])
        for d in datas:
            writer.writerow(d)