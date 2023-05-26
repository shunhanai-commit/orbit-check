import os
import cv2
import numpy as np

f_input = ".src"
f_output = "./dst/"
visualize_color = (255, 0, 0)
THRESH = 70   # 問題なかったので使用
files = [f for f in os.listdir(f_input) if '.mp4' in f]

for file in files:
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