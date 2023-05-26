import cv2
import csv

# 動画ファイル名と出力CSVファイル名を指定する
video_file = "input.mp4"
output_file = "output.csv"

# 動画を開く
cap = cv2.VideoCapture(video_file)

# 動画のフレームレートを取得する
fps = cap.get(cv2.CAP_PROP_FPS)

# CSVファイルに書き込むためのリストを初期化する
data = []

# 動く点の輪郭を検出するための閾値を設定する
threshold_value = 50

# 初期位置を取得する
ret, frame = cap.read()
if not ret:
    print("Failed to read video file")
    exit()
init_pos = (x, y) = (0, 0)  # ここを初期位置に設定する

# 時間をカウントするための変数を初期化する
time = 0

# 動画を読み込みながら1秒ごとに動く点の座標を取得する
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ここに動く点を検出する処理を実装する
    # 動画のフレームから動く点の座標を取得する
    # グレースケールに変換する
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2値化する
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 輪郭を検出する
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭から動く点の座標を抽出する
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
    
    pos = (x,y)

    # 時間が1秒経過したらCSVファイルに座標を書き込む
    if time % fps == 0:
        print(x,y)
        data.append([time / fps, pos[0] - init_pos[0], pos[1] - init_pos[1]])

    # 時間を更新する
    time += 1

# CSVファイルにデータを書き込む
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y"])
    writer.writerows(data)

# プログラムを終了する
cap.release()
cv2.destroyAllWindows()