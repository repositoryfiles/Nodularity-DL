# coding: utf-8
import datetime
import io
import os
import sys
import tkinter
from tkinter import filedialog

import cv2
import numpy as np
from keras.models import load_model

# VSCで日本語表示するためのおまじない
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 環境設定
iDir = "C:/Data"  # 画像ファイルが格納されているフォルダ
pic_width = 1920  # 入力画像のサイズによらず、画像処理や出力画像はこの幅に設定
# pic_height（高さ）は入力画像の幅と高さの比から計算
min_grainsize = 0.0071  # 画像の幅に対する黒鉛の最小長さ（撮影した画像に応じて設定が必要）
# min_grainsize=0.007はサンプル画像に対する値である。
# サンプル画像は幅142mmに表示させると、倍率100倍の組織画像になる。
# この場合、黒鉛の最小長さ（10μm）は1mmとなる。1mm÷142mm=0.007→min_grainsize
width = 224  # 推論時の黒鉛の画像サイズ


# contoursからmin_grainsize未満の小さい輪郭と、画像の端に接している輪郭を除いてcoutours1に格納
def select_contours(contours, pic_width, pic_height, min_grainsize):
    contours1 = []
    for e, cnt in enumerate(contours):
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
        (x_circle, y_circle), radius_circle = cv2.minEnclosingCircle(cnt)
        if (
            int(pic_width * min_grainsize) <= 2 * radius_circle
            and 0 < int(x_rect)
            and 0 < int(y_rect)
            and int(x_rect + w_rect) < pic_width
            and int(y_rect + h_rect) < pic_height
        ):
            contours1.append(cnt)
    return contours1

# ダイアログ形式によるファイル選択
def get_picture_filenames():
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("jpg", "*.jpg"), ("BMP", "*.bmp"), ("png", "*.png"), ("tiff", "*.tif")]
    filenames = filedialog.askopenfilenames(
        title="画像ファイルを選んでください", filetypes=fTyp, initialdir=iDir
    )
    return filenames

def main():

    Num1 = []
    Num2 = []
    Num3 = []
    Num4 = []
    Num5 = []
    Num6 = []
    Nodularity_ISO = []

    model_path = "C:/Data/TeachableMachineModel/keras_model.h5"
    label_path = "C:/Data/TeachableMachineModel/labels.txt"

    # 保存したモデルとラベルをロードする
    model = load_model(model_path)
    labels = open(label_path, "r").readlines()

    # 画像ファイル名の取り込み
    filenames = get_picture_filenames()
    if filenames == "":
        sys.exit()

    for filename in filenames:

        img_color = cv2.imread(filename)  # カラーで出力表示させるためカラーで読み込み
        img_height, img_width, channel = img_color.shape  # 画像のサイズ取得

        # 画像処理や出力画像のサイズ計算（pic_width, pic_height）
        pic_height = int(pic_width * img_height / img_width)
        img_color = cv2.resize(img_color, (pic_width, pic_height))  # 画像のサイズ変換
        img_color1 = img_color.copy()

        # カラー→グレー変換、白黒反転の二値化、輪郭の検出、球状化率の評価に用いる輪郭の選別
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        ret, img_inv_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(img_inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        num_1 = num_2 = num_3 = num_4 = num_5 = num_6 = 0
        sum_graphite_areas = 0
        sum_graphite_areas_5and6 = 0

        # 必要な輪郭のみcontours1に格納
        contours1 = select_contours(contours, pic_width, pic_height, min_grainsize)  # 球状化率の評価に用いる輪郭をcoutours1に格納

        for e, cnt in enumerate(contours1):
            x, y, w, h = cv2.boundingRect(cnt)
            img = img_color[y : y + h, x : x + w]
            tmp = img[:, :]
            hei, wid = img.shape[:2]
            if hei > wid:
                size = hei
                limit = wid
            else:
                size = wid
                limit = hei
            start = int((size - limit) / 2)
            fin = int((size + limit) / 2)
            img1 = np.zeros((size, size, 3), np.uint8)

            cv2.rectangle(img1, (0, 0), (size, size), (255, 255, 255), thickness=-1, lineType=cv2.LINE_4)

            if size == hei:
                img1[:, start:fin] = tmp
            else:
                img1[start:fin, :] = tmp

            img = cv2.resize(img, (width, width))

            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1  # -1～1

            # 予測
            probabilities = model.predict(img, verbose=0)
            pred = labels[np.argmax(probabilities)]

            graphite_area = cv2.contourArea(cnt)
            sum_graphite_areas += graphite_area

            if pred[2] == "1":
                num_1 = num_1 + 1
                cv2.drawContours(img_color1, contours1, e, (128, 0, 0), -1)  # 紺
            if pred[2] == "2":
                num_2 = num_2 + 1
                cv2.drawContours(img_color1, contours1, e, (255, 0, 0), -1)  # 青
            if pred[2] == "3":
                num_3 = num_3 + 1
                cv2.drawContours(img_color1, contours1, e, (0, 0, 128), -1)  # 茶
            if pred[2] == "4":
                num_4 = num_4 + 1
                cv2.drawContours(img_color1, contours1, e, (128, 0, 128), -1)  # 紫
            if pred[2] == "5":
                num_5 = num_5 + 1
                sum_graphite_areas_5and6 += graphite_area
                cv2.drawContours(img_color1, contours1, e, (0, 0, 255), -1)  # 赤
            if pred[2] == "6":
                num_6 = num_6 + 1
                sum_graphite_areas_5and6 += graphite_area
                cv2.drawContours(img_color1, contours1, e, (128, 128, 0), -1)  # 青緑

        Num1.append(num_1)
        Num2.append(num_2)
        Num3.append(num_3)
        Num4.append(num_4)
        Num5.append(num_5)
        Num6.append(num_6)

        # 球状化率（ISO法）
        Nodularity_ISO.append(sum_graphite_areas_5and6 / sum_graphite_areas * 100)

        # 結果のファイル保存
        src = filename
        idx = src.rfind(r".")
        result = src[:idx] + "_res." + src[idx + 1 :]
        cv2.imwrite(result, img_color1)

    now = datetime.datetime.now()
    file_path = str(os.path.dirname(filenames[0])) + "/result_{0:%Y%m%d%H%M}".format(now) + ".txt"
    with open(file_path, mode='w') as f2:
        for i in range(len(filenames)):
            print("{}, {}, {}, {}, {}, {}, {}, {}".format(str(filenames[i]), Num1[i], Num2[i], Num3[i], Num4[i], Num5[i], Num6[i], Nodularity_ISO[i], ), file=f2, )  # ファイル名

if __name__ == "__main__":
    main()
