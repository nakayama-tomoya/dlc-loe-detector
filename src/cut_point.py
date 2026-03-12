import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
[[[[464, 264], [774, 554]], [[478, 590], [760, 818]], [
        [1108, 262], [1434, 538]], [[1102, 588], [1420, 852]]],

    [[[308, 20], [573, 287]], [[603, 29], [885, 300]], [
        [303, 317], [560, 572]], [[591, 327], [848, 584]]],

    [[[306, 15], [576, 288]], [[603, 12], [879, 300]], [
        [303, 312], [566, 578]], [[587, 327], [846, 594]]],

    [[[308, 17], [575, 290]], [[593, 23], [878, 302]], [
        [293, 311], [564, 572]], [[585, 321], [836, 585]]],

    [[[222, 48], [367, 193]], [[209, 225], [354, 370]], [[216, 380], [361, 525]]],

    [[[89, 254], [340, 507]], [[378, 269], [626, 510]], [[658, 284], [900, 517]]],

    [[[230, 440], [498, 681]], [[527, 449], [783, 699]], [[812, 462], [1061, 705]]],
    
    [[[51, 50], [386, 363]], [[420, 32], [746, 345]], [[773, 26], [1106, 321]], [
        [80, 407], [405, 722]], [[437, 374], [770, 716]], [[788, 357], [1110, 683]]]
]

def mask(img,threshold=100):
    hsvLower = np.array([20, 50, threshold])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([60, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3)  # クロージング
    kernel = np.ones((5, 5), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=30)  # クロージング
    result = cv2.bitwise_and(img, img, mask=hsv_mask)  # 元画像とマスクを合成
    return result



def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0, 0] - pt0[0, 0])
    dy1 = float(pt1[0, 1] - pt0[0, 1])
    dx2 = float(pt2[0, 0] - pt0[0, 0])
    dy2 = float(pt2[0, 1] - pt0[0, 1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
    return (dx1*dx2 + dy1*dy2) / v

# 画像上の四角形を検出
def findSquares(img, cond_area=1000, threshold=100):
    img=img.copy()
    height, width, channels = img.shape[:3]
    max_area=height*width/4
    image = mask(img, threshold)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,  bin_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 輪郭取得
    contours, _ = cv2.findContours(
        bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cut_points=[]
    for i, cnt in enumerate(contours):
        # 輪郭の周囲に比例する精度で輪郭を近似する
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.02, True)
        # 凸性の確認
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > cond_area and area < max_area and cv2.isContourConvex(approx):
            maxCosine = 0
            for j in range(2, 5):
                # 辺間の角度の最大コサインを算出
                cosine = abs(angle(approx[j % 4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCosine, cosine)

            # すべての角度の余弦定理が小さい場合
            # （すべての角度は約90度です）次に、quandrangeを書き込みます
            # 結果のシーケンスへの頂点
            if maxCosine < 0.3:
                rcnt = approx.reshape(-1, 2)
                point=([width,height],[0,0])
                for i in rcnt:
                    if i[0] < point[0][0]:
                        point[0][0]=i[0]
                    if i[0] > point[1][0]:
                        point[1][0]=i[0]
                    if i[1] < point[0][1]:
                        point[0][1]=i[1]
                    if i[1] > point[1][1]:
                        point[1][1]=i[1]
                cut_points.append(point)
                print(area)
                image=cv2.polylines(img, [rcnt], True, (0, 0, 255),
                              thickness=2, lineType=cv2.LINE_8)
    return image,cut_points

