import numpy as np
import pandas as pd
import cv2
import math
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def get_informations(image_path, require_thresh = False):
    # print(image_path)
    origin_image = cv2.imread(image_path)
    image = 255 - origin_image[ :, :, 1]
    rectangle = 255 - cv2.imread('D:/code/.contest/dgnl/assets/test.png', 0)
    small_rect = rectangle.copy()
    rectangle = cv2.resize(rectangle, (50, 50))
    # rectangle = np.full((25, 25), 255, dtype='int')
    # cv2.imwrite("rectangle.png", rectangle)
    circle = cv2.imread('D:/code/.contest/dgnl/assets/choice.png', 0)
    h, w = rectangle.shape

    origin_image_ = origin_image.copy()
    image_ = image.copy()
    res = cv2.matchTemplate(image_,rectangle,cv2.TM_CCORR_NORMED)
    threshold = 0.95
    loc = np.where(res >= threshold)
    minx = 9999
    maxx = 0
    ans = (-100, -100)
    pts = zip(*loc[::-1])
    model = KMeans(n_clusters=8).fit(np.array([loc[0], loc[1]]).T)
    label = model.predict(np.array([loc[0], loc[1]]).T)
    vis = [0, 0, 0, 0, 0, 0, 0, 0]
    points = []
    for i, pt in enumerate(pts):
        # print(abs(pt[0] - ans[0]))
        if (vis[label[i]] == 0):
            vis[label[i]] = 1
            cv2.rectangle(origin_image_, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
            minx = min(minx, pt[0])
            maxx = max(maxx, pt[0]+w)
            points.append(pt)

    p = []
    points = sorted(points, key = lambda x : x[0] + x[1])
    p.append(points[0])
    p.append((points[-1][0] + 50, points[-1][1] + 50))
    points = sorted(points, key = lambda x : x[0] + (2000 - x[1]))
    p.append((points[0][0], points[0][1] +50))
    p.append((points[-1][0] + 50, points[-1][1] ))
    tmp = four_point_transform(origin_image_, np.array(p))

    points = sorted(points, key = lambda x : x[1])
    top = points[:3]
    mid = points[3:6]
    bot = points[6:]
    top = sorted(top)
    mid = sorted(mid)
    bot = sorted(bot)
    points = top + mid + bot

    delta = int(tmp.shape[0] * 0.021)
    answer_sheet_corners = []
    answer_sheet_corners.append((points[3][0] + delta, points[3][1] + delta))
    answer_sheet_corners.append((points[5][0] , points[5][1] + delta))
    answer_sheet_corners.append((points[6][0] + delta, points[6][1] ))
    answer_sheet_corners.append((points[7][0] , points[7][1] ))
    answer_sheet = four_point_transform(origin_image_, np.array(answer_sheet_corners))
    answer_sheet_ = answer_sheet.copy()

    answer_sheet__ = 255 - answer_sheet_.copy()

    beginy = int(answer_sheet__.shape[0]*0.032)
    beginx = int(answer_sheet__.shape[1]*0.067)
    # d = int(answer_sheet__.shape[0]*0.032)
    d = int(answer_sheet__.shape[0]/31.5)
    w = int(answer_sheet__.shape[1]*0.042)
    s = np.zeros((30, 17))
    for _ in range(30):
        currenty = beginy + d*(_)
        currentx = beginx
        for i in range(1, 17):
            cv2.rectangle(answer_sheet__, (currentx, currenty), (currentx + d, currenty + d), (0, 255, 0), 2)
            s[_, i] = sum(sum(answer_sheet__[currenty:currenty + d,currentx:currentx + d, 2] // 150))
            currentx += w
            if (i % 4 == 0):
                currentx += 2*w

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    answers = []
    x = 0
    y = 1
    DAPAN = ['A', 'B', 'C', 'D']
    for _ in range(120):
        tmp = []
        for j in range(y, y+4):
            tmp.append(s[x, j])
        answers.append(DAPAN[tmp.index(max(tmp))])
        x += 1
        if (x == 30):
            x = 0
            y += 4
    
    infor_sheet_corners = []
    infor_sheet_corners.append((points[1][0] + delta, points[1][1] + delta))
    infor_sheet_corners.append((points[2][0] , points[2][1] + delta))
    infor_sheet_corners.append((points[4][0] + delta, points[4][1] ))
    infor_sheet_corners.append((points[5][0] , points[5][1] ))
    infor_sheet = four_point_transform(origin_image_, np.array(infor_sheet_corners))

    infor_sheet_ = 255 - infor_sheet.copy()

    beginy = int(infor_sheet_.shape[0]*0.2125)
    beginx = 0
    d = int(infor_sheet_.shape[1]*0.095)
    w = int(infor_sheet_.shape[1]*0.0935)
    h = int(infor_sheet_.shape[0]*0.077)
    s_ = np.zeros((10, 10))
    cv2.rectangle(infor_sheet_, (beginx, beginy), (beginx + d, beginy + d), (0, 255, 0), 2)
    for _ in range(10):
        currenty = beginy + h*(_)
        currentx = beginx
        for i in range(1, 10):
            cv2.rectangle(infor_sheet_, (currentx, currenty), (currentx + d, currenty + d), (0, 255, 0), 2)
            s_[_, i] = int(sum(sum(infor_sheet_[currenty:currenty + d,currentx:currentx + d, 2] // 150)))
            currentx += w
            if (i % 6 == 0):
                currentx += int(1.7*w)

    sbd = ''
    made = ''
    for j in range(1, 7):
        num = []
        for i in range(10):
            num.append(s_[i, j])
        # print(num)
        sbd += str(num.index(max(num)))
    for j in range(7, 10):
        num = []
        for i in range(10):
            num.append(s_[i, j])
        made += str(num.index(max(num)))
    if (require_thresh):
        return sbd, made, answers, answer_sheet__
    else:
        return sbd, made, answers
