import numpy as np
import pandas as pd
import cv2
import math
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from PIL import Image

QUYDOI = ['A', 'B', 'C', 'D']
def get(warp, kind = 'answers', number_of_quests = 30, number_of_answers = 4):
    answers = []
    if (kind == 'answers'):
        sub_h = warp.shape[0] // number_of_quests
        # print(warp.shape)
        for i in range(number_of_quests):
            sub_image = warp[sub_h*i:sub_h*(i+1), 40:]
            sub_w = sub_image.shape[1] // number_of_answers
            choice = []
            # fig, ax = plt.subplots(1, 4, figsize=(5, 1))
            for _ in range(number_of_answers):
                point = sub_image[:, sub_w*_:sub_w*(_+1)] // 255
                # ax[_].imshow(point, cmap='gray')
                choice.append(sum(sum(point)))
            answers.append(QUYDOI[choice.index(max(choice))])
    elif (kind == 'ID'):
        sub_w = warp.shape[1] // number_of_answers
        # print(warp.shape)
        for i in range(number_of_answers):
            sub_image = warp[:, sub_w*i: sub_w * (i+1)]
            sub_h = sub_image.shape[0] // number_of_quests
            choice = []
            # fig, ax = plt.subplots(1, number_of_quests, figsize=(5, 1))
            for _ in range(number_of_quests):
                point = sub_image[sub_h*_: sub_h*(_+1), :] // 255
                # ax[_].imshow(point, cmap='gray')
                choice.append(sum(sum(point)))
            answers.append(choice.index(max(choice)))
    elif (kind == 'question_ID'):
        sub_w = warp.shape[1] // number_of_answers
        # print(warp.shape)
        for i in range(number_of_answers):
            sub_image = warp[:, sub_w*i: sub_w * (i+1)]
            sub_h = sub_image.shape[0] // number_of_quests
            choice = []
            # fig, ax = plt.subplots(1, number_of_quests, figsize=(5, 1))
            for _ in range(number_of_quests):
                point = sub_image[sub_h*_: sub_h*(_+1), :] // 255
                # ax[_].imshow(point, cmap='gray')
                choice.append(sum(sum(point)))
            answers.append(choice.index(max(choice)))
    else:
        raise Exception(f"There are no option {kind}!")
    return answers

def get_ID_infors(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    contours4 = contours[:4]
    contours4 = sorted(contours4, key=lambda x: x[0][0][0])
    wrap_ = []

    cnt = 0
    for _ in range(100):
        approx = cv2.approxPolyDP(contours[_], 0.01 * cv2.arcLength(contours[_], True), True)
        rect = cv2.minAreaRect(contours[_])
        box = cv2.boxPoints(rect)
        corner = find_corner_by_rotated_rect(box,approx)
        image_ = four_point_transform(thresh,corner)
        # print(_, image_.shape[1]/need_w)
        s = sum(sum(image_//255))
        # print(_, s, need_w)
        if (s != 0):
            wrap_.append(image_)
            cnt += 1
        if (cnt == 2):
            break
    ID = ''.join(list(map(str, get(wrap_[0], 'ID', 10, 6))))
    question_ID = ''.join(list(map(str, get(wrap_[1], 'question_ID', 10, 3))))
    return ID, question_ID

def get_informations(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (1200, 1600), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,3)
    blurred_ = cv2.GaussianBlur(thresh, (5, 5), 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    contours4 = contours[:4]
    contours4 = sorted(contours4, key=lambda x: x[0][0][0])
    wrap_ = []
    for _ in range(4):
        approx = cv2.approxPolyDP(contours4[_], 0.01 * cv2.arcLength(contours4[_], True), True)
        rect = cv2.minAreaRect(contours4[_])
        box = cv2.boxPoints(rect)
        corner = find_corner_by_rotated_rect(box,approx)
        image_ = four_point_transform(image,corner)
        wrap_.append(four_point_transform(thresh,corner))

    need_w = sum(sum(wrap_[0]))
    wrap_.append(0)
    wrap_.append(0)
    
    answers = get(wrap_[0])
    answers += get(wrap_[1])
    answers += get(wrap_[2])
    answers += get(wrap_[3])

    h, w = thresh.shape
    thresh = thresh[:2*(h//5),3*(w//5):]
    ID, question_ID = get_ID_infors(thresh)

    return ID, question_ID, answers