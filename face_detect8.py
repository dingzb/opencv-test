#!/usr/bin/env python
# coding=utf-8
"""
Face detect

Keys:
    ESC    - exit
"""

from __future__ import print_function
import cv2
import numpy as np
import math
import time
from common import clock


def detect(img, cascade, option=None):
    """detect base on cascade type"""
    if option is None:
        option = {
            "scaleFactor": 1.4,
            "minNeighbors": 4,
            "minSize": (30, 30),
            "flags": cv2.CASCADE_SCALE_IMAGE
        }
    # detects = cascade.detectMultiScale(img, scaleFactor=option["scaleFactor"], minNeighbors=option["minNeighbors"],
    #                                    minSize=option["minSize"], flags=option["flags"])
    detects = cascade.detectMultiScale(img, **option)
    if len(detects) == 0:
        return []
    detects[:, 2:] += detects[:, :2]
    return detects


def draw_rectangle(img, rectangle, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    x0, y0, x1, y1 = rectangle
    xd, yd = scale((x0, y0), (x1, y1), scaling)
    cv2.rectangle(img, xd, yd, color, 2)


def scale((x0, y0), (x1, y1), scaling=1.0, limit=None):
    """scaling. limit is a array [w, h]"""
    scaling = scaling - 1
    xd = int(abs(x1 - x0) * scaling / 2)
    yd = int(abs(y1 - y0) * scaling / 2)

    rx0 = x0 - xd
    ry0 = y0 - yd
    rx1 = x1 + xd
    ry1 = y1 + yd

    if limit is not None:
        if rx0 < 0:
            rx0 = 0
        if ry0 < 0:
            ry0 = 0
        if rx1 > limit[0]:
            rx1 = limit[0]
        if ry1 > limit[1]:
            ry1 = limit[1]

    return (rx0, ry0), (rx1, ry1)


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0)):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, front, lineType=cv2.LINE_AA)


def draw_rectangles(img, rectangles, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    for x0, y0, x1, y1 in rectangles:
        xd, yd = scale((x0, y0), (x1, y1), scaling)
        cv2.rectangle(img, xd, yd, color, 2)


def main():
    """
    选取八个区域进行规律查找
    :return:
    """
    print(__doc__)
    cap = cv2.VideoCapture(0)
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

    tracks = []
    frame_index = 0
    detect_interval = 5
    track_len = 10

    has_face = False

    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rectangles = detect(gray, f_cascade)

        mask = np.zeros_like(gray)  # 设置关键点遮罩

        if len(rectangles) == 1:  # 限制一张人脸
            if not (has_face and True):
                tracks = []
            has_face = True
            for rectangle in rectangles:
                rx0, ry0, rx1, ry1 = rectangle
                # if not (140 < rx1 - rx0 < 160 and 140 < ry1 - ry0 < 160):  # 限定人脸识别框的大小
                #     continue
                draw_rectangle(img, rectangle, color=(0, 225, 0))  # 人脸范围
                rectangles_eye = detect(gray[ry0:ry1, rx0:rx1], e_cascade)  # 获取眼睛范围
                draw_rectangles(img[ry0:ry1, rx0:rx1], rectangles_eye, color=(255, 0, 225))

                # 绘制关键点轨迹
                # 会删除位移较大的关键点
                # 不影响其他的鉴别
                if len(tracks) > 0:
                    img0, img1 = prev_gray, gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 0.5
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        if not (rx0 < x < rx1 and ry0 < y < ry1):
                            continue
                        tr.append((x, y))
                        if len(tr) > track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                    tracks = new_tracks
                    cv2.polylines(img, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                    draw_str(img, (20, 20), 'track count: %d' % len(tracks))

                # 限定人脸为兴趣区域
                cv2.fillPoly(mask, np.array([[[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]]), (255, 255, 255))
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)  # 排除上一次的关键点

                if frame_index % detect_interval == 0:
                    eye_tr = []  # 存放眼睛区域内的关键点
                    for tr in tracks:
                        (x0, y0) = tr[0]
                        (x1, y1) = tr[-1]

                        # 判断各个关键点(关键点轨迹中的最后一个)是否在眼睛区域范围内
                        for erx0, ery0, erx1, ery1 in rectangles_eye:
                            if erx0 + rx0 < x1 < erx1 + rx0 and ery0 + ry0 < y1 < ery1 + ry0:
                                eye_tr.append(tr)

                    # # 计算各个关键点的角度和位移，以便观察
                    #     l = round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2)
                    #     if l > 2:
                    #         print(round(math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180, 2), end=':')
                    #         print(l, end='\t')
                    # print('\n+++++++++++++++')

                    print(len(eye_tr))
                    for tr in eye_tr:
                        cv2.circle(img, tr[-1], 4, (255, 255, 0), -1)
                    p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

            frame_index += 1
        else:
            has_face = False

        prev_gray = gray
        dt = clock() - t
        # draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow("Face detect", img)
        cv2.imshow('mask', mask)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
