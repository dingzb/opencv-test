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

    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        rectangles = detect(gray, f_cascade)
        # draw_rectangles(img, rectangles, color=(255, 225, 0))  # 人脸范围
        # draw_rectangles(img, rectangles, scaling=1.3)  # 扩大后的范围

        if len(tracks) > 0:
            img0, img1 = prev_gray, gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(img, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            draw_str(img, (20, 20), 'track count: %d' % len(tracks))

        mask = np.zeros_like(gray)

        if len(rectangles) == 1:  # 限制一张人脸
            for rectangle in rectangles:
                rx0, ry0, rx1, ry1 = rectangle
                if not (140 < rx1 - rx0 < 160 and 140 < ry1 - ry0 < 160):  # 限定人脸识别框的大小
                    continue
                draw_rectangle(img, rectangle, color=(0, 225, 0))  # 人脸范围

                if frame_index % detect_interval == 0:
                    cv2.fillPoly(mask, np.array([[[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]]), (255, 255, 255))   # 限定人脸为兴趣区域
                    p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])
                            # cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                            # cv2.circle(mask, (x, y), 5, 0, -1)          # 排除当前点

                frame_index += 1

        prev_gray = gray
        dt = clock() - t
        # draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow("Face detect", img)
        cv2.imshow('mask', mask)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
