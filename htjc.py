#!/usr/bin/env python
# coding=utf-8

"""
活体检测

Keys:
    ESC    - exit
"""
from __future__ import division

import math
import sys
import cv2
from common import clock, draw_str
import numpy as np


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.9, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color=(0, 255, 0)):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    (q, t) = (0, len(lines))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        if math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2-y1),2)) > 5.8:
            cv2.circle(vis, (x2, y2), 1, (0, 0, 255), -1)  # TODO 在这里根据一定的算法判断点移动的情况
            q += 1
    if 0.5 > q/t > 0.19:
        cv2.circle(vis, (0, 0), 40, (25, 212, 255), -1)
    return vis


def main(video, fc, ec):
    print __doc__
    cap = cv2.VideoCapture(video)
    fCascade = cv2.CascadeClassifier(fc)
    eCascade = cv2.CascadeClassifier(ec)
    # for opt_flow
    ret, prev = cap.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        if cv2.waitKey(1) == 27:
            break
        t = clock()
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detect(gray, fCascade)
        for sx1, sy1, sx2, sy2 in rects:
            # for opt_flow
            flow = cv2.calcOpticalFlowFarneback(prevgray[sy1:sy2, sx1:sx2], gray[sy1:sy2, sx1:sx2], None, 0.5, 3,
                                                15, 3, 5, 1.2, 0)
            draw_flow(img[sy1:sy2, sx1:sx2], flow)
        draw_rects(img, rects, (0, 255, 0))
        if not eCascade.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = img[y1:y2, x1:x2]
                subrects = detect(roi.copy(), eCascade)
                draw_rects(vis_roi, subrects, (255, 0, 0))

        prevgray = gray
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow('HTJC', img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    v, f, e = (0, 'C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_frontalface_alt.xml',
               'C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml')
    if len(sys.argv) > 2:
        v = sys.argv[1]
    if len(sys.argv) > 3:
        h = sys.argv[2]
    main(v, f, e)
