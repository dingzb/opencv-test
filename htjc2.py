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
    rects = cascade.detectMultiScale(img, scaleFactor=1.4, minNeighbors=4, minSize=(30, 30),
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
    cv2.polylines(vis, lines, 0, (0, 0, 255))
    (q, t) = (0, len(lines))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    #     if math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)) > 5.8:
    #         cv2.circle(vis, (x2, y2), 1, (0, 0, 255), -1)  # TODO 在这里根据一定的算法判断点移动的情况
    #         q += 1
    # if 0.5 > q / t > 0.19:
    #     cv2.circle(vis, (0, 0), 40, (25, 212, 255), -1)

    if yzx(lines):
        cv2.circle(vis, (0, 0), 100, (105, 212, 255), -1)

    return vis


# 一致性判断
def yzx(vectors):
    # print(vectors.var())
    vs = []
    (lu, ru, rd, ld) = (0, 0, 0, 0)
    for (x0, y0), (x1, y1) in vectors:
        if x1 - x0 < 0 and y1 - y0 > 0:
            lu += 1
            vs.append(math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 > 0 and y1 - y0 > 0:
            ru += 1
            vs.append(180 - math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 > 0 and y1 - y0 < 0:
            rd += 1
            vs.append(180 + math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 < 0 and y1 - y0 < 0:
            ld += 1
            vs.append(360 - math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
    if len(vs) == 0:
        return False
    else:
        fc = var(vs)
        if 13000 > fc > 9000:
            print fc
        return 13000 > fc > 9000
    # print(vs)
    total = lu + ru + rd + ld
    # fz = 0.5  # 阈值

    # if lu > 0 or ru > 0 or rd > 0 or ld > 0:
    #     print(lu, ru, rd, ld)

    # if total != 0:
    #     return lu / total > fz or ru / total > fz or rd / total > fz or ld / total > fz
    # else:
    #     return False


def var(array):
    if len(array) == 0:
        return
    sum = 0
    for num in array:
        sum += num
    avg = sum / len(array)

    sum2 = 0

    for num in array:
        jj = num-avg
        if jj > 180:
            jj = 360 - jj
        sum2 += math.pow(jj,2)
    rs = sum2/len(array)
    # df = open('d:\\a.txt', 'w')
    # for a in array:
    #     df.write(str(a) + '\t')
    #     print(a)
    # df.close()
    return rs

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
        draw_rects(img, rects, (0, 255, 0))
        for sx1, sy1, sx2, sy2 in rects:
            # for opt_flow
            flow = cv2.calcOpticalFlowFarneback(prevgray[sy1:sy2, sx1:sx2], gray[sy1:sy2, sx1:sx2], None, 0.5, 3,
                                                15, 3, 5, 1.2, 0)
            draw_flow(img[sy1:sy2, sx1:sx2], flow)

        # if not eCascade.empty():
        #     for x1, y1, x2, y2 in rects:
        #         roi = gray[y1:y2, x1:x2]
        #         vis_roi = img[y1:y2, x1:x2]
        #         subrects = detect(roi.copy(), eCascade)
        #         draw_rects(vis_roi, subrects, (255, 0, 0))

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
