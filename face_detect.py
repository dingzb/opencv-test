#!/usr/bin/env python
# coding=utf-8
"""
Face detect

Keys:
    ESC    - exit
"""

import cv2
import numpy as np
import math
from common import clock


def scale((x0, y0), (x1, y1), scaling=1.0):
    """scaling"""
    scaling = scaling - 1
    xd = int(abs(x1 - x0) * scaling / 2)
    yd = int(abs(y1 - y0) * scaling / 2)
    return (x0 - xd, y0 - yd), (x1 + xd, y1 + yd)


def draw_rectangles(img, rectangles, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    for x0, y0, x1, y1 in rectangles:
        xd, yd = scale((x0, y0), (x1, y1), scaling)
        cv2.rectangle(img, xd, yd, color, 2)


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


def draw_flow(detected, flow, show=True, step=8, color=(0, 0, 255)):
    """draw as line with flow"""
    h, w = detected.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = detected
    if show:
        cv2.polylines(vis, lines, 0, color)  # draw lines
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # redraw the start point
    return lines


def opt_flow(p_gray, n_gray, option=None):
    def_option = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    if option is not None:
        def_option.update(option)
    flow = cv2.calcOpticalFlowFarneback(p_gray, n_gray, None, **def_option)
    return flow


def anti_spoofing(lines):
    """anti spoofing"""
    # return  True
    if offset_check(lines, 10, 0.015):
        return False
    vs = []
    (lu, ru, rd, ld) = (0, 0, 0, 0)
    for (x0, y0), (x1, y1) in lines:
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


def offset_check(vectors, max_offset, percent):
    """offset check. if the vector module great than max_offset return true"""
    count = 0
    for (x0, y0), (x1, y1) in vectors:
        length = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        if length > max_offset:
            count += 1
        print 1.0 * count / len(vectors)
    return 1.0 * count / len(vectors) > percent


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0)):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, front, lineType=cv2.LINE_AA)


def main():
    print __doc__
    cap = cv2.VideoCapture(0)
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        rectangles = detect(gray, f_cascade)
        draw_rectangles(img, rectangles, color=(255, 225, 0)) # 人脸范围
        # draw_rectangles(img, rectangles, scaling=1.3)   # 扩大后的范围
        for rx0, ry0, rx1, ry1 in rectangles:
            (x0, y0), (x1, y1) = scale((rx0, ry0), (rx1, ry1), 1.3)
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > w:
                x1 = w
            if y1 > h:
                y1 = h
            flow = opt_flow(prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1])  # get opt flow
            lines = draw_flow(img[y0:y1, x0:x1], flow, False)
            if anti_spoofing(lines):
                # draw_rectangles(img, [(rx0-20, ry0-30, rx0 + 55, ry0-5)])
                draw_str(img, (rx0-15, ry0-15), "Pass", (0, 69, 255), (255, 255, 255)) # print success
        prev_gray = gray
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow("Face detect", img)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
