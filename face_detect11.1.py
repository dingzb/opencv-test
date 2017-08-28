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
import threading
from common import clock

el_check_count = 30  # 判断为电子设备时 持续的帧数
share = {
    'color': [0] * el_check_count,
    'gray': [0] * el_check_count
}  # 摄像头和红外摄像头 最近 el_check_count 帧的人像采集情况，如果彩色摄像头的连续 el_check_count 帧都能识别到人脸，而红外摄像头一帧也没有采集到，则鉴定为电子屏


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


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0), size=1.0):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, size, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, size, front, lineType=cv2.LINE_AA)


def draw_rectangles(img, rectangles, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    for x0, y0, x1, y1 in rectangles:
        xd, yd = scale((x0, y0), (x1, y1), scaling)
        cv2.rectangle(img, xd, yd, color, 2)


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


def draw_flow(detected, flow, show=True, step=16, color=(0, 0, 255)):
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


def main(cam_id, c_type):
    """
    由于眼部检测的不稳定性，这里采用在已经识别出的人脸范围内假定的眼睛区域作为眼睛区域的判断
    采用 关键点检测 加 伪眼部光流检测
    关键点静止但眼部光流明显时判定为活体
    这个主要判断 眨眼 来做为活体检测的依据
    更换为双摄像头
    并判断
    1、电子设备（普通摄像头中有人脸区域而红外摄像头中没有）
    2、照片（眼睛区域关键点位移的方差与眼睛外部的方差 差值小于阈值时判定为照片，但有很大程度会把真人识别成照片
        修改为判断连续 n 帧的方差差值，如果全部符合再判定为照片）
    3、通过，活体（。。。）
    :param cam_id: 摄像头ID
    :param c_type: 类型（color:Visible Light 可见光, gray:infrared 红外）
    :return:
    """

    print(__doc__)
    cap = cv2.VideoCapture(cam_id)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('D:/data.avi', fourcc, 30, (640, 480))
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

    tracks = []
    frame_index = 0  # 总体的帧数， 用于判断电子设备
    face_frame_index = 0  # 脸部区域的帧数，用于 关键点轨迹
    detect_interval = 3
    track_len = 10
    msg_show_success = 0  # 包含光流和关键点判断通过的
    msg_show_opt = 0  # 通过信息显示帧数
    msg_show_key = 0  # 通过信息显示帧数
    msg_show_success_f = 0  # 没有通过包含光流和关键点判断通过的
    msg_show_opt_f = 0  # 没有通过通过信息显示帧数
    msg_show_key_f = 0  # 没有通过通过信息显示帧数
    has_face = False
    sustain = 10  # 信息持续时间

    ph_check_count = 30  # 判断为照片时 持续的帧数
    photo_rec = [0] * ph_check_count  # 记录每一帧的照片判定

    hu_check_count = 30 # 判断为真人时 持续的帧数
    human_rec = [0] * hu_check_count    # 记录为真人

    # 存储每一帧的光流
    eye_flow_lines_t = []

    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        if c_type == 'gray':
            img = img[24:456, 32:608]
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frame_index += 1
        rectangles = detect(gray, f_cascade)

        mask = np.zeros_like(gray)  # 设置关键点遮罩

        photo_rec[frame_index % ph_check_count] = 0
        human_rec[frame_index % hu_check_count] = 0

        if len(rectangles) == 1:  # 限制一张人脸
            share[c_type][frame_index % el_check_count] = 1  # 识别出人脸后将值设置为1
            if not (has_face and True):
                tracks = []
            has_face = True
            for rectangle in rectangles:
                rx0, ry0, rx1, ry1 = rectangle
                if rx1 - rx0 > 211:
                    draw_str(img, (20, 20), 'Close', front=(0, 0, 255))
                elif rx1 - rx0 < 211:
                    draw_str(img, (20, 20), 'Away', front=(0, 0, 255))
                else:
                    draw_str(img, (20, 20), 'OK. Hold.', front=(0, 255, 0))

                draw_rectangle(img, rectangle, color=(0, 225, 0))  # 人脸范围
                rectangles_eye = detect(gray[ry0:ry1, rx0:rx1], e_cascade)  # 获取眼睛范围
                # draw_rectangles(img[ry0:ry1, rx0:rx1], rectangles_eye, color=(255, 0, 225))

                # 眼部光流场功能
                eye_flow_lines = []
                # for erx0, ery0, erx1, ery1 in rectangles_eye:
                #     eye_flow = opt_flow(prev_gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1],
                #                         gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1])  # get opt flow
                #     eye_flow_lines.append(draw_flow(img[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1],
                #                                     eye_flow, step=4))  # 显示光流点

                # 假眼部位置，假设脸纵向上部1/4位置到2/4位置及 横向左部1/6到5/6位置为眼部，以抵消眼部识别不能每次都有效地问题

                face_h = ry1 - ry0
                face_w = rx1 - rx0
                face_hs = face_h / 4
                face_he = face_h / 2
                face_ws = face_w / 6
                face_we = face_w / 6 * 5
                eye_flow = opt_flow(prev_gray[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we],
                                    gray[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we])
                eye_flow_lines.append(
                    draw_flow(img[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we], eye_flow, step=4))

                eye_sorted = []  # 排序后的长度集合(眼睛)
                eye_sorted2 = []
                for lines in eye_flow_lines:
                    mds = []
                    for (x1, y1), (x2, y2) in lines:
                        md = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        mds.append(md)
                        eye_sorted2.append(md)
                    eye_sorted.append(sorted(mds, reverse=True))
                    eye_flow_lines_t.append(eye_sorted2)  # 存储每一帧的光流位移信息
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
                    # draw_str(img, (20, 20), 'track count: %d' % len(tracks))

                # 限定人脸为兴趣区域
                cv2.fillPoly(mask, np.array([[[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]]), (255, 255, 255))
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)  # 排除上一次的关键点

                if face_frame_index % detect_interval == 0:
                    # print('**************** start ***************')
                    l_sorted = []
                    l_sorted_eye = []  # 眼睛区域的关键点
                    l_sorted_out = []  # 眼睛外部的关键点

                    l_tmp = []
                    l_tmp_eye = []
                    l_tmp_out = []
                    for tr in tracks:
                        (x0, y0) = tr[0]
                        (x1, y1) = tr[-1]

                        if rx0 + face_ws < x1 < rx0 + face_we and ry0 + face_hs < y1 < ry1 + face_he:
                            l_tmp_eye.append(round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2))
                        else:
                            l_tmp_out.append(round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2))

                        l = round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2)
                        l_tmp.append(l)
                        # if l > 0:
                        # print(round(math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180, 2), end=':')
                        # print(l, end='\t')
                    # print('\n+++++++++++++++')

                    l_sorted = sorted(l_tmp, reverse=True)
                    l_sorted_eye = sorted(l_tmp_eye, reverse=True)
                    l_sorted_out = sorted(l_tmp_out, reverse=True)
                    if len(l_sorted_eye)>0:
                        print(l_sorted_eye[0])
                    if len(l_sorted_out)>0:
                        print(l_sorted_out[0])
                    print("--------------")
                    if len(l_sorted_out) > 3 and len(l_sorted_eye) > 3 \
                            and l_sorted_out[0] < 1 and l_sorted_eye[0] > 1 \
                            and l_sorted_eye[0] - l_sorted_out[0] > 1:
                        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                        msg_show_key = sustain
                        msg_show_success = sustain
                        # human_rec[frame_index % hu_check_count] = 1
                    elif (len(l_sorted_out) > 3 and len(l_sorted_eye) > 3 and abs(np.var(l_sorted_eye[:3]) - np.var(l_sorted_out[:3])) < 0.0005)\
                            or (len(l_sorted_eye) > 1 and len(l_sorted_out) > 1 and l_sorted_out[0] < 0.1 and l_sorted_eye[0] < 0.1):
                        print(np.var(l_sorted_eye[:3]) - np.var(l_sorted_out[:3]))
                        print("yesyesyesyesyesyes")
                        # 判定照片
                        # msg_show_key_f = sustain
                        # msg_show_success_f = sustain
                        photo_rec[frame_index % ph_check_count] = 1

                    # ========打印前十个=========================
                    # if True:
                    #     for i, md2 in enumerate(eye_sorted):
                    #         count = 0
                    #         print('眼睛', str(i + 1), end=':\t')
                    #         for md in md2:
                    #             count += 1
                    #             if count > 150:
                    #                 break
                    #             print(round(md, 2), end=',')
                    #         print()
                    #     print('###################')

                    # 活体检测
                    np_eye = np.array(sorted(eye_sorted2, reverse=True)[:30])
                    np_eye = np_eye[np_eye > 0]
                    np_l = np.array(l_sorted[:10])

                    # print(np_eye.size, '+++++', np_l.size)
                    if np_eye.size != 0 and np_l.size != 0:
                        flow_pre = np_eye[np_eye > 2].size * 1.0 / np_eye.size
                        ln_pre = np_l[np_l > 2].size * 1.0 / np_l.size
                        # print(flow_pre, '---', ln_pre)
                        if 0.8 > flow_pre > 0.05 and ln_pre < 0.2:
                            msg_show_opt = sustain
                            msg_show_success = sustain
                            # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                            # elif flow_pre < 0.05 and ln_pre < 0.2:
                            #     msg_show_opt_f = sustain
                            #     msg_show_success_f = sustain
                    # print('**************** end ***************')
                    # 判断关键点
                    p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

            face_frame_index += 1
        else:
            has_face = False
            share[c_type][frame_index % el_check_count] = 0  # 没识别出人脸（或识别出多个）后将值设置为0

        prev_gray = gray
        dt = clock() - t
        # draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        # if msg_show_key > 0:
        #     draw_str(img, (450, 20), 'YES by KEY', front=(0, 0, 255))
        #     msg_show_key -= 1
        # if msg_show_opt > 0:
        #     draw_str(img, (300, 20), 'YES by OPT', front=(0, 0, 255))
        # if c_type == 'color'> 0:
        #     print(sum(photo_rec))
        if sum(share['color']) > el_check_count * 0.99 and sum(
                share['gray']) == 0:  # color中80%的帧里面识别出人脸，gray中大于80%的帧中没有识别出人脸
            draw_str(img, (400, 30), 'Electronic', front=(0, 0, 255), size=2)
            msg_show_success = 0
            msg_show_success_f = 0
        elif sum(photo_rec) > ph_check_count * 0.1:
            draw_str(img, (400, 30), 'Photo', front=(0, 0, 255), size=2)
            msg_show_success = 0
        elif sum(share['color']) > el_check_count * 0.99 \
                and msg_show_success > 0: # and sum(human_rec) > 0:
            draw_str(img, (400, 30), 'Pass', front=(0, 255, 0), size=2)
        if msg_show_success > 0:
            msg_show_success -= 1
        # msg_show_success_f -= 1

        # if c_type == 'color':
        cv2.imshow(c_type + str(cam_id), img)
        out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 添加满足添加的点的数量的判断 如 满足条件的点数量必须大于10
    print('start capture.')
    video_srcs = [(0, 'gray'), (1, 'color')]
    tsk = []
    for video_src in video_srcs:
        t = threading.Thread(target=main, args=video_src)
        t.start()
        tsk.append(t)
    for t in tsk:
        t.join()
