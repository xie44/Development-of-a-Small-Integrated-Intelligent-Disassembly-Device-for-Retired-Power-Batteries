#!/usr/bin/env python3
"""
Edge Impulse 视觉引导机械臂拆螺丝 + 吸上盖
流程：检测四个螺丝 → 依次拆螺丝 → 吸上盖 → 退出
"""

import cv2
import json
import time
import sys
import numpy as np
import RPi.GPIO as GPIO
from edge_impulse_linux.image import ImageImpulseRunner
import TT

# ==================== 配置 ====================
MODEL_PATH = "/home/hk/.ei-linux-runner/models/886864/v4-runner-linux-armv7-impulse3/model.eim"
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ZOOM_FACTOR = 4.0
DISPLAY_WIDTH = 512
CONFIDENCE_THRESHOLD = 0.5
ROTATE_DISPLAY = True

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 9600

X_MIN_MM, X_MAX_MM = 0.0, 190.0
Y_MIN_MM, Y_MAX_MM = 0.0, 100.0

Z_SAFE = 40000          # 40mm
Z_GRASP = 0          # 0cm
Z_PLACE = 0          # 0cm
PLACE_X, PLACE_Y = 0, 0  # 螺丝放置点 (0,0mm)

# 电磁铁（吸盘）控制引脚
MAGNET_PIN = 21
MAGNET_HOLD_TIME = 1.0
INTER_TARGET_DELAY = 0.8

# 气泵PWM控制
PUMP_PIN = 14
PUMP_FREQ = 50
PUMP_DUTY_ON = 12.5      # 开启占空比
PUMP_DUTY_OFF = 0

# 步进电机控制（需根据实际硬件实现）
def stepper_rotate(angle):
    """
    步进电机旋转指定角度（度）
    angle为正顺时针，负逆时针
    """
    print(f"步进电机旋转 {angle} 度")
    # 这里放置实际的步进电机控制代码
    # 示例：使用GPIO控制
    # steps = int(abs(angle) * 200 / 360)   # 1.8°步距角
    # direction = GPIO.HIGH if angle > 0 else GPIO.LOW
    # ... 发送脉冲
    pass
# =================================================

# 标定数据
pixel_points = np.array([
    [533, 315], [640, 315], [746, 315],
    [533, 405], [640, 405], [746, 405]
], dtype=np.float32)
robot_points = np.array([
    [134985, 81990], [79995, 84000], [22005, 84000],
    [132990, 35985], [77010, 35985], [22020, 37020]
], dtype=np.float32)

H, _ = cv2.findHomography(pixel_points, robot_points)
if H is None:
    print("单应性矩阵计算失败")
    exit(1)

def pixel_to_robot(u, v):
    pts = np.array([[[u, v]]], dtype=np.float32)
    X_raw, Y_raw = cv2.perspectiveTransform(pts, H)[0, 0]
    X_mm = np.clip(X_raw/1000.0, X_MIN_MM, X_MAX_MM)
    Y_mm = np.clip(Y_raw/1000.0, Y_MIN_MM, Y_MAX_MM)
    return int(X_mm*1000), int(Y_mm*1000)

def ensure_robot_ready():
    TT.ALARMReset()
    TT.AxleEnabled(7, 1)
    time.sleep(0.3)

def move_xy(x, y):
    return TT.AxleMoveAbsolute(1,100,100,50,int(x)) and TT.AxleMoveAbsolute(2,100,100,50,int(y))

def move_z(z):
    return TT.AxleMoveAbsolute(4,100,100,50,int(z))

def refresh_display(sec, cap, runner, start_x, start_y, crop_w, crop_h):
    """在指定秒数内持续刷新实时视频窗口（不画框）"""
    steps = int(sec * 10)
    for _ in range(steps):
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                _, display_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h, draw_boxes=False)
                if display_img is not None:
                    if ROTATE_DISPLAY:
                        display_img = cv2.rotate(display_img, cv2.ROTATE_180)
                    cv2.imshow("Real-time Camera", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
        time.sleep(0.1)

def pick_and_place(tx, ty, cap, runner, start_x, start_y, crop_w, crop_h):
    """抓取并放置一个螺丝（电磁铁吸合/释放）"""
    if not move_z(Z_SAFE): return False
    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    if not move_xy(tx, ty): return False
    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    if not move_z(Z_GRASP): return False
    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)
    GPIO.output(MAGNET_PIN, GPIO.LOW)   # 吸合
    time.sleep(MAGNET_HOLD_TIME)
    cv2.waitKey(1)
    if not move_z(Z_SAFE): return False
    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    if not move_xy(PLACE_X, PLACE_Y): return False
    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    if not move_z(Z_PLACE): return False
    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)
    GPIO.output(MAGNET_PIN, GPIO.HIGH)  # 释放
    time.sleep(MAGNET_HOLD_TIME)
    cv2.waitKey(1)
    if not move_z(Z_SAFE): return False
    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    return True

def process_frame(frame, runner, start_x, start_y, crop_w, crop_h, draw_boxes=True):
    """
    返回 (centers, display_img)
    draw_boxes: 是否绘制检测框和中心点（实时视频为False，定格画面为True）
    """
    cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w].copy()
    if cropped.size == 0:
        return [], None

    resized = cv2.resize(cropped, (96,96))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    features,_ = runner.get_features_from_image(rgb)
    res = runner.classify(features)
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except:
            return [], None

    boxes = [b for b in res.get("result",{}).get("bounding_boxes",[]) if b["value"]>CONFIDENCE_THRESHOLD]

    display_scale = DISPLAY_WIDTH / crop_w
    display_img = cv2.resize(cropped, (DISPLAY_WIDTH, int(crop_h*display_scale)))

    scale_x = crop_w / 96.0
    scale_y = crop_h / 96.0

    centers = []
    for b in boxes:
        x,y,w,h = b["x"],b["y"],b["width"],b["height"]
        cx = (x + w/2)*scale_x + start_x
        cy = (y + h/2)*scale_y + start_y
        centers.append((cx,cy,b["label"],b["value"]))
        if draw_boxes:
            # 绘制边框
            x1,y1 = int(x*scale_x), int(y*scale_y)
            x2,y2 = int((x+w)*scale_x), int((y+h)*scale_y)
            dx1,dy1 = int(x1*display_scale), int(y1*display_scale)
            dx2,dy2 = int(x2*display_scale), int(y2*display_scale)
            cv2.rectangle(display_img, (dx1,dy1), (dx2,dy2), (0,255,0), 2)
            # 中心点
            cd_x = int(((x + w/2)*scale_x)*display_scale)
            cd_y = int(((y + h/2)*scale_y)*display_scale)
            cv2.circle(display_img, (cd_x,cd_y), 5, (0,0,255), -1)

    centers.sort(key=lambda x:x[3], reverse=True)
    return centers, display_img

def order_points(pts):
    """将四个点按左上、右上、右下、左下排序"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下
    return rect

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MAGNET_PIN, GPIO.OUT)
    GPIO.output(MAGNET_PIN, GPIO.HIGH)   # 初始释放

    # 气泵PWM初始化
    GPIO.setup(PUMP_PIN, GPIO.OUT)
    pump = GPIO.PWM(PUMP_PIN, PUMP_FREQ)
    pump.start(PUMP_DUTY_OFF)

    crop_w = int(FRAME_WIDTH / ZOOM_FACTOR)
    crop_h = int(FRAME_HEIGHT / ZOOM_FACTOR)
    start_x = (FRAME_WIDTH - crop_w)//2
    start_y = (FRAME_HEIGHT - crop_h)//2

    # 机械臂初始化
    TT.Link(SERIAL_PORT, BAUDRATE, timeout=5)
    time.sleep(0.5)
    if not TT.Test_Call():
        GPIO.cleanup()
        return
    ensure_robot_ready()
    TT.AxleToZero(7)
    time.sleep(5)
    move_xy(0,0)
    refresh_display(2, None, None, None, None, None, None)
    move_z(0)
    refresh_display(2, None, None, None, None, None, None)

    # 视觉初始化
    runner = ImageImpulseRunner(MODEL_PATH)
    runner.init()
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        runner.stop()
        TT.Downline(0)
        GPIO.cleanup()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow("Real-time Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Camera", DISPLAY_WIDTH, int(DISPLAY_WIDTH*crop_h/crop_w))
    print("实时视频窗口已打开，按 Ctrl+C 退出")
    print("等待检测到四个螺丝...")

    try:
        # 等待检测到四个螺丝
        frozen_centers = None
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            centers, display_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h, draw_boxes=False)
            if display_img is not None:
                if ROTATE_DISPLAY:
                    display_img = cv2.rotate(display_img, cv2.ROTATE_180)
                cv2.imshow("Real-time Camera", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt

            if len(centers) == 4:
                frozen_centers = centers.copy()
                # 生成定格画面（带框、数字、对角线）
                ret_f, frame_f = cap.read()
                if ret_f:
                    _, freeze_img = process_frame(frame_f, runner, start_x, start_y, crop_w, crop_h, draw_boxes=True)
                    if freeze_img is not None:
                        # 提取四个点的像素坐标
                        points = np.array([(u, v) for (u, v, _, _) in frozen_centers], dtype=np.float32)
                        # 按空间顺序排序
                        ordered = order_points(points)
                        # 计算中心点（对角线的交点，即四个点的均值）
                        center_u = np.mean(points[:,0])
                        center_v = np.mean(points[:,1])
                        # 绘制对角线
                        display_scale = DISPLAY_WIDTH / crop_w
                        disp_pts = []
                        for (u, v) in ordered:
                            crop_u = u - start_x
                            crop_v = v - start_y
                            dx = int(crop_u * display_scale)
                            dy = int(crop_v * display_scale)
                            disp_pts.append((dx, dy))
                        # 绘制对角线
                        cv2.line(freeze_img, disp_pts[0], disp_pts[2], (255,0,0), 2)
                        cv2.line(freeze_img, disp_pts[1], disp_pts[3], (255,0,0), 2)
                        # 绘制中心点
                        crop_cu = center_u - start_x
                        crop_cv = center_v - start_y
                        disp_cx = int(crop_cu * display_scale)
                        disp_cy = int(crop_cv * display_scale)
                        cv2.circle(freeze_img, (disp_cx, disp_cy), 8, (0,0,255), -1)
                        cv2.putText(freeze_img, "CENTER", (disp_cx+10, disp_cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        # 绘制数字
                        for idx, (dx, dy) in enumerate(disp_pts):
                            cv2.putText(freeze_img, str(idx+1), (dx+15, dy-15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                        if ROTATE_DISPLAY:
                            freeze_img = cv2.rotat#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单文件整合版：
- 《开环自动抓取》代码内容原样嵌入，不改内容
- 《上盖吸取》代码内容原样嵌入，不改内容
- 本文件只在外层做流程衔接：
  1) 先按开环自动抓取识别并拆螺丝
  2) 用初始冻结的四个螺丝点求上盖中心
  3) 再按你提供的《上盖吸取》参考代码吸取上盖
  4) 完成后回到初始位置 (0,0,0)
"""

import types
import time
import sys

OPEN_LOOP_SOURCE = '#!/usr/bin/env python3\n"""\nEdge Impulse 视觉引导机械臂拆螺丝（增强版：抓取中持续刷新视频）\n实时视频窗口始终显示最新画面，定格画面窗口短暂出现。\n连续30秒无目标自动退出。\n"""\n\nimport cv2\nimport json\nimport time\nimport sys\nimport numpy as np\nimport RPi.GPIO as GPIO\nfrom edge_impulse_linux.image import ImageImpulseRunner\nimport TT\n\n# ==================== 配置 ====================\nMODEL_PATH = "/home/hk/.ei-linux-runner/models/886864/v4-runner-linux-armv7-impulse3/model.eim"\nCAMERA_ID = 0\nFRAME_WIDTH = 1280\nFRAME_HEIGHT = 720\nZOOM_FACTOR = 4.0\nDISPLAY_WIDTH = 512\nCONFIDENCE_THRESHOLD = 0.5\nROTATE_DISPLAY = True\n\nSERIAL_PORT = "/dev/ttyUSB0"\nBAUDRATE = 9600\n\nX_MIN_MM, X_MAX_MM = 0.0, 190.0\nY_MIN_MM, Y_MAX_MM = 0.0, 100.0\n\nZ_SAFE = 40000          # 40mm\nZ_GRASP = 5000          # 0.5cm\nZ_PLACE = 5000          # 0.5cm\nPLACE_X, PLACE_Y = 0, 0  # (0,0mm)\n\nMAGNET_PIN = 21\nMAGNET_HOLD_TIME = 1.0\nINTER_TARGET_DELAY = 0.8\n\nDETECTION_TIMEOUT = 8 # 秒\n# =================================================\n\n# 标定数据（保持不变）\npixel_points = np.array([\n    [533, 315], [640, 315], [746, 315],\n    [533, 405], [640, 405], [746, 405]\n], dtype=np.float32)\nrobot_points = np.array([\n    [134985, 81990], [79995, 84000], [22005, 84000],\n    [132990, 35985], [77010, 35985], [22020, 37020]\n], dtype=np.float32)\n\nH, _ = cv2.findHomography(pixel_points, robot_points)\nif H is None:\n    print("单应性矩阵计算失败")\n    exit(1)\n\ndef pixel_to_robot(u, v):\n    pts = np.array([[[u, v]]], dtype=np.float32)\n    X_raw, Y_raw = cv2.perspectiveTransform(pts, H)[0, 0]\n    X_mm = np.clip(X_raw/1000.0, X_MIN_MM, X_MAX_MM)\n    Y_mm = np.clip(Y_raw/1000.0, Y_MIN_MM, Y_MAX_MM)\n    return int(X_mm*1000), int(Y_mm*1000)\n\ndef ensure_robot_ready():\n    TT.ALARMReset()\n    TT.AxleEnabled(7, 1)\n    time.sleep(0.3)\n\ndef move_xy(x, y):\n    return TT.AxleMoveAbsolute(1,100,100,50,int(x)) and TT.AxleMoveAbsolute(2,100,100,50,int(y))\n\ndef move_z(z):\n    return TT.AxleMoveAbsolute(4,100,100,50,int(z))\n\ndef refresh_display(sec, cap, runner, start_x, start_y, crop_w, crop_h):\n    """\n    在指定秒数内持续刷新实时视频窗口（如果cap不为None）\n    同时处理按键事件\n    """\n    steps = int(sec * 10)\n    for _ in range(steps):\n        if cap is not None:\n            ret, frame = cap.read()\n            if ret:\n                _, display_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h)\n                if display_img is not None:\n                    if ROTATE_DISPLAY:\n                        display_img = cv2.rotate(display_img, cv2.ROTATE_180)\n                    cv2.imshow("Real-time Camera", display_img)\n        if cv2.waitKey(1) & 0xFF == ord(\'q\'):\n            raise KeyboardInterrupt\n        time.sleep(0.1)\n\ndef pick_and_place(tx, ty, cap, runner, start_x, start_y, crop_w, crop_h):\n    """在每次移动间隙刷新视频"""\n    # 1.抬升Z\n    if not move_z(Z_SAFE): return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 2.移动XY\n    if not move_xy(tx, ty): return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 3.下降Z\n    if not move_z(Z_GRASP): return False\n    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 4.吸合\n    GPIO.output(MAGNET_PIN, GPIO.LOW)\n    time.sleep(MAGNET_HOLD_TIME)\n    cv2.waitKey(1)\n    # 5.抬升Z\n    if not move_z(Z_SAFE): return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 6.移动XY到放置点\n    if not move_xy(PLACE_X, PLACE_Y): return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 7.下降Z\n    if not move_z(Z_PLACE): return False\n    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)\n    # 8.释放\n    GPIO.output(MAGNET_PIN, GPIO.HIGH)\n    time.sleep(MAGNET_HOLD_TIME)\n    cv2.waitKey(1)\n    # 9.抬升Z\n    if not move_z(Z_SAFE): return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n    return True\n\ndef process_frame(frame, runner, start_x, start_y, crop_w, crop_h):\n    """返回 (centers, display_img)"""\n    cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w].copy()\n    if cropped.size == 0:\n        return [], None\n\n    resized = cv2.resize(cropped, (96,96))\n    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)\n    features,_ = runner.get_features_from_image(rgb)\n    res = runner.classify(features)\n    if isinstance(res, str):\n        try:\n            res = json.loads(res)\n        except:\n            return [], None\n\n    boxes = [b for b in res.get("result",{}).get("bounding_boxes",[]) if b["value"]>CONFIDENCE_THRESHOLD]\n\n    display_scale = DISPLAY_WIDTH / crop_w\n    display_img = cv2.resize(cropped, (DISPLAY_WIDTH, int(crop_h*display_scale)))\n\n    scale_x = crop_w / 96.0\n    scale_y = crop_h / 96.0\n\n    centers = []\n    for b in boxes:\n        x,y,w,h = b["x"],b["y"],b["width"],b["height"]\n        cx = (x + w/2)*scale_x + start_x\n        cy = (y + h/2)*scale_y + start_y\n        centers.append((cx,cy,b["label"],b["value"]))\n        # 绘制边框\n        x1,y1 = int(x*scale_x), int(y*scale_y)\n        x2,y2 = int((x+w)*scale_x), int((y+h)*scale_y)\n        dx1,dy1 = int(x1*display_scale), int(y1*display_scale)\n        dx2,dy2 = int(x2*display_scale), int(y2*display_scale)\n        cv2.rectangle(display_img, (dx1,dy1), (dx2,dy2), (0,255,0), 2)\n        # 中心点\n        cd_x = int(((x + w/2)*scale_x)*display_scale)\n        cd_y = int(((y + h/2)*scale_y)*display_scale)\n        cv2.circle(display_img, (cd_x,cd_y), 5, (0,0,255), -1)\n\n    centers.sort(key=lambda x:x[3], reverse=True)\n    return centers, display_img\n\ndef main():\n    GPIO.setmode(GPIO.BCM)\n    GPIO.setup(MAGNET_PIN, GPIO.OUT)\n    GPIO.output(MAGNET_PIN, GPIO.HIGH)\n\n    crop_w = int(FRAME_WIDTH / ZOOM_FACTOR)\n    crop_h = int(FRAME_HEIGHT / ZOOM_FACTOR)\n    start_x = (FRAME_WIDTH - crop_w)//2\n    start_y = (FRAME_HEIGHT - crop_h)//2\n\n    # 机械臂初始化\n    TT.Link(SERIAL_PORT, BAUDRATE, timeout=5)\n    time.sleep(0.5)\n    if not TT.Test_Call():\n        GPIO.cleanup()\n        return\n    ensure_robot_ready()\n    TT.AxleToZero(7)\n    time.sleep(5)\n    move_xy(0,0)\n    refresh_display(2, None, None, None, None, None, None)  # 无摄像头，仅延时\n    move_z(0)\n    refresh_display(2, None, None, None, None, None, None)\n\n    # 视觉初始化\n    runner = ImageImpulseRunner(MODEL_PATH)\n    runner.init()\n    cap = cv2.VideoCapture(CAMERA_ID)\n    if not cap.isOpened():\n        runner.stop()\n        TT.Downline(0)\n        GPIO.cleanup()\n        return\n    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)\n    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)\n\n    cv2.namedWindow("Real-time Camera", cv2.WINDOW_NORMAL)\n    cv2.resizeWindow("Real-time Camera", DISPLAY_WIDTH, int(DISPLAY_WIDTH*crop_h/crop_w))\n    print("实时视频窗口已打开，按 Ctrl+C 退出")\n    print(f"若连续 {DETECTION_TIMEOUT} 秒未检测到目标，程序将自动退出")\n\n    try:\n        while True:\n            # 实时检测循环\n            frozen_centers = None\n            freeze_img = None\n            detection_start = time.time()\n\n            while True:\n                ret, frame = cap.read()\n                if not ret:\n                    continue\n                centers, display_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h)\n                if display_img is not None:\n                    if ROTATE_DISPLAY:\n                        display_img = cv2.rotate(display_img, cv2.ROTATE_180)\n                    cv2.imshow("Real-time Camera", display_img)\n                key = cv2.waitKey(1) & 0xFF\n                if key == ord(\'q\'):\n                    raise KeyboardInterrupt\n\n                if centers:\n                    frozen_centers = centers.copy()\n                    # 生成定格画面\n                    ret_f, frame_f = cap.read()\n                    if ret_f:\n                        _, freeze_img = process_frame(frame_f, runner, start_x, start_y, crop_w, crop_h)\n                        if freeze_img is not None:\n                            # 绘制数字\n                            display_scale = DISPLAY_WIDTH / crop_w\n                            disp_coords = []\n                            for (u, v, _, _) in frozen_centers:\n                                crop_u = u - start_x\n                                crop_v = v - start_y\n                                dx = int(crop_u * display_scale)\n                                dy = int(crop_v * display_scale)\n                                disp_coords.append((dx, dy))\n                            if ROTATE_DISPLAY:\n                                freeze_img = cv2.rotate(freeze_img, cv2.ROTATE_180)\n                                for idx, (dx, dy) in enumerate(disp_coords):\n                                    rx = freeze_img.shape[1] - 1 - dx\n                                    ry = freeze_img.shape[0] - 1 - dy\n                                    cv2.putText(freeze_img, str(idx+1), (rx+15, ry-15),\n                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)\n                            else:\n                                for idx, (dx, dy) in enumerate(disp_coords):\n                                    cv2.putText(freeze_img, str(idx+1), (dx+15, dy-15),\n                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)\n                            cv2.imshow("Frozen Frame", freeze_img)\n                            print(f"检测到 {len(frozen_centers)} 个目标，定格画面显示3秒...")\n                            # 等待3秒，同时刷新实时视频\n                            wait_end = time.time() + 3\n                            while time.time() < wait_end:\n                                ret_live, frame_live = cap.read()\n                                if ret_live:\n                                    _, live_disp = process_frame(frame_live, runner, start_x, start_y, crop_w, crop_h)\n                                    if live_disp is not None:\n                                        if ROTATE_DISPLAY:\n                                            live_disp = cv2.rotate(live_disp, cv2.ROTATE_180)\n                                        cv2.imshow("Real-time Camera", live_disp)\n                                if cv2.waitKey(50) & 0xFF == ord(\'q\'):\n                                    raise KeyboardInterrupt\n                            cv2.destroyWindow("Frozen Frame")\n                    break  # 跳出检测循环，开始抓取\n\n                # 超时检查\n                if time.time() - detection_start > DETECTION_TIMEOUT:\n                    print(f"\\n超时：连续 {DETECTION_TIMEOUT} 秒未检测到目标，程序退出")\n                    cap.release()\n                    cv2.destroyAllWindows()\n                    runner.stop()\n                    TT.Downline(0)\n                    GPIO.cleanup()\n                    sys.exit(0)\n\n            # 依次处理每个目标\n            for idx, (u, v, lab, conf) in enumerate(frozen_centers):\n                print(f"\\n--- 处理目标 {idx+1} ---")\n                tx, ty = pixel_to_robot(u, v)\n                if pick_and_place(tx, ty, cap, runner, start_x, start_y, crop_w, crop_h):\n                    print(f"  目标 {idx+1} 完成")\n                else:\n                    print(f"  目标 {idx+1} 失败，跳过")\n                    ensure_robot_ready()\n                if idx < len(frozen_centers) - 1:\n                    refresh_display(INTER_TARGET_DELAY, cap, runner, start_x, start_y, crop_w, crop_h)\n\n            # 归位\n            print("\\n所有目标处理完成，归位")\n            move_z(Z_SAFE)\n            refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n            move_xy(0, 0)\n            refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n            move_z(0)\n            refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n            print("归位完成，等待5秒后继续检测...")\n            refresh_display(5, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    except KeyboardInterrupt:\n        print("\\n用户中断")\n\n    cap.release()\n    cv2.destroyAllWindows()\n    runner.stop()\n    TT.Downline(0)\n    GPIO.cleanup()\n\nif __name__ == "__main__":\n    main()'
LID_PICK_SOURCE = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""\nEdge Impulse 视觉引导机械臂：单独吸上盖程序（修正版）\n\n本版按最新要求修改：\n1. 吸盘坐标补偿改为：X +1cm，Y -5.5cm。\n2. 电机第一转：顺时针 210°；第二转：逆时针 210°。\n3. 定格窗口不再旋转，保证画面端正。\n4. 吸上盖顺序调整为：\n   - 到吸盘中心\n   - 电机顺时针 210°\n   - 电磁阀置于“放/泄压”状态\n   - 开气泵\n   - Z 轴下降到吸取位\n   - 到位后切换到“吸附”状态（代码中对应 valve_close）\n   - 停顿 1.5s\n   - Z 轴回到安全位\n   - XY 到 (0,0)\n   - 电磁阀打开释放\n   - 关闭气泵\n   - 电机逆时针 210°\n   - Z 轴回原点\n5. 增加摄像头/串口异常保护，避免直接爆栈退出。\n"""\n\nimport cv2\nimport json\nimport time\nimport sys\nimport math\nimport numpy as np\nimport RPi.GPIO as GPIO\nfrom edge_impulse_linux.image import ImageImpulseRunner\nimport TT\n\n# ==================== 配置 ====================\nMODEL_PATH = "/home/hk/.ei-linux-runner/models/886864/v4-runner-linux-armv7-impulse3/model.eim"\nCAMERA_ID = 0\nFRAME_WIDTH = 1280\nFRAME_HEIGHT = 720\nZOOM_FACTOR = 4.0\nDISPLAY_WIDTH = 512\nCONFIDENCE_THRESHOLD = 0.5\n\nROTATE_LIVE_DISPLAY = True\nROTATE_FROZEN_DISPLAY = False\n\nSERIAL_PORT = "/dev/ttyUSB0"\nBAUDRATE = 9600\n\nX_MIN_MM, X_MAX_MM = 0.0, 190.0\nY_MIN_MM, Y_MAX_MM = 0.0, 100.0\n\nZ_SAFE = 40000\nZ_LID_PICK = 5000\nZ_PLACE = 5000\nPLACE_X, PLACE_Y = 0, 0\nPUMP_START_SETTLE = 0.8\nRELEASE_SETTLE = 0.4\n\n# 最新偏置：X +1cm, Y -5.5cm\nSUCTION_OFFSET_X = 10000\nSUCTION_OFFSET_Y = -55000\n\n# 气泵 / 电磁阀\nPUMP_PIN = 14\nVALVE_PIN = 15\nPUMP_FREQ = 50\nVALVE_FREQ = 50\nPUMP_DUTY = 12.5\nVALVE_CLOSE_DUTY = 0.0      # 吸附状态\nVALVE_OPEN_DUTY = 12.5      # 释放状态\n\n# 步进电机（Waveshare Stepper Motor HAT, M1接口）\nSTEP_PIN = 19\nDIR_PIN = 13\nENABLE_PIN = 12\nSTEPS_PER_REV = 200\nSTEP_DELAY = 0.002\nMOTOR_STEP_HIGH_TIME = 0.0001\nROTATE_ANGLE_DEG = 210.0\nROTATE_STEPS = int(round(STEPS_PER_REV * ROTATE_ANGLE_DEG / 360.0))\nDIR_CCW_LEVEL = GPIO.HIGH\nDIR_CW_LEVEL = GPIO.LOW\n\n# 如果模型里有专门的“上盖四点”标签，可在这里填写，例如 {"point"}\nLID_POINT_LABELS = set()\n# ==============================================\n\npixel_points = np.array([\n    [533, 315], [640, 315], [746, 315],\n    [533, 405], [640, 405], [746, 405]\n], dtype=np.float32)\nrobot_points = np.array([\n    [134985, 81990], [79995, 84000], [22005, 84000],\n    [132990, 35985], [77010, 35985], [22020, 37020]\n], dtype=np.float32)\n\nH, _ = cv2.findHomography(pixel_points, robot_points)\nif H is None:\n    print("单应性矩阵计算失败")\n    sys.exit(1)\n\npump = None\nvalve = None\ncamera_warned = False\nrobot_error = False\n\n\ndef pixel_to_robot(u, v):\n    pts = np.array([[[u, v]]], dtype=np.float32)\n    x_raw, y_raw = cv2.perspectiveTransform(pts, H)[0, 0]\n    x_mm = np.clip(x_raw / 1000.0, X_MIN_MM, X_MAX_MM)\n    y_mm = np.clip(y_raw / 1000.0, Y_MIN_MM, Y_MAX_MM)\n    return int(x_mm * 1000), int(y_mm * 1000)\n\n\ndef suction_pixel_to_robot(u, v):\n    tx, ty = pixel_to_robot(u, v)\n    tx += SUCTION_OFFSET_X\n    ty += SUCTION_OFFSET_Y\n    tx = int(np.clip(tx / 1000.0, X_MIN_MM, X_MAX_MM) * 1000)\n    ty = int(np.clip(ty / 1000.0, Y_MIN_MM, Y_MAX_MM) * 1000)\n    return tx, ty\n\n\ndef tt_safe_call(func, *args, action_name="TT动作", none_is_success=False, **kwargs):\n    global robot_error\n    try:\n        result = func(*args, **kwargs)\n        if result is None and none_is_success:\n            return True\n        return result\n    except Exception as e:\n        robot_error = True\n        print(f"{action_name}失败: {e}")\n        return False\n\n\ndef ensure_robot_ready():\n    ok1 = tt_safe_call(TT.ALARMReset, action_name="报警复位")\n    ok2 = tt_safe_call(TT.AxleEnabled, 7, 1, action_name="轴使能")\n    time.sleep(0.3)\n    return bool(ok1 and ok2)\n\n\ndef move_xy(x, y):\n    ok1 = tt_safe_call(TT.AxleMoveAbsolute, 1, 100, 100, 50, int(x), action_name=f"X轴移动到 {int(x)}")\n    ok2 = tt_safe_call(TT.AxleMoveAbsolute, 2, 100, 100, 50, int(y), action_name=f"Y轴移动到 {int(y)}")\n    return bool(ok1 and ok2)\n\n\ndef move_z(z):\n    # Z轴按测试代码使用更保守的速度，减少卡顿概率\n    return bool(tt_safe_call(TT.AxleMoveAbsolute, 4, 30, 30, 30, int(z), action_name=f"Z轴移动到 {int(z)}"))\n\n\ndef safe_cap_read(cap):\n    global camera_warned\n    if cap is None:\n        return False, None\n    try:\n        if hasattr(cap, "isOpened") and not cap.isOpened():\n            if not camera_warned:\n                print("[警告] 摄像头当前不可用，跳过实时刷新")\n                camera_warned = True\n            return False, None\n        ret, frame = cap.read()\n        if not ret:\n            return False, None\n        return True, frame\n    except Exception as e:\n        if not camera_warned:\n            print(f"[警告] 摄像头读取失败，跳过实时刷新: {e}")\n            camera_warned = True\n        return False, None\n\n\ndef setup_gpio():\n    global pump, valve\n    GPIO.setwarnings(False)\n    GPIO.setmode(GPIO.BCM)\n\n    GPIO.setup(PUMP_PIN, GPIO.OUT)\n    pump = GPIO.PWM(PUMP_PIN, PUMP_FREQ)\n    pump.start(0)\n\n    GPIO.setup(VALVE_PIN, GPIO.OUT)\n    valve = GPIO.PWM(VALVE_PIN, VALVE_FREQ)\n    valve.start(VALVE_CLOSE_DUTY)   # 默认关闭，避免持续嗡响和额外负载\n\n    GPIO.setup(STEP_PIN, GPIO.OUT)\n    GPIO.setup(DIR_PIN, GPIO.OUT)\n    GPIO.setup(ENABLE_PIN, GPIO.OUT)\n    GPIO.output(ENABLE_PIN, GPIO.HIGH)\n    GPIO.output(STEP_PIN, GPIO.LOW)\n    GPIO.output(DIR_PIN, GPIO.LOW)\n\n\ndef pump_on():\n    if pump is not None:\n        pump.ChangeDutyCycle(PUMP_DUTY)\n        print(f"气泵开启（占空比 {PUMP_DUTY}%）")\n\n\ndef pump_off():\n    if pump is not None:\n        pump.ChangeDutyCycle(0)\n        print("气泵关闭")\n\n\ndef valve_close():\n    if valve is not None:\n        valve.ChangeDutyCycle(VALVE_CLOSE_DUTY)\n        print("电磁阀关闭（吸）")\n\n\ndef valve_open():\n    if valve is not None:\n        valve.ChangeDutyCycle(VALVE_OPEN_DUTY)\n        print("电磁阀打开（放）")\n\n\ndef motor_enable():\n    GPIO.output(ENABLE_PIN, GPIO.LOW)\n\n\ndef motor_disable():\n    GPIO.output(ENABLE_PIN, GPIO.HIGH)\n\n\ndef motor_set_direction_cw():\n    GPIO.output(DIR_PIN, DIR_CW_LEVEL)\n\n\ndef motor_set_direction_ccw():\n    GPIO.output(DIR_PIN, DIR_CCW_LEVEL)\n\n\ndef motor_step_once():\n    GPIO.output(STEP_PIN, GPIO.HIGH)\n    time.sleep(MOTOR_STEP_HIGH_TIME)\n    GPIO.output(STEP_PIN, GPIO.LOW)\n\n\ndef rotate_motor_steps(steps, direction_name):\n    try:\n        motor_enable()\n        print(f"步进电机{direction_name} {ROTATE_ANGLE_DEG:.0f}°，步数: {steps}")\n        for _ in range(steps):\n            motor_step_once()\n            time.sleep(STEP_DELAY)\n        motor_disable()\n        return True\n    except Exception as e:\n        print(f"步进电机旋转失败: {e}")\n        try:\n            motor_disable()\n        except Exception:\n            pass\n        return False\n\n\ndef rotate_motor_cw_210():\n    motor_set_direction_cw()\n    return rotate_motor_steps(ROTATE_STEPS, "顺时针")\n\n\ndef rotate_motor_ccw_210():\n    motor_set_direction_ccw()\n    return rotate_motor_steps(ROTATE_STEPS, "逆时针")\n\n\ndef cleanup_gpio():\n    global pump, valve\n    try:\n        if pump is not None:\n            pump.stop()\n    except Exception:\n        pass\n    try:\n        if valve is not None:\n            valve.stop()\n    except Exception:\n        pass\n    try:\n        motor_disable()\n    except Exception:\n        pass\n    try:\n        GPIO.cleanup()\n    except Exception:\n        pass\n\n\ndef crop_frame(frame, start_x, start_y, crop_w, crop_h):\n    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()\n    return cropped if cropped.size != 0 else None\n\n\ndef classify_cropped(cropped, runner):\n    resized = cv2.resize(cropped, (96, 96))\n    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)\n    features, _ = runner.get_features_from_image(rgb)\n    res = runner.classify(features)\n    if isinstance(res, str):\n        try:\n            res = json.loads(res)\n        except Exception:\n            return []\n    return res.get("result", {}).get("bounding_boxes", [])\n\n\ndef process_frame(frame, runner, start_x, start_y, crop_w, crop_h, allowed_labels=None):\n    """\n    返回：\n    centers: [(cx, cy, label, conf, x1, y1, x2, y2), ...]\n    live_img: 实时窗口显示图（不画框）\n    """\n    cropped = crop_frame(frame, start_x, start_y, crop_w, crop_h)\n    if cropped is None:\n        return [], None\n\n    boxes = classify_cropped(cropped, runner)\n\n    scale_x = crop_w / 96.0\n    scale_y = crop_h / 96.0\n\n    centers = []\n    for b in boxes:\n        conf = b.get("value", 0)\n        label = b.get("label", "target")\n        if conf <= CONFIDENCE_THRESHOLD:\n            continue\n        if allowed_labels and label not in allowed_labels:\n            continue\n\n        x, y, w, h = b["x"], b["y"], b["width"], b["height"]\n        cx = (x + w / 2.0) * scale_x + start_x\n        cy = (y + h / 2.0) * scale_y + start_y\n        x1 = int(x * scale_x)\n        y1 = int(y * scale_y)\n        x2 = int((x + w) * scale_x)\n        y2 = int((y + h) * scale_y)\n        centers.append((cx, cy, label, conf, x1, y1, x2, y2))\n\n    centers.sort(key=lambda item: item[3], reverse=True)\n\n    display_scale = DISPLAY_WIDTH / crop_w\n    live_img = cv2.resize(cropped, (DISPLAY_WIDTH, int(crop_h * display_scale)))\n    return centers, live_img\n\n\ndef show_live_frame(frame, runner, start_x, start_y, crop_w, crop_h):\n    _, live_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h)\n    if live_img is not None:\n        if ROTATE_LIVE_DISPLAY:\n            live_img = cv2.rotate(live_img, cv2.ROTATE_180)\n        cv2.imshow("Real-time Camera", live_img)\n\n\ndef refresh_display(sec, cap, runner, start_x, start_y, crop_w, crop_h):\n    steps = max(1, int(sec * 10))\n    for _ in range(steps):\n        ret, frame = safe_cap_read(cap)\n        if ret:\n            show_live_frame(frame, runner, start_x, start_y, crop_w, crop_h)\n        if cv2.waitKey(1) & 0xFF == ord(\'q\'):\n            raise KeyboardInterrupt\n        time.sleep(0.1)\n\n\ndef draw_number(img, pt, text, color=(255, 255, 0)):\n    x, y = int(pt[0]), int(pt[1])\n    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)\n    cv2.putText(img, str(text), (x + 12, y - 12),\n                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)\n\n\ndef order_points_clockwise(points):\n    pts = np.array(points, dtype=np.float32)\n    center = np.mean(pts, axis=0)\n\n    def angle(p):\n        return math.atan2(p[1] - center[1], p[0] - center[0])\n\n    sorted_pts = sorted(pts.tolist(), key=angle)\n    sorted_pts = np.array(sorted_pts, dtype=np.float32)\n    sums = sorted_pts[:, 0] + sorted_pts[:, 1]\n    start_idx = int(np.argmin(sums))\n    sorted_pts = np.roll(sorted_pts, -start_idx, axis=0)\n    return sorted_pts\n\n\ndef line_intersection(p1, p2, p3, p4):\n    x1, y1 = p1\n    x2, y2 = p2\n    x3, y3 = p3\n    x4, y4 = p4\n\n    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)\n    if abs(denom) < 1e-6:\n        return np.mean(np.array([p1, p2, p3, p4], dtype=np.float32), axis=0)\n\n    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom\n    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom\n    return np.array([px, py], dtype=np.float32)\n\n\ndef make_lid_frozen_view(frame, quad_points, center_uv, start_x, start_y, crop_w, crop_h):\n    cropped = crop_frame(frame, start_x, start_y, crop_w, crop_h)\n    if cropped is None:\n        return None\n\n    display_scale = DISPLAY_WIDTH / crop_w\n    freeze_img = cv2.resize(cropped, (DISPLAY_WIDTH, int(crop_h * display_scale)))\n\n    disp_pts = []\n    for u, v in quad_points:\n        crop_u = u - start_x\n        crop_v = v - start_y\n        dx = int(crop_u * display_scale)\n        dy = int(crop_v * display_scale)\n        disp_pts.append((dx, dy))\n\n    center_crop_u = center_uv[0] - start_x\n    center_crop_v = center_uv[1] - start_y\n    center_disp = (int(center_crop_u * display_scale), int(center_crop_v * display_scale))\n\n    for i in range(4):\n        cv2.line(freeze_img, disp_pts[i], disp_pts[(i + 1) % 4], (0, 255, 0), 2)\n\n    cv2.line(freeze_img, disp_pts[0], disp_pts[2], (255, 0, 0), 2)\n    cv2.line(freeze_img, disp_pts[1], disp_pts[3], (255, 0, 0), 2)\n\n    for idx, pt in enumerate(disp_pts, start=1):\n        draw_number(freeze_img, pt, idx)\n\n    cv2.circle(freeze_img, center_disp, 8, (0, 255, 255), -1)\n    cv2.putText(freeze_img, "CENTER", (center_disp[0] + 12, center_disp[1] + 20),\n                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)\n\n    if ROTATE_FROZEN_DISPLAY:\n        freeze_img = cv2.rotate(freeze_img, cv2.ROTATE_180)\n    return freeze_img\n\n\ndef capture_lid_center(cap, runner, start_x, start_y, crop_w, crop_h):\n    print("开始检测上盖四个特征点...")\n    while True:\n        ret, frame = safe_cap_read(cap)\n        if not ret:\n            if cv2.waitKey(1) & 0xFF == ord(\'q\'):\n                raise KeyboardInterrupt\n            time.sleep(0.05)\n            continue\n\n        centers, live_img = process_frame(\n            frame, runner, start_x, start_y, crop_w, crop_h,\n            allowed_labels=(LID_POINT_LABELS if LID_POINT_LABELS else None)\n        )\n\n        if live_img is not None:\n            if ROTATE_LIVE_DISPLAY:\n                live_img = cv2.rotate(live_img, cv2.ROTATE_180)\n            cv2.imshow("Real-time Camera", live_img)\n\n        key = cv2.waitKey(1) & 0xFF\n        if key == ord(\'q\'):\n            raise KeyboardInterrupt\n\n        if len(centers) >= 4:\n            top4 = centers[:4]\n            pts = [(item[0], item[1]) for item in top4]\n            quad = order_points_clockwise(pts)\n            center_uv = line_intersection(quad[0], quad[2], quad[1], quad[3])\n\n            ret_f, frame_f = safe_cap_read(cap)\n            if ret_f:\n                freeze_img = make_lid_frozen_view(\n                    frame_f, quad, center_uv,\n                    start_x, start_y, crop_w, crop_h\n                )\n                if freeze_img is not None:\n                    cv2.imshow("Frozen Frame", freeze_img)\n                    print("已识别到 4 个点，Frozen Frame 显示编号、连线、对角线和交点")\n\n                    wait_end = time.time() + 3\n                    while time.time() < wait_end:\n                        ret_live, frame_live = safe_cap_read(cap)\n                        if ret_live:\n                            show_live_frame(frame_live, runner, start_x, start_y, crop_w, crop_h)\n                        if cv2.waitKey(50) & 0xFF == ord(\'q\'):\n                            raise KeyboardInterrupt\n\n            return center_uv, quad\n\n\ndef stop_suction_safely():\n    try:\n        pump_off()\n    except Exception:\n        pass\n    try:\n        valve_close()\n    except Exception:\n        pass\n\n\ndef pickup_lid_and_finish(suction_x, suction_y, cap, runner, start_x, start_y, crop_w, crop_h):\n    print("\\n开始吸上盖流程（抗干扰顺序）")\n\n    # 1. Z 升到 4cm 安全高度\n    if not move_z(Z_SAFE):\n        stop_suction_safely()\n        return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 2. XY 移动到吸盘中心点\n    print(f"吸盘中心机械坐标: X={suction_x}, Y={suction_y}")\n    if not move_xy(suction_x, suction_y):\n        stop_suction_safely()\n        return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 3. 电机顺时针旋转 210°\n    if not rotate_motor_cw_210():\n        stop_suction_safely()\n        return False\n    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 4. 先开阀释放，但先不启动气泵，避免“开泵+下Z”同时发生\n    valve_open()\n    refresh_display(RELEASE_SETTLE, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 5. 先把 Z 下到吸取位，整个下放过程不带气泵负载\n    if not move_z(Z_LID_PICK):\n        stop_suction_safely()\n        return False\n    refresh_display(0.8, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 6. 到位后再开气泵，并静止等待电源稳定\n    pump_on()\n    refresh_display(PUMP_START_SETTLE, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 7. 切换到吸附状态并保持 1.5s\n    valve_close()\n    refresh_display(1.5, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 8. Z 回升到 4cm\n    if not move_z(Z_SAFE):\n        stop_suction_safely()\n        return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 9. XY 移动到指定放置位置\n    if not move_xy(PLACE_X, PLACE_Y):\n        stop_suction_safely()\n        return False\n    refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 10. 下放到放置高度；必须等到达指定位置后再松开\n    if not move_z(Z_PLACE):\n        stop_suction_safely()\n        return False\n    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 11. 到位后再释放上盖/电池盒\n    valve_open()\n    refresh_display(RELEASE_SETTLE, cap, runner, start_x, start_y, crop_w, crop_h)\n    pump_off()\n    refresh_display(0.3, cap, runner, start_x, start_y, crop_w, crop_h)\n    valve_close()\n    refresh_display(0.2, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 12. 用户要求：释放完成后不要再先抬到安全位，保持当前高度\n    refresh_display(0.5, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 13. 电机逆时针旋转 210°\n    if not rotate_motor_ccw_210():\n        return False\n    refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)\n\n    # 14. 不在这里回原点；由外层统一回到 (0,0,0)\n    return True\n\ndef main():\n    cap = None\n    runner = None\n    try:\n        setup_gpio()\n\n        if not tt_safe_call(TT.Link, SERIAL_PORT, BAUDRATE, None, action_name="串口连接", none_is_success=True):\n            print("TT 串口连接失败")\n            return\n        time.sleep(0.5)\n\n        if not tt_safe_call(TT.Test_Call, action_name="TT 通讯测试"):\n            print("TT 通讯失败")\n            return\n\n        if not ensure_robot_ready():\n            return\n\n        tt_safe_call(TT.AxleToZero, 7, action_name="机械臂回零")\n        time.sleep(5)\n        if not move_xy(0, 0):\n            return\n        time.sleep(2)\n        if not move_z(0):\n            return\n        time.sleep(2)\n\n        runner = ImageImpulseRunner(MODEL_PATH)\n        runner.init()\n\n        cap = cv2.VideoCapture(CAMERA_ID)\n        if not cap.isOpened():\n            print("摄像头打开失败")\n            return\n\n        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)\n        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)\n\n        crop_w = int(FRAME_WIDTH / ZOOM_FACTOR)\n        crop_h = int(FRAME_HEIGHT / ZOOM_FACTOR)\n        start_x = (FRAME_WIDTH - crop_w) // 2\n        start_y = (FRAME_HEIGHT - crop_h) // 2\n\n        cv2.namedWindow("Real-time Camera", cv2.WINDOW_NORMAL)\n        cv2.resizeWindow("Real-time Camera", DISPLAY_WIDTH, int(DISPLAY_WIDTH * crop_h / crop_w))\n        cv2.namedWindow("Frozen Frame", cv2.WINDOW_NORMAL)\n        cv2.resizeWindow("Frozen Frame", DISPLAY_WIDTH, int(DISPLAY_WIDTH * crop_h / crop_w))\n\n        print("实时窗口：只显示实时画面")\n        print("定格窗口：显示 4 点编号、四边形、对角线和交点（端正显示）")\n        print("按 q 可随时退出")\n\n        center_uv, quad = capture_lid_center(cap, runner, start_x, start_y, crop_w, crop_h)\n        print(f"\\n交点像素坐标: ({center_uv[0]:.2f}, {center_uv[1]:.2f})")\n\n        suction_x, suction_y = suction_pixel_to_robot(center_uv[0], center_uv[1])\n        print(f"工具偏置修正后的吸盘机械坐标: X={suction_x}, Y={suction_y}")\n\n        if pickup_lid_and_finish(suction_x, suction_y, cap, runner, start_x, start_y, crop_w, crop_h):\n            print("\\n吸上盖流程完成，程序结束")\n        else:\n            print("\\n吸上盖流程失败，程序结束")\n\n    except KeyboardInterrupt:\n        print("\\n用户中断")\n    finally:\n        try:\n            if cap is not None:\n                cap.release()\n        except Exception:\n            pass\n        try:\n            cv2.destroyAllWindows()\n        except Exception:\n            pass\n        try:\n            if runner is not None:\n                runner.stop()\n        except Exception:\n            pass\n        try:\n            tt_safe_call(TT.Downline, 0, action_name="TT 下线")\n        except Exception:\n            pass\n        try:\n            stop_suction_safely()\n        except Exception:\n            pass\n        cleanup_gpio()\n\n\nif __name__ == "__main__":\n    main()\n'


def load_embedded_module(name: str, source: str):
    mod = types.ModuleType(name)
    mod.__file__ = f"<{name}>"
    code = compile(source, mod.__file__, "exec")
    exec(code, mod.__dict__)
    return mod


screw_ref = load_embedded_module("screw_ref", OPEN_LOOP_SOURCE)
lid_ref = load_embedded_module("lid_ref", LID_PICK_SOURCE)

# ==================== 用户可直接修改的基本参数（集中在顶部） ====================
# 下面这些参数，都是你现场最常改的；以后优先改这里，不用再翻下面的函数。

# -------------------- 一、相机与视觉显示参数 --------------------
CAMERA_ID = 0                      # 摄像头 ID，默认 0；如果有多个摄像头可改成 1、2...
MODEL_PATH = "/home/hk/.ei-linux-runner/models/886864/v4-runner-linux-armv7-impulse3/model.eim"  # Edge Impulse 模型路径
FRAME_WIDTH = 1280                 # 相机采集宽度
FRAME_HEIGHT = 720                 # 相机采集高度
DISPLAY_WIDTH = 512                # OpenCV 窗口显示宽度
OPEN_LOOP_ZOOM_FACTOR = 4.2        # 视觉裁剪放大倍数，数值越大，视野越小
CONFIDENCE_THRESHOLD = 0.5         # 检测置信度阈值
DRAW_BOXES_ON_REALTIME = False     # 实时视频是否画检测框；False=不画框，只显示画面
ROTATE_REALTIME_DISPLAY = True     # 实时窗口是否翻转 180°
ROTATE_FROZEN_DISPLAY = False      # 定格窗口是否翻转 180°
SHOW_REALTIME_WINDOW = False      # 是否显示实时窗口；当前关闭，只保留定格窗口

# -------------------- 二、机械臂通信参数 --------------------
SERIAL_PORT = "/dev/ttyUSB0"      # 机械臂控制串口
BAUDRATE = 9600                    # 串口波特率

# -------------------- 三、拆螺丝阶段参数 --------------------
SCREW_Z_GRASP = 0                  # 拆螺丝时 Z 轴下放高度，当前按你的要求设为 0
SCREW_PLACE_X = 0                  # 螺丝放置点 X 坐标
SCREW_PLACE_Y = 0                  # 螺丝放置点 Y 坐标
SCREW_XY_SPEED = 60                # 拆螺丝时 XY 速度
SCREW_XY_ACCEL = 40                # 拆螺丝时 XY 加速度
SCREW_XY_DECEL = 40                # 拆螺丝时 XY 减速度
SCREW_Z_SPEED = 30                 # 拆螺丝时 Z 速度
SCREW_Z_ACCEL = 30                 # 拆螺丝时 Z 加速度
SCREW_Z_DECEL = 30                 # 拆螺丝时 Z 减速度
SCREW_MAGNET_HOLD_TIME = 2.2       # 电磁铁吸附建立时间（秒）
SCREW_INTER_TARGET_DELAY = 1.0     # 两颗螺丝之间的停顿时间（秒）
SCREW_TARGET_SETTLE = 0.5          # XY 到达螺丝上方后的稳定时间（秒）
SCREW_Z_AT_GRASP_SETTLE = 0.5      # Z 到达抓取高度后的稳定时间（秒）
SCREW_RELEASE_SETTLE = 0.5         # 到达放置位后，释放前的稳定时间（秒）

# -------------------- 四、上盖吸取与放置参数 --------------------
LID_PLACE_X = 0                    # 上盖放置点 X 坐标
LID_PLACE_Y = 90000                # 上盖放置点 Y 坐标，这里是 9 cm
RETURN_HOME_AFTER_LID = True       # 吸完上盖并放下后，是否回到初始位置
HOME_X = 0                         # 回初始位置时的 X 坐标
HOME_Y = 0                         # 回初始位置时的 Y 坐标
HOME_Z = 0                         # 回初始位置时的 Z 坐标

# -------------------- 五、坐标变换标定矩阵 --------------------
# 这是你现在正在使用的像素点 -> 机械臂点矩阵。
# 如果你重新踩点，只改下面两组点，不用改下面的函数。
CALIB_PIXEL_POINTS = [
    [538, 316], [640, 316], [741, 316],
    [538, 402], [640, 402], [741, 402],
]
CALIB_ROBOT_POINTS = [
    [138015, 85020], [84015, 85020], [30015, 85020],
    [138015, 40020], [84015, 40020], [30015, 40020],
]
# =============================================================================


def _process_frame_no_boxes(frame, runner, start_x, start_y, crop_w, crop_h):
    """实时视频只显示画面，不画框；仍然返回检测中心点。"""
    cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w].copy()
    if cropped.size == 0:
        return [], None

    resized = screw_ref.cv2.resize(cropped, (96, 96))
    rgb = screw_ref.cv2.cvtColor(resized, screw_ref.cv2.COLOR_BGR2RGB)
    features, _ = runner.get_features_from_image(rgb)
    res = runner.classify(features)
    if isinstance(res, str):
        try:
            res = screw_ref.json.loads(res)
        except Exception:
            return [], None

    boxes = [b for b in res.get("result", {}).get("bounding_boxes", [])
             if b["value"] > screw_ref.CONFIDENCE_THRESHOLD]

    display_scale = screw_ref.DISPLAY_WIDTH / crop_w
    display_img = screw_ref.cv2.resize(cropped, (screw_ref.DISPLAY_WIDTH, int(crop_h * display_scale)))

    scale_x = crop_w / 96.0
    scale_y = crop_h / 96.0

    centers = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        cx = (x + w / 2) * scale_x + start_x
        cy = (y + h / 2) * scale_y + start_y
        centers.append((cx, cy, b["label"], b["value"]))

    centers.sort(key=lambda item: item[3], reverse=True)
    return centers, display_img


def _patched_screw_move_xy(x, y):
    return screw_ref.TT.AxleMoveAbsolute(1, SCREW_XY_SPEED, SCREW_XY_ACCEL, SCREW_XY_DECEL, int(x)) and \
           screw_ref.TT.AxleMoveAbsolute(2, SCREW_XY_SPEED, SCREW_XY_ACCEL, SCREW_XY_DECEL, int(y))


def _patched_screw_move_z(z):
    return screw_ref.TT.AxleMoveAbsolute(4, SCREW_Z_SPEED, SCREW_Z_ACCEL, SCREW_Z_DECEL, int(z))


def _patched_screw_pick_and_place(tx, ty, cap, runner, start_x, start_y, crop_w, crop_h):
    """v5 精修：保持原开环链路，但把吸附/释放时序收紧。"""
    # 1) 先抬到安全位
    if not screw_ref.move_z(screw_ref.Z_SAFE):
        return False
    screw_ref.refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)

    # 2) XY 到目标位
    if not screw_ref.move_xy(tx, ty):
        return False
    screw_ref.refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    time.sleep(SCREW_TARGET_SETTLE)

    # 3) Z 到 0cm 抓取位，到位后先停一下再吸
    if not screw_ref.move_z(screw_ref.Z_GRASP):
        return False
    screw_ref.refresh_display(1, cap, runner, start_x, start_y, crop_w, crop_h)
    time.sleep(SCREW_Z_AT_GRASP_SETTLE)

    # 4) 再开启电磁铁，并给足吸附建立时间
    screw_ref.GPIO.output(screw_ref.MAGNET_PIN, screw_ref.GPIO.LOW)
    time.sleep(SCREW_MAGNET_HOLD_TIME)
    screw_ref.cv2.waitKey(1)

    # 5) 抬回安全位
    if not screw_ref.move_z(screw_ref.Z_SAFE):
        return False
    screw_ref.refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)

    # 6) 再移动到螺丝放置位（默认 0,0）
    if not screw_ref.move_xy(screw_ref.PLACE_X, screw_ref.PLACE_Y):
        return False
    screw_ref.refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)
    time.sleep(SCREW_RELEASE_SETTLE)

    # 7) 到位后才释放电磁铁
    screw_ref.GPIO.output(screw_ref.MAGNET_PIN, screw_ref.GPIO.HIGH)
    time.sleep(0.8)
    screw_ref.cv2.waitKey(1)
    return True


def apply_user_overrides():
    """把顶部参数统一覆盖到拆螺丝模块和上盖模块，方便现场只改最上面。"""
    # ---------- 视觉与相机参数 ----------
    screw_ref.CAMERA_ID = CAMERA_ID
    lid_ref.CAMERA_ID = CAMERA_ID
    screw_ref.MODEL_PATH = MODEL_PATH
    lid_ref.MODEL_PATH = MODEL_PATH
    screw_ref.FRAME_WIDTH = FRAME_WIDTH
    screw_ref.FRAME_HEIGHT = FRAME_HEIGHT
    lid_ref.FRAME_WIDTH = FRAME_WIDTH
    lid_ref.FRAME_HEIGHT = FRAME_HEIGHT
    screw_ref.DISPLAY_WIDTH = DISPLAY_WIDTH
    lid_ref.DISPLAY_WIDTH = DISPLAY_WIDTH
    screw_ref.ZOOM_FACTOR = OPEN_LOOP_ZOOM_FACTOR
    lid_ref.ZOOM_FACTOR = OPEN_LOOP_ZOOM_FACTOR
    screw_ref.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    lid_ref.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    screw_ref.ROTATE_DISPLAY = ROTATE_REALTIME_DISPLAY
    lid_ref.ROTATE_LIVE_DISPLAY = ROTATE_REALTIME_DISPLAY
    lid_ref.ROTATE_FROZEN_DISPLAY = ROTATE_FROZEN_DISPLAY

    # ---------- 串口参数 ----------
    screw_ref.SERIAL_PORT = SERIAL_PORT
    screw_ref.BAUDRATE = BAUDRATE
    lid_ref.SERIAL_PORT = SERIAL_PORT
    lid_ref.BAUDRATE = BAUDRATE

    # ---------- 拆螺丝阶段参数 ----------
    screw_ref.Z_GRASP = SCREW_Z_GRASP
    screw_ref.MAGNET_HOLD_TIME = SCREW_MAGNET_HOLD_TIME
    screw_ref.INTER_TARGET_DELAY = SCREW_INTER_TARGET_DELAY
    screw_ref.PLACE_X = SCREW_PLACE_X
    screw_ref.PLACE_Y = SCREW_PLACE_Y

    # 用更稳的速度覆盖拆螺丝阶段运动函数
    screw_ref.move_xy = _patched_screw_move_xy
    screw_ref.move_z = _patched_screw_move_z
    screw_ref.pick_and_place = _patched_screw_pick_and_place

    # ---------- 上盖放置参数 ----------
    lid_ref.PLACE_X = LID_PLACE_X
    lid_ref.PLACE_Y = LID_PLACE_Y

    # ---------- 坐标变换矩阵 ----------
    # 同一套矩阵同时用于拆螺丝和上盖吸取，避免前后不一致
    screw_ref.pixel_points = screw_ref.np.array(CALIB_PIXEL_POINTS, dtype=screw_ref.np.float32)
    screw_ref.robot_points = screw_ref.np.array(CALIB_ROBOT_POINTS, dtype=screw_ref.np.float32)
    screw_ref.H, _ = screw_ref.cv2.findHomography(screw_ref.pixel_points, screw_ref.robot_points)

    lid_ref.pixel_points = lid_ref.np.array(CALIB_PIXEL_POINTS, dtype=lid_ref.np.float32)
    lid_ref.robot_points = lid_ref.np.array(CALIB_ROBOT_POINTS, dtype=lid_ref.np.float32)
    lid_ref.H, _ = lid_ref.cv2.findHomography(lid_ref.pixel_points, lid_ref.robot_points)

    # ---------- 实时窗口是否画框 ----------
    if not DRAW_BOXES_ON_REALTIME:
        screw_ref.process_frame = _process_frame_no_boxes


def init_screw_gpio():
    screw_ref.GPIO.setwarnings(False)
    screw_ref.GPIO.setmode(screw_ref.GPIO.BCM)
    screw_ref.GPIO.setup(screw_ref.MAGNET_PIN, screw_ref.GPIO.OUT)
    screw_ref.GPIO.output(screw_ref.MAGNET_PIN, screw_ref.GPIO.HIGH)


def robot_init():
    # 按《开环自动抓取》原始初始化思路执行，不改其内部代码
    screw_ref.TT.Link(screw_ref.SERIAL_PORT, screw_ref.BAUDRATE, timeout=5)
    time.sleep(0.5)
    if not screw_ref.TT.Test_Call():
        print("TT 通讯失败")
        return False
    screw_ref.ensure_robot_ready()
    screw_ref.TT.AxleToZero(7)
    time.sleep(5)
    screw_ref.move_xy(0, 0)
    screw_ref.refresh_display(2, None, None, None, None, None, None)
    screw_ref.move_z(0)
    screw_ref.refresh_display(2, None, None, None, None, None, None)
    return True


def open_camera_and_runner():
    runner = screw_ref.ImageImpulseRunner(screw_ref.MODEL_PATH)
    runner.init()
    cap = screw_ref.cv2.VideoCapture(screw_ref.CAMERA_ID)
    if not cap.isOpened():
        runner.stop()
        print("摄像头打开失败")
        return None, None
    cap.set(screw_ref.cv2.CAP_PROP_FRAME_WIDTH, screw_ref.FRAME_WIDTH)
    cap.set(screw_ref.cv2.CAP_PROP_FRAME_HEIGHT, screw_ref.FRAME_HEIGHT)
    return cap, runner


def get_crop_params():
    crop_w = int(screw_ref.FRAME_WIDTH / screw_ref.ZOOM_FACTOR)
    crop_h = int(screw_ref.FRAME_HEIGHT / screw_ref.ZOOM_FACTOR)
    start_x = (screw_ref.FRAME_WIDTH - crop_w) // 2
    start_y = (screw_ref.FRAME_HEIGHT - crop_h) // 2
    return start_x, start_y, crop_w, crop_h


def show_windows(crop_w, crop_h):
    if SHOW_REALTIME_WINDOW:
        screw_ref.cv2.namedWindow("Real-time Camera", screw_ref.cv2.WINDOW_NORMAL)
        screw_ref.cv2.resizeWindow(
            "Real-time Camera",
            screw_ref.DISPLAY_WIDTH,
            int(screw_ref.DISPLAY_WIDTH * crop_h / crop_w),
        )
    screw_ref.cv2.namedWindow("Frozen Frame", screw_ref.cv2.WINDOW_NORMAL)
    screw_ref.cv2.resizeWindow(
        "Frozen Frame",
        screw_ref.DISPLAY_WIDTH,
        int(screw_ref.DISPLAY_WIDTH * crop_h / crop_w),
    )


def detect_initial_screws_and_center(cap, runner, start_x, start_y, crop_w, crop_h):
    print("开始识别初始四个螺丝点...")
    while True:
        ret, frame = cap.read()
        if not ret:
            if screw_ref.cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
            continue

        centers, display_img = screw_ref.process_frame(frame, runner, start_x, start_y, crop_w, crop_h)

        if SHOW_REALTIME_WINDOW and display_img is not None:
            if screw_ref.ROTATE_DISPLAY:
                display_img = screw_ref.cv2.rotate(display_img, screw_ref.cv2.ROTATE_180)
            screw_ref.cv2.imshow("Real-time Camera", display_img)

        key = screw_ref.cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt

        if len(centers) >= 4:
            frozen_centers = centers[:4].copy()
            pts = [(item[0], item[1]) for item in frozen_centers]
            quad = lid_ref.order_points_clockwise(pts)
            center_uv = lid_ref.line_intersection(quad[0], quad[2], quad[1], quad[3])

            ret_f, frame_f = cap.read()
            if ret_f:
                freeze_img = lid_ref.make_lid_frozen_view(
                    frame_f, quad, center_uv, start_x, start_y, crop_w, crop_h
                )
                if freeze_img is not None:
                    screw_ref.cv2.imshow("Frozen Frame", freeze_img)
                    print("已冻结初始四个螺丝点，并计算上盖中心")
                    wait_end = time.time() + 3
                    while time.time() < wait_end:
                        ret_live, frame_live = cap.read()
                        if ret_live:
                            _, live_disp = screw_ref.process_frame(
                                frame_live, runner, start_x, start_y, crop_w, crop_h
                            )
                            if SHOW_REALTIME_WINDOW and live_disp is not None:
                                if screw_ref.ROTATE_DISPLAY:
                                    live_disp = screw_ref.cv2.rotate(live_disp, screw_ref.cv2.ROTATE_180)
                                screw_ref.cv2.imshow("Real-time Camera", live_disp)
                        if screw_ref.cv2.waitKey(50) & 0xFF == ord("q"):
                            raise KeyboardInterrupt

            return frozen_centers, quad, center_uv


def run_screw_phase(frozen_centers, cap, runner, start_x, start_y, crop_w, crop_h):
    print("\n开始按《开环自动抓取》原流程依次拆螺丝...")
    print("螺丝阶段采用开环执行：冻结初始点后，不再刷新摄像头，避免 V4L2 timeout")
    for idx, (u, v, lab, conf, *rest) in enumerate(frozen_centers, start=1):
        print(f"\n--- 处理螺丝 {idx} / {len(frozen_centers)} ---")
        tx, ty = screw_ref.pixel_to_robot(u, v)
        print(f"像素坐标=({u:.1f}, {v:.1f}) -> 机器人坐标=({tx}, {ty})")
        ok = screw_ref.pick_and_place(tx, ty, None, None, start_x, start_y, crop_w, crop_h)
        if ok:
            print(f"螺丝 {idx} 完成")
        else:
            print(f"螺丝 {idx} 失败，尝试继续")
            screw_ref.ensure_robot_ready()
        if idx < len(frozen_centers):
            time.sleep(screw_ref.INTER_TARGET_DELAY)

    print("\n螺丝阶段完成，保持 Z 轴安全高度，直接衔接上盖流程")
    screw_ref.move_z(screw_ref.Z_SAFE)
    time.sleep(0.8)


def run_lid_phase(center_uv, cap, runner, start_x, start_y, crop_w, crop_h):
    print("\n开始衔接《上盖吸取》流程...")
    suction_x, suction_y = lid_ref.suction_pixel_to_robot(center_uv[0], center_uv[1])
    print(f"上盖中心像素坐标: ({center_uv[0]:.2f}, {center_uv[1]:.2f})")
    print(f"偏置修正后的吸盘机械坐标: X={suction_x}, Y={suction_y}")

    # 上盖中心已经算完，后半段不再需要摄像头；直接关掉，避免 V4L2 select() timeout
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        screw_ref.cv2.destroyAllWindows()
    except Exception:
        pass

    lid_ref.setup_gpio()
    ok = lid_ref.pickup_lid_and_finish(
        suction_x, suction_y, None, None, start_x, start_y, crop_w, crop_h
    )

    if not ok:
        return False

    if RETURN_HOME_AFTER_LID:
        print("\n上盖释放完成，保持当前高度，直接回到初始位置...")
        # 用户要求：上盖释放完成后不再先抬升到安全高度。
        # 参考上盖代码内部结束时已完成自身收尾，外层这里只做稳定等待后回原点。
        time.sleep(0.8)
        if not _move_xy_with_retry(HOME_X, HOME_Y, "保持当前高度，直接回到原点 XY"):
            return False
        time.sleep(1.0)
        if not _move_z_with_retry(HOME_Z, "最后回到原点 Z"):
            return False
        time.sleep(1.0)

    return True




def _retry_lid_robot_ready():
    try:
        if hasattr(lid_ref, "ensure_robot_ready"):
            return bool(lid_ref.ensure_robot_ready())
        lid_ref.TT.ALARMReset()
        lid_ref.TT.AxleEnabled(7, 1)
        time.sleep(0.3)
        return True
    except Exception:
        return False


def _move_z_with_retry(z, label):
    print(f"[上盖回零] {label}: Z -> {int(z)}")
    if lid_ref.move_z(z):
        return True
    print("[上盖回零] Z 轴第一次执行失败，复位报警后重试一次")
    _retry_lid_robot_ready()
    time.sleep(0.5)
    return lid_ref.move_z(z)


def _move_xy_with_retry(x, y, label):
    print(f"[上盖回零] {label}: XY -> ({int(x)}, {int(y)})")
    if lid_ref.move_xy(x, y):
        return True
    print("[上盖回零] XY 第一次执行失败，复位报警后重试一次")
    _retry_lid_robot_ready()
    time.sleep(0.5)
    return lid_ref.move_xy(x, y)

def final_cleanup(cap, runner):
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        screw_ref.cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        if runner is not None:
            runner.stop()
    except Exception:
        pass
    try:
        lid_ref.stop_suction_safely()
    except Exception:
        pass
    try:
        lid_ref.cleanup_gpio()
    except Exception:
        pass
    try:
        screw_ref.TT.Downline(0)
    except Exception:
        pass
    try:
        screw_ref.GPIO.cleanup()
    except Exception:
        pass


def main():
    cap = None
    runner = None
    try:
        apply_user_overrides()
        init_screw_gpio()
        if not robot_init():
            return

        cap, runner = open_camera_and_runner()
        if cap is None or runner is None:
            return

        start_x, start_y, crop_w, crop_h = get_crop_params()
        show_windows(crop_w, crop_h)

        frozen_centers, quad, center_uv = detect_initial_screws_and_center(
            cap, runner, start_x, start_y, crop_w, crop_h
        )

        run_screw_phase(frozen_centers, cap, runner, start_x, start_y, crop_w, crop_h)

        ok = run_lid_phase(center_uv, cap, runner, start_x, start_y, crop_w, crop_h)
        if ok:
            print("\n整合流程完成")
        else:
            print("\n上盖吸取流程失败")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        final_cleanup(cap, runner)


if __name__ == "__main__":
    main()


        # 9. Z轴上升至安全高度
        if not move_z(Z_SAFE):
            print("吸上盖：Z轴上升失败")
            return
        refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)

        # 10. 移动XY到原点(0,0)
        if not move_xy(0, 0):
            print("吸上盖：移动到原点失败")
            return
        refresh_display(2, cap, runner, start_x, start_y, crop_w, crop_h)

        # 11. 打开电磁阀（释放）
        GPIO.output(MAGNET_PIN, GPIO.HIGH)
        print("电磁阀打开，吸盘释放上盖")
        time.sleep(1)

        # 12. 步进电机顺时针旋转180°
        print("步进电机顺时针旋转180°")
        stepper_rotate(180)
        time.sleep(1)

        # 13. 关闭气泵
        pump.ChangeDutyCycle(PUMP_DUTY_OFF)
        print("气泵关闭")

        print("\n全部动作完成，程序退出")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        runner.stop()
        TT.Downline(0)
        pump.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()