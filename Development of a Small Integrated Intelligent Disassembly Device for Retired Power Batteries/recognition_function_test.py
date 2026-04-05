#!/usr/bin/env python3
"""
Edge Impulse 识别测试代码（无机械臂控制）
功能：
- 实时显示放大画面（4倍），可翻转180°显示。
- 检测目标并绘制绿色边框和红色中心点。
- 在终端实时打印每个目标的像素坐标和置信度。
- 按 'q' 键退出。
"""

import cv2
import json
import time
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

# ==================== 配置 ====================
MODEL_PATH = "/home/hk/.ei-linux-runner/models/886864/v4-runner-linux-armv7-impulse3/model.eim"
CAMERA_ID = 0                          # 根据实际情况修改
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ZOOM_FACTOR = 4.0                       # 放大倍数
DISPLAY_WIDTH = 512                      # 显示窗口宽度
CONFIDENCE_THRESHOLD = 0.5                # 置信度阈值
ROTATE_DISPLAY = True                     # 是否翻转显示画面
# ==============================================

def process_frame(frame, runner, start_x, start_y, crop_width, crop_height):
    """对一帧图像进行推理，返回带标注的显示图像"""
    cropped = frame[start_y:start_y+crop_height, start_x:start_x+crop_width].copy()
    if cropped.size == 0:
        return None

    resized = cv2.resize(cropped, (96, 96))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    features, _ = runner.get_features_from_image(rgb)
    res = runner.classify(features)
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except:
            return None

    boxes = res.get("result", {}).get("bounding_boxes", [])
    boxes = [b for b in boxes if b["value"] > CONFIDENCE_THRESHOLD]

    display_scale = DISPLAY_WIDTH / crop_width
    display_img = cv2.resize(cropped, (DISPLAY_WIDTH, int(crop_height * display_scale)))

    scale_x = crop_width / 96.0
    scale_y = crop_height / 96.0

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        conf = box["value"]
        label = box["label"]
        # 计算中心在原始图像中的坐标
        cx = (x + w/2) * scale_x + start_x
        cy = (y + h/2) * scale_y + start_y
        print(f"检测到 {label} 置信度 {conf:.2f} 像素中心 ({cx:.1f}, {cy:.1f})")

        # 绘制边框（映射到显示图像）
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        dx1, dy1 = int(x1 * display_scale), int(y1 * display_scale)
        dx2, dy2 = int(x2 * display_scale), int(y2 * display_scale)
        cv2.rectangle(display_img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

        # 绘制中心红点
        cd_x = int(((x + w/2) * scale_x) * display_scale)
        cd_y = int(((y + h/2) * scale_y) * display_scale)
        cv2.circle(display_img, (cd_x, cd_y), 5, (0, 0, 255), -1)

    return display_img

def main():
    # 计算裁剪区域
    crop_w = int(FRAME_WIDTH / ZOOM_FACTOR)
    crop_h = int(FRAME_HEIGHT / ZOOM_FACTOR)
    start_x = (FRAME_WIDTH - crop_w) // 2
    start_y = (FRAME_HEIGHT - crop_h) // 2
    print(f"裁剪区域：({start_x}, {start_y}) 大小 {crop_w} x {crop_h}")

    # 初始化模型
    runner = ImageImpulseRunner(MODEL_PATH)
    runner.init()
    print("模型加载成功")

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        runner.stop()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow("Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection Test", DISPLAY_WIDTH, int(DISPLAY_WIDTH * crop_h / crop_w))

    print("\n实时检测中，按 'q' 退出...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_img = process_frame(frame, runner, start_x, start_y, crop_w, crop_h)
        if display_img is not None:
            if ROTATE_DISPLAY:
                display_img = cv2.rotate(display_img, cv2.ROTATE_180)
            cv2.imshow("Detection Test", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    runner.stop()
    print("程序退出")

if __name__ == "__main__":
    main()