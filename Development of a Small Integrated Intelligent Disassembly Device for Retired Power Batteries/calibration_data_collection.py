#!/usr/bin/env python3
"""
标定数据采集脚本 - 在放大画面内均匀分布6个点（支持画面翻转，数字正向）
放大倍数改为4倍，画面翻转后数字保持正向。
"""

import cv2
import numpy as np

# ==================== 配置 ====================
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ZOOM_FACTOR = 4.0          # 改为4倍
DISPLAY_WIDTH = 512
ROTATE_DISPLAY = True      # 是否将显示画面旋转180°（仅显示）
# ==============================================

# 计算裁剪区域
crop_width = int(FRAME_WIDTH / ZOOM_FACTOR)
crop_height = int(FRAME_HEIGHT / ZOOM_FACTOR)
start_x = (FRAME_WIDTH - crop_width) // 2
start_y = (FRAME_HEIGHT - crop_height) // 2
print(f"裁剪区域：({start_x}, {start_y}) 大小 {crop_width} x {crop_height}")

# 在裁剪区域内生成6个均匀分布的点（2行3列）
points_crop = []  # 裁剪图像内的坐标
rows, cols = 2, 3
for row in range(rows):
    for col in range(cols):
        crop_u = int((col + 0.5) * crop_width / cols)
        crop_v = int((row + 0.5) * crop_height / rows)
        points_crop.append((crop_u, crop_v))

# 转换为原始图像坐标
points_pixel = [(x + start_x, y + start_y) for x, y in points_crop]

print("\n六个标定点的原始图像坐标：")
for i, (u, v) in enumerate(points_pixel):
    print(f"点{i+1}: ({u}, {v})")

# 存储机械臂坐标（毫米）
points_robot = [None] * 6

def draw_points(img, points_crop, collected):
    """在显示图像上绘制点，并根据ROTATE_DISPLAY决定是否翻转，且数字保持正向"""
    # 计算显示缩放比例
    display_scale = DISPLAY_WIDTH / crop_width
    h_disp = int(crop_height * display_scale)
    # 创建显示图像（直接从裁剪区域缩放）
    cropped = img[start_y:start_y+crop_height, start_x:start_x+crop_width].copy()
    display_img = cv2.resize(cropped, (DISPLAY_WIDTH, h_disp))

    # 存储每个点的显示坐标（未旋转）
    disp_coords = []
    for i, (cu, cv_) in enumerate(points_crop):
        dx = int(cu * display_scale)
        dy = int(cv_ * display_scale)
        disp_coords.append((dx, dy))
        color = (0, 255, 0) if collected[i] else (0, 0, 255)
        cv2.drawMarker(display_img, (dx, dy), color, cv2.MARKER_CROSS, 20, 2)

    if ROTATE_DISPLAY:
        # 旋转图像
        display_img = cv2.rotate(display_img, cv2.ROTATE_180)
        # 在旋转后的图像上重新绘制数字（正向）
        for i, (dx, dy) in enumerate(disp_coords):
            # 计算旋转后的坐标
            rx = display_img.shape[1] - 1 - dx
            ry = display_img.shape[0] - 1 - dy
            color = (0, 255, 0) if collected[i] else (0, 0, 255)
            cv2.putText(display_img, str(i+1), (rx+15, ry-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # 不旋转，直接在原显示图像上绘制数字
        for i, (dx, dy) in enumerate(disp_coords):
            color = (0, 255, 0) if collected[i] else (0, 0, 255)
            cv2.putText(display_img, str(i+1), (dx+15, dy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return display_img

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    collected = [False] * 6
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", DISPLAY_WIDTH, int(DISPLAY_WIDTH * crop_height / crop_width))

    print("\n=== 标定点采集（放大画面内，4倍变焦）===")
    print("画面中的十字标记对应放大画面内的均匀分布点。")
    if ROTATE_DISPLAY:
        print("画面已旋转180°，但数字保持正向，便于观察。")
    print("请依次将机械臂末端移动到每个标记对应的物理位置，然后按下对应的数字键1-6，")
    print("并在终端输入该点的机械臂 X Y 坐标（毫米，空格分隔）。")
    print("按 'q' 退出。\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_img = draw_points(frame, points_crop, collected)
        cv2.imshow("Calibration", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('6'):
            idx = key - ord('1')
            if collected[idx]:
                print(f"点 {idx+1} 已采集，如需重采请重启程序。")
                continue

            u, v = points_pixel[idx]
            print(f"\n点 {idx+1} 像素坐标: ({u}, {v})")
            print("请输入机械臂 X Y 坐标（毫米）：")
            try:
                inp = input("X Y: ").strip().split()
                if len(inp) != 2:
                    print("格式错误，请重新按数字键输入。")
                    continue
                X = float(inp[0])
                Y = float(inp[1])
                points_robot[idx] = (X, Y)
                collected[idx] = True
                print(f"点 {idx+1} 记录成功: ({X:.1f}, {Y:.1f})")
            except ValueError:
                print("输入无效，请重新按数字键输入。")

        if all(collected):
            print("\n所有点采集完成！")
            src_pts = np.array(points_pixel, dtype=np.float32)
            dst_pts = np.array(points_robot, dtype=np.float32)
            H, mask = cv2.findHomography(src_pts, dst_pts)
            if H is not None:
                print("\n单应性矩阵 H =")
                print(H)
                np.save("homography_zoom.npy", H)
                print("已保存到 homography_zoom.npy")
                # 重投影误差
                proj = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), H).reshape(-1,2)
                errors = np.linalg.norm(proj - dst_pts, axis=1)
                print("\n重投影误差（毫米）：")
                for i, err in enumerate(errors):
                    print(f"点{i+1}: {err:.3f}")
                print(f"平均误差: {np.mean(errors):.3f}")
            else:
                print("计算单应性矩阵失败。")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()