import cv2
import numpy as np

import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # for grayscale mask
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is None:
        return
    img = np.copy(img)
    blank_img = np.zeros_like(img)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

    img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
    return img

cap = cv2.VideoCapture(0)  # 또는 동영상 파일: 'lane_video.mp4'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    
    # Grayscale & Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # ROI 설정 (삼각형 형태로 도로 아래 부분만)
    roi_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )

    # 차선 선 그리기
    frame_with_lines = draw_lines(frame, lines)

    # 결과 출력
    cv2.imshow('Lane Detection', frame_with_lines)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

