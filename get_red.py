import numpy as np
import cv2

# 이미지 파일 읽기
src = cv2.imread("origin.png")
if src is None:
    print("origin.png 파일을 찾을 수 없습니다.")
    exit()

# BGR → HSV 변환
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# 빨간색 HSV 범위 정의 (2개)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 두 범위를 마스크로 만들고 OR 연산
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask1, mask2)

# 빨간색 영역 추출
res_red = cv2.bitwise_and(src, src, mask=mask_red)

# 빨간색 객체 윤곽선 검출
gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
_, bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = None
largest_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > largest_area:
        largest_area = area
        largest_contour = cnt

# 가장 큰 객체에 빨간 박스 그리기
if largest_contour is not None and largest_area > 500:
    x, y, width, height = cv2.boundingRect(largest_contour)
    cv2.rectangle(src, (x, y), (x + width, y + height), (0, 0, 255), 2)
    center_x = x + width // 2
    center_y = y + height // 2
    print("center: (%d, %d)" % (center_x, center_y))

# 결과 출력
cv2.imshow("Original + Red Box", src)
cv2.imshow("Red Extracted", res_red)

# 빨간색 추출 이미지 저장
cv2.imwrite("red.png", res_red)

# 종료 대기
print("ESC 키를 누르면 종료됩니다.")
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

