import cv2

cap = cv2.VideoCapture(5)

while True:
    ret, frame = cap.read()  # read camera frame
    print(ret)
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 단순히 frame을 출력
    cv2.imshow("Camera Frame", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

