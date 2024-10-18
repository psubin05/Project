import cv2
import numpy as np

# 얼굴 인식을 위한 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 불러오기
def load_image(image_path):
    return cv2.imread(image_path)

# 얼굴 인식 함수
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # 얼굴 인식
    return faces

# 이미지에서 얼굴을 인식하고 표시하는 함수
def recognize_faces(image_path):
    image = load_image(image_path)
    faces = detect_face(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴에 사각형 그리기

    # 결과 이미지 출력
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 사용자 얼굴 이미지 파일 경로
image_path = 'path/to/your/image.jpg'  # 여기에 이미지 파일 경로를 입력하세요
recognize_faces(image_path)
