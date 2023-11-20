import time
import cv2
import numpy as np
import os

# YOLOv7 모델 설정 및 가중치 경로 설정
yolo_config = "yolov7x.cfg"
yolo_weights = "yolov7x.weights"
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# 사람 클래스의 인덱스
person_class_id = 0

# 디렉토리 경로 설정
print('컴아트 비식별화(sleep 0.2) - yolov7x / 416size / auto_fps / blur 7 container-mp4 23-11-09 proto -version 0.3\n')
base_dir = input("비디오 파일이 있는 디렉토리(상위폴더) 경로를 입력하세요: \n\n")

# 모든 파일에 대한 반복
for root, dirs, files in os.walk(base_dir):
    for file in files:
        try:
            if file.endswith(".mp4"):
                video_file = os.path.join(root, file)
                output_file = os.path.join(root, os.path.splitext(file)[0] + "_MOS.mp4")

                # OpenCV 비디오 캡처 객체 생성
                cap = cv2.VideoCapture(video_file)

                # 원본 동영상의 프레임 속도 가져오기
                original_fps = cap.get(cv2.CAP_PROP_FPS)

                # 비디오 녹화 객체 생성
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output = cv2.VideoWriter(output_file, fourcc, original_fps, (int(cap.get(3)), int(cap.get(4))))  # 원본 fps 설정

                # 마지막 ( detection 객체값을 최댓값으로 고정 하기위해 temp 선언)
                last_detection_boxes = []
                last_confidences = []

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    height, width, _ = frame.shape

                    # YOLOv7를 사용하여 객체 감지
                    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(net.getUnconnectedOutLayersNames())

                    confidences = []
                    boxes = []

                    for out in outs:
                        for detection in out:
                            try:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]

                                if class_id == person_class_id and confidence > 0.25:  # 감지 임계값을 낮추기
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)
                                    x = center_x - w // 2
                                    y = center_y - h // 2

                                    confidences.append(float(confidence))
                                    boxes.append([x, y, w, h])
                            except:continue

                    # 디텍션 값 저장 ( 박스는 항상 최댓값을 유지하기 위해 )
                    if(len(last_detection_boxes) <= len(boxes)):
                        last_detection_boxes = boxes[:]
                    if(len(last_detection_boxes) > len(boxes)):
                        boxes = last_detection_boxes[:]
                    if(len(last_confidences) <= len(confidences)):
                        last_confidences = confidences[:]
                    if(len(last_confidences) > len(confidences)):
                        confidences = last_confidences[:]

                    # 비최대 억제(NMS)를 사용하여 중복된 감지 제거 (NMS 기본값으로 변경)
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)  # NMS 임계값 변경

                    for i in range(len(boxes)):
                        try:
                            if i in indices:
                                x, y, w, h = boxes[i]
                                roi = frame[y:y+h, x:x+w]
                                roi = cv2.GaussianBlur(roi, (25, 25), 7)  # 가우시안 블러 적용
                                frame[y:y+h, x:x+w] = roi
                        except:continue

                    output.write(frame)  # 모자이크된 프레임을 출력 파일에 쓰기

                    # CPU 사용량을 제한하기 위해 0.2초 동안 슬립
                    #time.sleep(0.2)

                # 파일을 닫고 정리
                cap.release()
                output.release()
        except:continue

print("\n작업 끝.")
