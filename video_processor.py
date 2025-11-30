import cv2
import supervision as sv
from ultralytics import YOLO
from database import Database
from pose_analyzer import PoseAnalyzer
import text_detection
import socket
import random


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model_pose = YOLO('yolov8n-pose.pt')
        self.model_main = YOLO('yolo11n.pt')
        self.tracker = sv.ByteTrack()
        self.db = Database()
        self.pose_analyzer = PoseAnalyzer()
        self.prev_keypoints = None
        self.similar_boxes_count = 0
        self.last_box = None
        self.resize_flag = False
        self.xcut = 0

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Обработка видео: {total_frames} кадров, {fps} FPS")

        frame_count = 0
        self.person_id_counter = 0
        self.train_id_counter = 0
        self.track_history = {}

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 5000))
        server_socket.listen(1)

        print("Ожидание подключения клиента...")
        conn, addr = server_socket.accept()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            if frame_count % 10 == 0: # анализ каждого N кадра
                if self.resize_flag == True:
                    if self.xcut > 320: # объект преимущественно справа
                        person_frame = frame[:, : -self.xcut]
                        train_frame = frame[:, self.xcut :]

                    else: # объект преимущественно слева
                        person_frame = frame[:, self.xcut :]
                        train_frame = frame[:, : -self.xcut]

                    # обработка части кадра с работниками
                    self.main_analyzer(person_frame, [0])
                    if random.randrange(10) == 0:   #обработка части кадра с поездом
                        self.main_analyzer(train_frame, [0, 6])
                else:
                    # Начальная стадия (обработка полного кадра)
                    self.main_analyzer(frame, [0, 6])

            
            # Отправка дэшборду обработанного кадра
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            if result:
                frame_data = encoded_frame.tobytes()
                frame_size = len(frame_data)
                try:
                    conn.sendall(frame_size.to_bytes(4, byteorder='big'))
                    conn.sendall(frame_data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    print("Клиент отключился")
                    break

        cap.release()
        cv2.destroyAllWindows()

    def main_analyzer(self, frame, classes):
        results = self.model_main.track(frame, classes=classes, conf=0.4, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if int(box.cls) == 0:  # класс "person"
                        self.person_analyze(frame)
                    if int(box.cls) == 6:  # класс "train"
                        self.train_analyze(frame, box)

    def person_analyze(self, frame):
        # Детекция позы
        results = self.model_pose.track(frame, persist=True, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            for i, keypoints in enumerate(results[0].keypoints.data):
                keypoints_np = keypoints.cpu().numpy()

                if keypoints_np.size == 0:
                    continue

                action, confidence = self.pose_analyzer.simple_action_detection(
                    keypoints_np, self.prev_keypoints
                )

                if len(results[0].boxes) > i:
                    bbox = results[0].boxes[i].xyxy[0].cpu().numpy().tolist()

                    if i not in self.track_history:
                        self.person_id_counter += 1
                        self.track_history[i] = self.person_id_counter

                    person_id = self.track_history[i]

                    self.db.save_person_event(person_id, action, confidence, bbox)
                    self.draw_detection(frame, bbox, action, person_id)

            if len(results[0].keypoints.data) > 0:
                self.prev_keypoints = results[0].keypoints.data[0].cpu().numpy()

    def train_analyze(self, frame, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # отсечение ложного срабатывания на весь фрейм
        if ((x2 - x1 >= 300)):
            return
        
        current_box = (x1, y1, x2, y2)
        if self.is_similar_box(current_box):
            self.similar_boxes_count += 1
        else:
            self.similar_boxes_count = 1
            self.last_box = current_box

        self.draw_train_box(frame, box.xyxy[0].cpu().numpy().tolist())
        
        # Если набралось 10 похожих боксов, уменьшаем вероятность поиска человека в кадре
        if self.similar_boxes_count >= 10:
            train_frame = frame[int((y2-y1)/2):y2, x1:x2]
            train_num = text_detection.detect_text(train_frame)
            
            self.resize_flag = True
            self.xcut = x1 + int((x2-x1)/2) # середина кадра объекта
            if train_num == None:
                train_num = "[Undefined]"
            
            self.db.save_train_event(1, train_num)
            self.similar_boxes_count = 0

    def is_similar_box(self, current_box, threshold=0.8):
        if not hasattr(self, 'last_box') or self.last_box is None:
            return False
        
        x1_curr, y1_curr, x2_curr, y2_curr = current_box
        x1_prev, y1_prev, x2_prev, y2_prev = self.last_box
        
        area_curr = (x2_curr - x1_curr) * (y2_curr - y1_curr)
        area_prev = (x2_prev - x1_prev) * (y2_prev - y1_prev)
        
        x_left = max(x1_curr, x1_prev)
        y_top = max(y1_curr, y1_prev)
        x_right = min(x2_curr, x2_prev)
        y_bottom = min(y2_curr, y2_prev)
        
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        union_area = area_curr + area_prev - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou >= threshold

    def draw_detection(self, frame, bbox, action, person_id):
        x1, y1, x2, y2 = map(int, bbox)

        colors = {
            'standing': (0, 255, 0),
            'walking': (255, 255, 0),
            'bending': (0, 255, 255),
            'sitting': (255, 0, 0),
            'unknown': (128, 128, 128)
        }

        color = colors.get(action, (128, 128, 128))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{person_id} {action}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_train_box(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.putText(frame, f"train", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)