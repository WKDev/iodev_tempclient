import sys
import asyncio
from datetime import datetime
import uuid
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGridLayout, QTextEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import numpy as np
from ultralytics import YOLO
from utils.ConfigManager import ConfigManager
from utils.setup_logger import setup_logger
import traceback
import logging
import time
import glob
import random
import paho.mqtt.client as mqtt
from datetime import datetime

import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import time

import cv2
import numpy as np
import os
import random
import glob

birds_paths = glob.glob(os.path.join('base_bird', '*.png'))
    
# 새 이미지 데이터 로드
birds_data = []
for b in birds_paths:
    b_data = cv2.imread(b, cv2.IMREAD_UNCHANGED)
    birds_data.append([b, b_data])
bird_file = random.choice(birds_paths)
bird_img = cv2.imread(bird_file, cv2.IMREAD_UNCHANGED)


def add_birds_to_frame(frame, num_birds=5):
    # 새 이미지 파일들을 읽어옵니다
    
    # frame의 크기를 가져옵니다
    height, width = frame.shape[:2]
    
    # 결과 프레임을 준비합니다
    result_frame = frame.copy()
    
    for _ in range(num_birds):
        # 랜덤하게 새 이미지를 선택합니다
        
        
        # 새 이미지의 크기를 랜덤하게 조정합니다
        scale = random.uniform(0.05, 0.15)
        bird_resized = cv2.resize(bird_img, None, fx=scale, fy=scale)
        
        # 새 이미지를 넣을 위치를 랜덤하게 선택합니다
        x = random.randint(0, width - bird_resized.shape[1])
        y = random.randint(0, height - bird_resized.shape[0])
        
        # 알파 채널을 분리합니다
        if bird_resized.shape[2] == 4:
            bird_rgb = bird_resized[:,:,:3]
            alpha = bird_resized[:,:,3]
        else:
            bird_rgb = bird_resized
            alpha = np.ones(bird_rgb.shape[:2], dtype=bird_rgb.dtype) * 255
        
        # RGB 채널에 대해 색상을 반전시킵니다
        bird_inverted = cv2.bitwise_not(bird_rgb)
        
        # 알파 채널을 이용하여 마스크를 만듭니다
        alpha = cv2.merge([alpha, alpha, alpha])
        alpha = alpha.astype(float) / 255
        
        # 새 이미지를 프레임에 합성합니다
        for c in range(0, 3):
            result_frame[y:y+bird_inverted.shape[0], x:x+bird_inverted.shape[1], c] = \
                (alpha[:,:,c] * bird_inverted[:,:,c] + 
                 (1 - alpha[:,:,c]) * result_frame[y:y+bird_inverted.shape[0], x:x+bird_inverted.shape[1], c])
    
    return result_frame

# def add_birds_to_frame(frame, num_birds=20):
#     # 새 이미지 경로 가져오기
    
#     # 추가할 새의 수 결정
#     if num_birds is None:
#         num_birds = random.randrange(1, min(5, len(birds_data)))
#     else:
#         num_birds = min(num_birds, len(birds_data))
    
#     # 랜덤하게 새 선택
#     selected_birds = random.sample(birds_data, num_birds)
    
#     for bird in selected_birds:
#         bird_img = bird[1]
        
#         # 새 이미지 전처리
#         bird_img = preprocess_bird(bird_img)
        
#         # 프레임에 새 추가
#         frame = place_bird(frame, bird_img)
    
#     return frame

def preprocess_bird(bird_img):
    # 랜덤으로 새 뒤집기

    bird_img = cv2.flip(bird_img, random.randrange(-1, 2))
    
    # 랜덤으로 새 회전
    angle = random.randrange(-180, 180)
    bird_img = rotate_image(bird_img, angle)
    
    # 새 크기 조정
    size = random.randrange(15, 25)
    bird_img = resize_image(bird_img, size)
    
    return bird_img

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def resize_image(image, target_size):
    height, width = image.shape[:2]
    if width > target_size:
        aspect_ratio = height / width
        new_width = target_size
        new_height = int(new_width * aspect_ratio)
        return cv2.resize(image, (new_width, new_height))
    return image

def place_bird(frame, bird_img):
    frame_height, frame_width = frame.shape[:2]
    bird_height, bird_width = bird_img.shape[:2]
    
    # 랜덤 위치 선택
    x = random.randrange(0, frame_width - bird_width)
    y = random.randrange(0, frame_height - bird_height)
    
    # 알파 채널이 있는 경우 (PNG)
    if bird_img.shape[2] == 4:
        alpha_channel = bird_img[:, :, 3] / 255.0
        rgb_channels = bird_img[:, :, :3]

        for c in range(3):
            frame[y:y+bird_height, x:x+bird_width, c] = (
                (1 - alpha_channel) * frame[y:y+bird_height, x:x+bird_width, c] +
                alpha_channel * rgb_channels[:, :, c]
            )
    else:
        # 알파 채널이 없는 경우 (JPG)
        frame[y:y+bird_height, x:x+bird_width] = bird_img
    
    return frame

class MQTTManager(QObject):
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    connection_failed = pyqtSignal()

    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.client = None
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self.run)

    def run(self):
        while True:
            try:
                self.connect()
                break
            except Exception as e:
                self.logger.error(f"MQTT Connection failed: {str(e)}")
                self.connection_failed.emit()
                time.sleep(5)  # Wait for 5 seconds before trying again

    def connect(self):
        self.client = mqtt.Client(reconnect_on_failure=True)
        if self.config["username"] and self.config["password"]:
            self.client.username_pw_set(self.config["username"], self.config["password"])
        
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        try:
            self.client.connect(self.config["host"], self.config["port"])
            self.client.loop_start()
        except Exception as e:
            self.logger.error(f"MQTT Connection error: {str(e)}")
            raise

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            self.connected.emit()
        else:
            self.logger.error(f"MQTT Connection failed with code {rc}")
            self.connection_failed.emit()

    def on_disconnect(self, client, userdata, rc):
        self.logger.warning("Disconnected from MQTT broker")
        self.disconnected.emit()
        if rc != 0:
            self.logger.error(f"Unexpected MQTT disconnection. Will auto-reconnect. RC = {rc}")

    def publish(self, topic, message):
        if self.client and self.client.is_connected():
            self.client.publish(topic, message)
        else:
            self.logger.warning("MQTT client is not connected. Message not sent.")

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.thread.quit()
        self.thread.wait()

class QtHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)

def remove_close_elements(points_prev, points, tol=10):
    if points_prev.size == 0 or points.size == 0:
        return points

    points_list = points.tolist()
    for p in points:
        for pp in points_prev:
            if abs(pp[0]-p[0]) < tol and abs(pp[1]-p[1]) < tol:
                try:
                    points_list.remove(p.tolist())
                    break
                except:
                    print(f"Error removing point: {p}, {pp}")
    return np.array(points_list)

def save_img(img_title, detection_path, max_save_count, src):
    os.makedirs(detection_path, exist_ok=True)
    saved_imgs = sorted(glob.glob(detection_path + '/*.jpg'))
    if len(saved_imgs) > max_save_count:
        for p in saved_imgs[:-max_save_count]:
            os.remove(p)
    cv2.imwrite(os.path.join(detection_path, f'{img_title}.jpg'), src)

def save_stream(stream_title, path, max_save_count):
    os.makedirs(path, exist_ok=True)
    saveds = sorted(glob.glob(path + '/*'))
    if len(saveds) > max_save_count:
        for p in saveds[:-max_save_count]:
            os.remove(p)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(os.path.join(path, stream_title), fourcc, 10.0, (640, 480))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(int, np.ndarray)

    def __init__(self, stream_id, address, logger):
        self.demo_mode = False
        self.demo_frame = 0
        self.demo_image = None
        self.demo_offset = 0

        super().__init__()
        self.stream_id = stream_id
        self.address = address
        self.running = True
        self.logger = logger

    def run(self):
        cap = cv2.VideoCapture(self.address)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        self.logger.info(f"Started video stream: {self.stream_id}")
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.demo_mode:
                    frame = add_birds_to_frame(frame)
                self.change_pixmap_signal.emit(self.stream_id, frame)
            else:
                self.logger.warning(f"Failed to read frame from stream {self.stream_id}")
                time.sleep(1)
        cap.release()
        self.logger.info(f"Stopped video stream: {self.stream_id}")

    def stop(self):
        self.running = False
        self.wait()

    def apply_demo_effect(self, frame):

        bg = np.zeros_like(frame)

        for img_path in glob.glob('assets/demo_images/*.png'):
            cv2.imread(img_path)

    






        # TODO
        pass

    def generate_random_image(self):
        # TODO 
        pass
    def set_demo_mode(self, mode):
        self.demo_mode = mode
        if mode:
            self.demo_frame = 0
            self.demo_image = self.generate_random_image()
            self.demo_offset = 0

    def stop(self):
        self.running = False
        self.wait()

class DetectionThread(QThread):
    detection_signal = pyqtSignal(int, np.ndarray, np.ndarray, dict)

    def __init__(self, model_path, config, logger):
        super().__init__()
        # check if model exists
        full_path = os.path.join(os.getcwd(),'assets', model_path)
        if not os.path.exists(full_path):
            print(os.getcwd())
            raise FileNotFoundError(f"Model file not found: {full_path}")
        else:
            print(f"Loading model from {full_path}")
            self.model = YOLO(full_path)
        self.running = True
        self.queue = asyncio.Queue()
        self.logger = logger
        self.config = config
        self.ret = [np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)]
        self.ret_prev = [np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)]
        self.bdd_start = [
            {'state': False, 'time': time.time()-10, 'det_id': '', 'max_cnt': 0},
            {'state': False, 'time': time.time()-10, 'det_id': '', 'max_cnt': 0}
        ]
        self.videos = [None, None]

    def run(self):
        self.logger.info("Started detection thread")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.process_frames())

    async def process_frames(self):
        while self.running:
            try:
                stream_id, frame = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                results = self.model(frame, verbose=True)
                
                annotated_frame = results[0].plot(line_width=1, font_size=0.5, labels=False)
                cv2.putText(annotated_frame, f"{0} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                ret = np.empty((0, 2), dtype=int)
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r_xywh = box.xywh[0].astype(int)
                        ret = np.append(ret, [r_xywh[:2]], axis=0)

                self.ret[stream_id] = remove_close_elements(self.ret_prev[stream_id], ret, tol=10)
                
                active_cnt = len(self.ret[stream_id])
                
                # BDD logic
                bdd_info = self.process_bdd_logic(stream_id, active_cnt, annotated_frame)

                self.detection_signal.emit(stream_id, annotated_frame, self.ret[stream_id], bdd_info)
                
                self.ret_prev[stream_id] = self.ret[stream_id]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in detection: {str(e)}")

    def process_bdd_logic(self, stream_id, active_cnt, annotated_frame):
        bdd_info = {'state': self.bdd_start[stream_id]['state'], 'max_cnt': self.bdd_start[stream_id]['max_cnt']}
        
        count_threshold = int(2)
        time_threshold = float(10)
        
        if (not self.bdd_start[stream_id]['state']) and active_cnt > count_threshold and time.time() - self.bdd_start[stream_id]['time'] > time_threshold:
            self.bdd_start[stream_id]['state'] = True
            self.bdd_start[stream_id]['time'] = time.time()
            self.bdd_start[stream_id]['det_id'] = str(uuid.uuid4())
            formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
            start_title = f"{"dev_id"}_{stream_id}_{formatted_time}"
            
            save_img(img_title=f'{start_title}_start',
                     detection_path=os.path.join(os.getenv('IODEV_PATH', '.'), 'detections', f'cam{stream_id}'),
                     max_save_count=100,
                     src=annotated_frame)
            
            self.videos[stream_id] = save_stream(
                stream_title=f'{start_title}.mp4',
                path=os.path.join(os.getenv('IODEV_PATH', '.'), 'streams', f'cam{stream_id}'),
                max_save_count=100
            )
            
            self.logger.debug(f"Detection started for stream {stream_id}, count: {active_cnt}")
            bdd_info['event'] = 'start'
        
        if self.bdd_start[stream_id]['state']:
            if self.videos[stream_id] and self.videos[stream_id].isOpened():
                self.videos[stream_id].write(annotated_frame)
            
            if active_cnt > self.bdd_start[stream_id]['max_cnt']:
                self.bdd_start[stream_id]['max_cnt'] = active_cnt
            
            time_elapsed = time.time() - self.bdd_start[stream_id]['time'] > 5
            sparse_detected = active_cnt < 5# int(self.config.get('ioss', 'count_threshold').split(',')[1])
            time_elapsed_long = time.time() - self.bdd_start[stream_id]['time'] > 30
            
            if (time_elapsed and sparse_detected) or time_elapsed_long:
                formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
                img_title = f"{0}_{stream_id}_{formatted_time}" # self.config.get('common', 'device_id')
                save_img(img_title=f'{img_title}_end',
                         detection_path=os.path.join(os.getenv('IODEV_PATH', '.'), 'detections', f'cam{stream_id}'),
                         max_save_count=100,
                         src=annotated_frame)
                
                self.logger.debug(f"Detection ended for stream {stream_id}, count: {active_cnt}")
                self.bdd_start[stream_id]['state'] = False
                self.bdd_start[stream_id]['time'] = time.time()
                self.bdd_start[stream_id]['max_cnt'] = 0
                if self.videos[stream_id]:
                    self.videos[stream_id].release()
                    self.videos[stream_id] = None
                bdd_info['event'] = 'end'
        
        return bdd_info

    def stop(self):
        self.running = False
        self.wait()
        self.logger.info("Stopped detection thread")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("iodev")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.config_manager = ConfigManager()
        self.logger, _ = setup_logger()

        self.video_threads = {}
        self.video_labels = {}
        self.demo_mode = False


        self.init_ui()
        self.init_streams()
        self.init_mqtt()

        model_path = self.config_manager.get_value("common", "dss", "model_path")
        self.detection_thread = DetectionThread(model_path, self.config_manager, self.logger)
        self.detection_thread.detection_signal.connect(self.update_detections)
        self.detection_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_events)
        self.timer.start(10)  # 10ms interval

    def init_ui(self):
        hbox_layout = QHBoxLayout()
        self.layout.addLayout(hbox_layout)

        for cam_id in [1, 2]:
            detected_label = QLabel(self)
            detected_label.setFixedSize(640, 480)
            hbox_layout.addWidget(detected_label)
            self.video_labels[f'cam{cam_id}'] = detected_label

        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        # ForceExt 버튼 추가
        self.force_ext_button = QPushButton("ForceExt", self)
        self.force_ext_button.clicked.connect(self.on_force_ext_clicked)
        button_layout.addWidget(self.force_ext_button)

        # Demo 버튼 추가
        self.demo_button = QPushButton("Demo", self)
        self.demo_button.clicked.connect(self.on_demo_clicked)
        button_layout.addWidget(self.demo_button)


        self.log_widget = QTextEdit(self)
        self.log_widget.setReadOnly(True)
        self.layout.addWidget(self.log_widget)

        qt_handler = QtHandler(self.log_widget)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(qt_handler)

    def init_streams(self):
        dss_devices = self.config_manager.get_value("devices", "dss")
        rtsp_port = self.config_manager.get_value("common", "dss", "rtsp_port")
        cam_prefix = self.config_manager.get_value("common", "dss", "prefix")
        
        for device_name, device_info in dss_devices.items():
            for cam_id in [1, 2]:
                address = f"rtsp://{device_info['address']}:{rtsp_port}/{cam_prefix}{cam_id}"
                thread = VideoThread(cam_id - 1, address, self.logger)
                thread.change_pixmap_signal.connect(self.update_image)
                self.video_threads[f'cam{cam_id}'] = thread
                thread.start()

    def update_image(self, stream_id, frame):
        self.detection_thread.queue.put_nowait((stream_id, frame))

    def on_demo_clicked(self):
        self.demo_mode = not self.demo_mode
        if self.demo_mode:
            self.demo_button.setText("Stop Demo")
        else:
            self.demo_button.setText("Demo")
        
        for thread in self.video_threads.values():
            thread.set_demo_mode(self.demo_mode)

    def update_detections(self, stream_id, frame, detections, bdd_info):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(image)

        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)

        for detection in detections:
            x, y = detection
            painter.drawRect(int(x) - 5, int(y) - 5, 10, 10)

        painter.end()

        stream_name = f'cam{stream_id + 1}'
        self.video_labels[stream_name].setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

        if 'event' in bdd_info:
            event_type = 'started' if bdd_info['event'] == 'start' else 'ended'
            self.logger.info(f"Detection {event_type} for stream {stream_id}. Active count: {len(detections)}, Max count: {bdd_info['max_cnt']}")

    def on_force_ext_clicked(self):
        current_timestamp = int(time.time())
        dss_devices = self.config_manager.get_value("devices", "dss")
        for device_name in dss_devices.keys():
            for cam_id in [1, 2]:
                topic = f"{device_name}_cam{cam_id}/time"
                self.mqtt_manager.publish(topic, current_timestamp)
                self.logger.info(f"Published timestamp {current_timestamp} to topic {topic}")

    def process_events(self):
        QApplication.processEvents()

    def closeEvent(self, event):
        self.logger.info("Closing application...")
        self.timer.stop()
        for thread in self.video_threads.values():
            thread.stop()
        self.detection_thread.stop()
        for thread in self.video_threads.values():
            thread.wait()
        self.detection_thread.wait()
        self.mqtt_manager.stop()
        self.logger.info("All threads stopped and MQTT disconnected. Application closed.")
        event.accept()

    def init_mqtt(self):
        mqtt_config = self.config_manager.get_value("mqtt")
        self.mqtt_manager = MQTTManager(mqtt_config, self.logger)
        self.mqtt_manager.connected.connect(self.on_mqtt_connected)
        self.mqtt_manager.disconnected.connect(self.on_mqtt_disconnected)
        self.mqtt_manager.connection_failed.connect(self.on_mqtt_connection_failed)
        self.mqtt_manager.thread.start()

    def on_mqtt_connected(self):
        self.logger.info("Successfully connected to MQTT broker")

    def on_mqtt_disconnected(self):
        self.logger.warning("Disconnected from MQTT broker")

    def on_mqtt_connection_failed(self):
        self.logger.error("Failed to connect to MQTT broker")

    def on_force_ext_clicked(self):
        current_timestamp = int(time.time())
        dss_devices = self.config_manager.get_value("devices", "dss")
        for stream_name in dss_devices.keys():
            topic = f"{stream_name}/time"
            self.mqtt_manager.publish(topic, current_timestamp)
            self.logger.info(f"Published timestamp {current_timestamp} to topic {topic}")

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Object Detection App")
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.layout = QVBoxLayout(self.central_widget)

#         self.config_manager = ConfigManager()
#         self.logger, _ = setup_logger()

#         self.video_threads = {}
#         self.video_labels = {}

#         self.init_ui()
#         self.init_streams()
#         self.init_mqtt()

#         model_path = self.config_manager.get_value("common", "dss", "model_path")
#         self.detection_thread = DetectionThread(model_path, self.config_manager, self.logger)
#         self.detection_thread.detection_signal.connect(self.update_detections)
#         self.detection_thread.start()

#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.process_events)
#         self.timer.start(10)  # 10ms interval

#     def init_ui(self):
#         grid_layout = QGridLayout()
#         self.layout.addLayout(grid_layout)

#         dss_devices = self.config_manager.get_value("devices", "dss")
#         rows = len(dss_devices)
#         cols = 2  # 각 스트림에 대해 2개의 열 (원본 및 객체 감지)

#         camera_index = 0
#         for device_name, device_info in dss_devices.items():
#             for camera in device_info['cameras']:
#                 original_label = QLabel(self)
#                 original_label.setFixedSize(640, 480)
#                 grid_layout.addWidget(original_label, camera_index, 0)

#                 detected_label = QLabel(self)
#                 detected_label.setFixedSize(640, 480)
#                 grid_layout.addWidget(detected_label, camera_index, 1)

#                 stream_name = f"{device_name}_cam{camera['id']}"
#                 self.video_labels[stream_name] = {'original': original_label, 'detected': detected_label}
#                 camera_index += 1
#         # ForceExt 버튼 추가
#         self.force_ext_button = QPushButton("ForceExt", self)
#         self.force_ext_button.clicked.connect(self.on_force_ext_clicked)
#         self.layout.addWidget(self.force_ext_button)

#         self.log_widget = QTextEdit(self)
#         self.log_widget.setReadOnly(True)
#         self.layout.addWidget(self.log_widget)

#         qt_handler = QtHandler(self.log_widget)
#         qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#         self.logger.addHandler(qt_handler)

#     def init_streams(self):
#         dss_devices = self.config_manager.get_value("devices", "dss")
#         self.logger.info(f"Initializing {len(dss_devices)} video streams")
#         for stream_id, (stream_name, stream_info) in enumerate(dss_devices.items()):
#             thread = VideoThread(stream_id, stream_info['address'], self.logger)
#             thread.change_pixmap_signal.connect(self.update_image)
#             self.video_threads[stream_name] = thread
#             thread.start()

#     def init_mqtt(self):
#         mqtt_config = self.config_manager.get_value("mqtt")
#         self.mqtt_manager = MQTTManager(mqtt_config, self.logger)
#         self.mqtt_manager.connected.connect(self.on_mqtt_connected)
#         self.mqtt_manager.disconnected.connect(self.on_mqtt_disconnected)
#         self.mqtt_manager.connection_failed.connect(self.on_mqtt_connection_failed)
#         self.mqtt_manager.thread.start()

#     def on_mqtt_connected(self):
#         self.logger.info("Successfully connected to MQTT broker")

#     def on_mqtt_disconnected(self):
#         self.logger.warning("Disconnected from MQTT broker")

#     def on_mqtt_connection_failed(self):
#         self.logger.error("Failed to connect to MQTT broker")

#     def on_force_ext_clicked(self):
#         current_timestamp = int(time.time())
#         dss_devices = self.config_manager.get_value("devices", "dss")
#         for stream_name in dss_devices.keys():
#             topic = f"{stream_name}/time"
#             self.mqtt_manager.publish(topic, current_timestamp)
#             self.logger.info(f"Published timestamp {current_timestamp} to topic {topic}")

#     def update_image(self, stream_id, frame):
#         self.detection_thread.queue.put_nowait((stream_id, frame))
        
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
#         pixmap = QPixmap.fromImage(image)
        
#         stream_name = list(self.video_labels.keys())[stream_id]
#         self.video_labels[stream_name]['original'].setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

#     def update_detections(self, stream_id, frame, detections, bdd_info):
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
#         pixmap = QPixmap.fromImage(image)

#         painter = QPainter(pixmap)
#         pen = QPen(Qt.red, 3)
#         painter.setPen(pen)

#         for detection in detections:
#             x, y = detection
#             painter.drawRect(int(x) - 5, int(y) - 5, 10, 10)

#         painter.end()

#         stream_name = list(self.video_labels.keys())[stream_id]
#         self.video_labels[stream_name]['detected'].setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

#         if 'event' in bdd_info:
#             event_type = 'started' if bdd_info['event'] == 'start' else 'ended'
#             self.logger.info(f"Detection {event_type} for stream {stream_id}. Active count: {len(detections)}, Max count: {bdd_info['max_cnt']}")

#     def on_force_ext_clicked(self):
#         current_timestamp = int(time.time())
#         dss_devices = self.config_manager.get_value("devices", "dss")
#         for stream_name in dss_devices.keys():
#             topic = f"{stream_name}/time"
#             self.mqtt_client.publish(topic, current_timestamp)
#             self.logger.info(f"Published timestamp {current_timestamp} to topic {topic}")

#     def process_events(self):
#         QApplication.processEvents()

#     def closeEvent(self, event):
#         self.logger.info("Closing application...")
#         self.timer.stop()
#         for thread in self.video_threads.values():
#             thread.stop()
#         self.detection_thread.stop()
#         for thread in self.video_threads.values():
#             thread.wait()
#         self.detection_thread.wait()
#         self.mqtt_manager.stop()
#         self.logger.info("All threads stopped and MQTT disconnected. Application closed.")
#         event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set up logging
    logger = setup_logger()[0]
    
    try:
        # Create and show the main window
        main_window = MainWindow()
        main_window.show()
        
        # Start the event loop
        exit_code = app.exec_()
        
        # Perform any necessary cleanup
        main_window.detection_thread.stop()
        for thread in main_window.video_threads.values():
            thread.stop()
        
        # Exit the application
        sys.exit(exit_code)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)