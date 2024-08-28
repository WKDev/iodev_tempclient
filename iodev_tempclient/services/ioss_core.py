import sys
import asyncio
from datetime import datetime
import uuid
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGridLayout, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import numpy as np
from ultralytics import YOLO
from ConfigManager import ConfigManager
from setup_logger import setup_logger

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
                self.change_pixmap_signal.emit(self.stream_id, frame)
            else:
                self.logger.warning(f"Failed to read frame from stream {self.stream_id}")
                time.sleep(1)
        cap.release()
        self.logger.info(f"Stopped video stream: {self.stream_id}")

    def stop(self):
        self.running = False
        self.wait()

class DetectionThread(QThread):
    detection_signal = pyqtSignal(int, np.ndarray, np.ndarray, dict)

    def __init__(self, model_path, config, logger):
        super().__init__()
        self.model = YOLO(model_path)
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
                results = self.model(frame, verbose=self.config.getboolean('common', 'yolo_verbose'))
                
                annotated_frame = results[0].plot(line_width=1, font_size=0.5, labels=False)
                cv2.putText(annotated_frame, f"{self.config.get('common','device_id')} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        
        count_threshold = int(self.config.get('ioss', 'count_threshold').split(',')[0])
        time_threshold = float(self.config.get('ioss', 'time_threshold', fallback='10'))
        
        if (not self.bdd_start[stream_id]['state']) and active_cnt > count_threshold and time.time() - self.bdd_start[stream_id]['time'] > time_threshold:
            self.bdd_start[stream_id]['state'] = True
            self.bdd_start[stream_id]['time'] = time.time()
            self.bdd_start[stream_id]['det_id'] = str(uuid.uuid4())
            formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
            start_title = f"{self.config.get('common', 'device_id')}_{stream_id}_{formatted_time}"
            
            save_img(img_title=f'{start_title}_start',
                     detection_path=os.path.join(os.getenv('IODEV_PATH', '.'), 'detections', f'cam{stream_id}'),
                     max_save_count=self.config.getint('common', 'max_save_count'),
                     src=annotated_frame)
            
            self.videos[stream_id] = save_stream(
                stream_title=f'{start_title}.mp4',
                path=os.path.join(os.getenv('IODEV_PATH', '.'), 'streams', f'cam{stream_id}'),
                max_save_count=self.config.getint('common', 'max_save_count')
            )
            
            self.logger.debug(f"Detection started for stream {stream_id}, count: {active_cnt}")
            bdd_info['event'] = 'start'
        
        if self.bdd_start[stream_id]['state']:
            if self.videos[stream_id] and self.videos[stream_id].isOpened():
                self.videos[stream_id].write(annotated_frame)
            
            if active_cnt > self.bdd_start[stream_id]['max_cnt']:
                self.bdd_start[stream_id]['max_cnt'] = active_cnt
            
            time_elapsed = time.time() - self.bdd_start[stream_id]['time'] > 5
            sparse_detected = active_cnt < int(self.config.get('ioss', 'count_threshold').split(',')[1])
            time_elapsed_long = time.time() - self.bdd_start[stream_id]['time'] > 30
            
            if (time_elapsed and sparse_detected) or time_elapsed_long:
                formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
                img_title = f"{self.config.get('common', 'device_id')}_{stream_id}_{formatted_time}"
                save_img(img_title=f'{img_title}_end',
                         detection_path=os.path.join(os.getenv('IODEV_PATH', '.'), 'detections', f'cam{stream_id}'),
                         max_save_count=self.config.getint('common', 'max_save_count'),
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
        self.setWindowTitle("Object Detection App")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.config_manager = ConfigManager()
        self.logger, _ = setup_logger()

        self.video_threads = {}
        self.video_labels = {}

        self.init_ui()
        self.init_streams()

        model_path = self.config_manager.get_value("common", "dss", "model_path")
        self.detection_thread = DetectionThread(model_path, self.config_manager, self.logger)
        self.detection_thread.detection_signal.connect(self.update_detections)
        self.detection_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_events)
        self.timer.start(10)  # 10ms interval

    def init_ui(self):
        grid_layout = QGridLayout()
        self.layout.addLayout(grid_layout)

        dss_devices = self.config_manager.get_value("devices", "dss")
        rows = int((len(dss_devices) + 1) ** 0.5)
        cols = (len(dss_devices) + rows - 1) // rows

        for idx, (stream_name, stream_info) in enumerate(dss_devices.items()):
            label = QLabel(self)
            label.setFixedSize(640, 480)
            grid_layout.addWidget(label, idx // cols, idx % cols)
            self.video_labels[stream_name] = label

        self.log_widget = QTextEdit(self)
        self.log_widget.setReadOnly(True)
        self.layout.addWidget(self.log_widget)

        qt_handler = QtHandler(self.log_widget)
        qt_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(qt_handler)

    def init_streams(self):
        dss_devices = self.config_manager.get_value("devices", "dss")
        for stream_id, (stream_name, stream_info) in enumerate(dss_devices.items()):
            thread = VideoThread(stream_id, stream_info['address'], self.logger)
            thread.change_pixmap_signal.connect(self.update_image)
            self.video_threads[stream_name] = thread
            thread.start()

    def update_image(self, stream_id, frame):
        self.detection_thread.queue.put_nowait((stream_id, frame))

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

        stream_name = list(self.video_labels.keys())[stream_id]
        self.video_labels[stream_name].setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
        
        if 'event' in bdd_info:
            event_type = 'started' if bdd_info['event'] == 'start' else 'ended'
            self.logger.info(f"Detection {event_type} for stream {stream_id}. Active count: {len(detections)}, Max count: {bdd_info['max_cnt']}")

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
        self.logger.info("All threads stopped. Application closed.")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main