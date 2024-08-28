import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QListWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the main layout
        layout = QVBoxLayout()

        # Create the button
        # Create the button
        button = QPushButton("Force ext")
        button.clicked.connect(self.start_stream)
        layout.addWidget(button)

        button = QPushButton("Start Stream")
        button.clicked.connect(self.start_stream)
        layout.addWidget(button)

        # Create the image widget
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Create the list view
        self.log_list = QListWidget()
        layout.addWidget(self.log_list)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create a timer for updating the image
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)

    def start_stream(self):
        # Start the stream
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # Update the image every 30 milliseconds

    def update_image(self):
        # Read a frame from the stream
        ret, frame = self.cap.read()

        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a QImage from the frame
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(image)

            # Scale the image to fit the label
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)

            # Set the pixmap on the label
            self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        # Stop the stream and timer when the window is closed
        self.cap.release()
        self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())