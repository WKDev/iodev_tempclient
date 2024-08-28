import logging
import coloredlogs
from logging.handlers import RotatingFileHandler
from PyQt5.QtCore import QStringListModel

def setup_logger(log_file_path="application.log", max_file_size_mb=10, backup_count=5):
    """
    로거를 설정하고 반환합니다.

    Parameters:
    log_file_path (str): 로그 파일의 경로 및 이름
    max_file_size_mb (int): 로그 파일의 최대 크기 (MB 단위)
    backup_count (int): 최대 백업 파일 수

    Returns:
    logger (logging.Logger): 설정된 로거 객체
    qt_list_model (QStringListModel): PyQt 리스트 모델 (로그 출력용)
    """
    # 로거 생성
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)

    # 로그 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # RotatingFileHandler 추가
    file_handler = RotatingFileHandler(log_file_path, maxBytes=max_file_size_mb * 1024 * 1024, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # PyQt 리스트 모델 생성
    qt_list_model = QStringListModel()

    # PyQt 핸들러 추가
    qt_handler = QtListHandler(qt_list_model)
    qt_handler.setFormatter(formatter)
    logger.addHandler(qt_handler)

    # coloredlogs로 콘솔 로그에 색상 추가
    coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return logger, qt_list_model

# PyQt 리스트를 위한 핸들러
class QtListHandler(logging.Handler):
    def __init__(self, qt_list_model):
        super().__init__()
        self.qt_list_model = qt_list_model
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        self.qt_list_model.setStringList(self.logs)


if __name__ == "__main__":

    # 함수 사용 예시
    logger, qt_list_model = setup_logger()

    # 예시 로그 출력
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

# PyQt 리스트 모델 사용 예시
# QListView 또는 QComboBox에 이 모델을 설정하여 로그를 표시할 수 있습니다.
# 예를 들어: list_view.setModel(qt_list_model)