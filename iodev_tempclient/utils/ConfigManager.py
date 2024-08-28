import yaml
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Any, Dict

class ConfigManager:
    def __init__(self, config_file: str = "configuration.yaml"):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.load_config()
        # self.start_watchdog()

    def load_config(self) -> None:
        if not os.path.exists(self.config_file):
            self.create_default_config()
        
        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def create_default_config(self) -> None:
        default_config = {
            'common': {
                'dss': {
                    'rtsp_port': 8554,
                    'prefix': 'cam',
                    'model_path': 'models/dss',
                    'model_name': 'dss'
                },
                'tr': {
                    'model_path': 'models/tr',
                    'model_name': 'tr'
                }
            },
            'log': {
                'level': 'INFO',
                'path': 'logs'
            },
            'mqtt': {
                'host': 'localhost',
                'port': 1883,
                'qos': 0,
                'keepalive': 60,
                'client_id': 'tempclient',
                'username': 'tempclient',
                'password': 'tempclient',
                'tls': False
            },
            'devices': {
                'dss': {
                    'dss0': {
                        'name': 'dss0',
                        'address': 'localhost'
                    },
                    'dss1': {
                        'name': 'dss1',
                        'address': 'localhost'
                    }
                }
            }
        }

        with open(self.config_file, 'w') as file:
            yaml.dump(default_config, file)

    def start_watchdog(self) -> None:
        event_handler = ConfigFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path=os.path.dirname(self.config_file), recursive=False)
        observer.start()

    def get_value(self, *keys: str) -> Any:
        value = self.config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        return value

class ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(self.config_manager.config_file):
            print(f"Config file changed: {event.src_path}")
            self.config_manager.load_config()

# Usage example:
if __name__ == "__main__":
    config_manager = ConfigManager()
    print(config_manager.get_value("common", "dss", "rtsp_port"))
    print(config_manager.get_value("mqtt", "host"))
    print(config_manager.get_value("devices", "dss", "dss0", "name"))