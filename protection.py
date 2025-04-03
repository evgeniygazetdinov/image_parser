import socks
import socket
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
import logging
from typing import Optional, Dict, Any
import json
import os
from dataclasses import dataclass
from contextlib import contextmanager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='proxy_protection.log'
)
logger = logging.getLogger(__name__)

@dataclass
class ProxyConfig:
    """Конфигурация прокси"""
    host: str = "localhost"
    port: int = 9050
    proxy_type: int = socks.SOCKS5
    verify_url: str = "http://check.torproject.org"
    timeout: int = 10

class ProxyProtection:
    """Класс для управления прокси-защитой"""
    
    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()
        self.original_socket = socket.socket
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def proxy_session(self):
        """Контекстный менеджер для использования прокси"""
        try:
            self.enable_proxy()
            yield
        finally:
            self.disable_proxy()
    
    def enable_proxy(self) -> None:
        """Включение прокси"""
        try:
            socks.set_default_proxy(
                self.config.proxy_type,
                self.config.host,
                self.config.port
            )
            socket.socket = socks.socksocket
            logger.info("Прокси успешно активирован")
        except Exception as e:
            logger.error(f"Ошибка при активации прокси: {str(e)}")
            raise
    
    def disable_proxy(self) -> None:
        """Отключение прокси и восстановление исходного сокета"""
        socket.socket = self.original_socket
        logger.info("Прокси отключен")
    
    def get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запроса"""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
    
    def verify_proxy_connection(self) -> bool:
        """Проверка работы прокси"""
        try:
            req = Request(
                self.config.verify_url,
                headers=self.get_headers(),
            )
            with urlopen(req, timeout=self.config.timeout) as response:
                html = response.read()
                soup = BeautifulSoup(html, "html.parser")
                title = soup("title")[0].get_text()
                
                if "Tor" in title:
                    logger.info("Прокси-соединение успешно проверено")
                    return True
                else:
                    logger.warning("Прокси-соединение работает, но возможно это не Tor")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при проверке прокси: {str(e)}")
            return False
    
    def save_config(self, filename: str = "proxy_config.json") -> None:
        """Сохранение конфигурации в файл"""
        try:
            config_dict = {
                "host": self.config.host,
                "port": self.config.port,
                "proxy_type": self.config.proxy_type,
                "verify_url": self.config.verify_url,
                "timeout": self.config.timeout
            }
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=4)
            logger.info(f"Конфигурация сохранена в {filename}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")
    
    @classmethod
    def load_config(cls, filename: str = "proxy_config.json") -> 'ProxyProtection':
        """Загрузка конфигурации из файла"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            config = ProxyConfig(**config_dict)
            return cls(config)
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            return cls()

def setup_protection(config_file: Optional[str] = None) -> ProxyProtection:
    """Функция для настройки защиты"""
    if config_file and os.path.exists(config_file):
        protection = ProxyProtection.load_config(config_file)
    else:
        protection = ProxyProtection()
    
    return protection
def do_some_protection():
    protection = setup_protection()
    
    # Использование через контекстный менеджер
    with protection.proxy_session():
        if protection.verify_proxy_connection():
            print("Прокси работает корректно")
        else:
            print("Проблема с прокси-соединением")
    
    protection.save_config()