from bs4 import BeautifulSoup
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from protection import do_some_protection
from fake_useragent import UserAgent
import pandas as pd
import logging
from datetime import datetime
import random
from typing import List, Dict, Any
import os

class AsyncRealEstateScraper:
    def __init__(self, base_url: str, max_pages: int = 100, delay_range: tuple = (1, 3)):
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.results = []
        self.session = None
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scraper.log'
        )
        self.logger = logging.getLogger(__name__)

    async def init_session(self):
        """Инициализация сессии"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()
            self.session = None

    def get_headers(self) -> dict:
        """Генерация случайных заголовков"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'TE': 'Trailers',
        }

    async def fetch_page(self, url: str) -> str:
        """Получение содержимого страницы"""
        await asyncio.sleep(random.uniform(*self.delay_range))
        try:
            async with self.session.get(url, headers=self.get_headers(), timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.error(f"Ошибка при загрузке {url}: статус {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке {url}: {str(e)}")
            return None

    async def parse_apartment_page(self, url: str) -> Dict[str, Any]:
        """Парсинг страницы объявления"""
        html = await self.fetch_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'lxml')
        try:
            # Здесь добавьте специфичные селекторы для вашего сайта
            data = {
                'url': url,
                'title': self._safe_extract(soup, 'h1'),
                'price': self._safe_extract_price(soup),
                'address': self.extract_adress(soup),
                'total_area': self._safe_extract_area(soup),
                'floor': self._safe_extract_floor(soup),
                'description': self._safe_extract(soup, '.description'),
                'timestamp': datetime.now().isoformat()
            }
            return data
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге {url}: {str(e)}")
            return None

    def extract_adress(self, soup: BeautifulSoup) -> str:
        result = []
        element = soup.select('[data-name="AddressContainer"]')
        element_addres = element[0].children
        for i in element_addres:
            if hasattr(i, 'next') or i == ', ':
                if hasattr(i,'name'):
                    name = i.name
                    if name != 'span':
                        result.append(i.string)
                else:
                    result.append(i)
        return " ".join(result)
    

    def _safe_extract(self, soup: BeautifulSoup, selector: str) -> str:
        """Безопасное извлечение данных из HTML"""
        try:
            element = soup.select_one(selector)
            return element.text.strip() if element else None
        except:
            return None

    def _safe_extract_price(self, soup: BeautifulSoup) -> float:
        """Извлечение и обработка цены"""
        try:
            price_data_element = soup.select('[data-testid="price-amount"]')
            if price_data_element:
                price_element = price_data_element[0].find('span')
                if price_element:
                    price_text = price_element.text.strip()
                    # Удаление всех нечисловых символов, кроме точки
                    price = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
                    return price
            return None
        except:
            return None

    def _safe_extract_area(self, soup: BeautifulSoup) -> float:
        """Извлечение площади"""
        try:
            area_element = soup.select_one('.area-selector')
            if area_element:
                area_text = area_element.text.strip()
                area = float(''.join(c for c in area_text if c.isdigit() or c == '.'))
                return area
            return None
        except:
            return None

    def _safe_extract_floor(self, soup: BeautifulSoup) -> str:
        """Извлечение этажа"""
        try:
            floor_element = soup.select_one('.floor-selector')
            return floor_element.text.strip() if floor_element else None
        except:
            return None

    async def get_apartment_urls(self, page_url: str) -> List[str]:
        """Получение списка URL объявлений с страницы списка"""
        html = await self.fetch_page(page_url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        urls = []
        # Настройте селектор под конкретный сайт
        for link in soup.find_all(attrs={"data-name": "CardComponent"}):
            url = link.find('a')
            if url:
                url=url.get('href')
                if not url.startswith('http'):
                    url = f"https://www.cian.ru{url}"
                urls.append(url)
        return urls

    async def process_page(self, page_number: int):
        """Обработка одной страницы со списком объявлений"""
        page_url = f"{self.base_url}/page-{page_number}"
        self.logger.info(f"Обработка страницы {page_number}")
        
        apartment_urls = await self.get_apartment_urls(page_url)
        for url in apartment_urls:
            data = await self.parse_apartment_page(url)
            if data:
                self.results.append(data)

    async def save_results(self):
        """Сохранение результатов в CSV"""
        if not self.results:
            self.logger.warning("Нет данных для сохранения")
            return

        df = pd.DataFrame(self.results)
        filename = f"apartments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"Результаты сохранены в {filename}")

    async def run(self):
        """Запуск скраппера"""
        try:
            await self.init_session()
            tasks = []
            for page in range(1, self.max_pages + 1):
                tasks.append(self.process_page(page))
                if len(tasks) >= 5:  # Обрабатываем по 5 страниц одновременно
                    await asyncio.gather(*tasks)
                    tasks = []
            
            if tasks:
                await asyncio.gather(*tasks)
            
            await self.save_results()
        finally:
            await self.close_session()

async def main():
    base_url = "https://spb.cian.ru/cat.php?deal_type=sale&engine_version=2&object_type%5B0%5D=1&offer_type=flat&region=2"
    scraper = AsyncRealEstateScraper(base_url, max_pages=10)
    await scraper.run()


if __name__ == "__main__":
    do_some_protection()
    asyncio.run(main())
