import aiohttp
import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass, asdict
import time
import re

logger = logging.getLogger(__name__)

@dataclass
class Paper:
    id: str
    title: str
    year: int
    url: str
    abstract: str
    citations: int
    theory: str
    source: str
    full_text: str = ""  # Добавлено поле для полного текста

class RateLimiter:
    def __init__(self, rate: float = 3.0):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
    
    async def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens < 1:
            sleep_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1

class SemanticScholarAPI:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.rate_limiter = RateLimiter(rate=1.67)
    
    async def search_papers(self, query: str, year_from: int, year_to: int, limit: int) -> List[Paper]:
        papers = []
        offset = 0
        
        async with aiohttp.ClientSession() as session:
            while len(papers) < limit:
                await self.rate_limiter.acquire()
                
                params = {
                    'query': query,
                    'year': f'{year_from}-{year_to}',
                    'limit': min(100, limit - len(papers)),
                    'offset': offset,
                    'fields': 'paperId,title,year,abstract,citationCount,url,openAccessPdf,externalIds'
                }
                
                try:
                    async with session.get(f'{self.BASE_URL}/paper/search', params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            batch = data.get('data', [])
                            
                            if not batch:
                                break
                            
                            for item in batch:
                                paper = Paper(
                                    id=item.get('paperId', ''),
                                    title=item.get('title', 'Untitled'),
                                    year=item.get('year', year_from),
                                    url=item.get('url', ''),
                                    abstract=item.get('abstract', ''),
                                    citations=item.get('citationCount', 0),
                                    theory=query,
                                    source='Semantic Scholar',
                                    full_text=""
                                )
                                
                                # Попытка получить полный текст из Open Access PDF
                                pdf_info = item.get('openAccessPdf')
                                if pdf_info and pdf_info.get('url'):
                                    try:
                                        full_text = await self._fetch_pdf_text(session, pdf_info['url'])
                                        if full_text:
                                            paper.full_text = full_text
                                            logger.info(f"✓ Extracted full text from PDF: {paper.title[:50]}")
                                    except Exception as e:
                                        logger.warning(f"Failed to extract PDF text: {e}")
                                
                                papers.append(paper)
                            
                            offset += len(batch)
                            logger.info(f"Collected {len(papers)} papers for '{query}'")
                            
                        elif resp.status == 429:
                            await asyncio.sleep(60)
                            continue
                        else:
                            break
                            
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    break
        
        return papers[:limit]
    
    async def _fetch_pdf_text(self, session: aiohttp.ClientSession, pdf_url: str, max_size_mb: int = 5) -> str:
        """
        Попытка извлечь текст из PDF.
        В production версии здесь должна быть библиотека для парсинга PDF.
        Сейчас - упрощенная версия через попытку получить текст напрямую.
        """
        try:
            async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content_length = resp.headers.get('Content-Length')
                    if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                        logger.warning(f"PDF too large: {int(content_length) / 1024 / 1024:.1f} MB")
                        return ""
                    
                    # Здесь должен быть PDF parser (PyPDF2, pdfplumber и т.д.)
                    # Для простоты - возвращаем пустую строку
                    # В production: использовать PyPDF2 или API для извлечения текста
                    logger.info(f"PDF found at: {pdf_url}")
                    return ""  # TODO: Implement PDF text extraction
        except Exception as e:
            logger.error(f"PDF fetch error: {e}")
        
        return ""
    
    async def enrich_paper_with_details(self, session: aiohttp.ClientSession, paper: Paper) -> Paper:
        """
        Дополнительно обогащает статью деталями, используя отдельный API запрос.
        Получает более полную информацию, включая sections, если доступны.
        """
        if not paper.id:
            return paper
        
        await self.rate_limiter.acquire()
        
        try:
            params = {
                'fields': 'title,abstract,year,url,openAccessPdf,tldr,embedding'
            }
            
            async with session.get(f'{self.BASE_URL}/paper/{paper.id}', params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Обновляем abstract если он был пустым
                    if not paper.abstract and data.get('abstract'):
                        paper.abstract = data['abstract']
                    
                    # Добавляем TLDR если есть
                    tldr = data.get('tldr')
                    if tldr and tldr.get('text'):
                        paper.full_text += f"\n\nTLDR: {tldr['text']}\n\n"
                    
                    # Если есть PDF - пытаемся получить текст
                    pdf_info = data.get('openAccessPdf')
                    if pdf_info and pdf_info.get('url') and not paper.full_text:
                        full_text = await self._fetch_pdf_text(session, pdf_info['url'])
                        if full_text:
                            paper.full_text = full_text
                    
        except Exception as e:
            logger.error(f"Paper enrichment error: {e}")
        
        return paper