from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
from dataclasses import asdict
import time

from api_clients import SemanticScholarAPI
from llm_providers import GroqExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aging Theory Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CollectRequest(BaseModel):
    queries: List[str]
    year_from: int = 2000
    year_to: int = 2025
    max_papers: int = 100

class ExtractRequest(BaseModel):
    papers: List[dict]
    llm_provider: str
    llm_api_key: Optional[str] = None

class BatchExtractRequest(BaseModel):
    """Обработка одного батча статей"""
    papers: List[dict]
    llm_provider: str
    llm_api_key: str
    batch_number: int
    total_batches: int

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Aging Theory Analyzer API</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    max-width: 800px; 
                    margin: 50px auto; 
                    padding: 20px;
                    background: #f8f9fa;
                }
                h1 { color: #2c3e50; }
                .status { color: #27ae60; font-weight: bold; font-size: 1.2em; }
                .endpoint { 
                    background: white; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .features {
                    background: #e8f4f8;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <h1> Aging Theory Analyzer API</h1>
            <p class="status"> Backend is running on Hugging Face Spaces!</p>
            
            <div class="features">
                <h3> Features:</h3>
                <ul>
                    <li> Paper collection from Semantic Scholar</li>
                    <li> LLM-powered Q1-Q9 extraction with Groq</li>
                    <li> Batch processing for large datasets (10 papers/batch)</li>
                    <li> Automatic rate limiting</li>
                </ul>
            </div>
            
            <h2> Available Endpoints:</h2>
            <div class="endpoint">
                <strong>GET</strong> <a href="/docs">/docs</a> - Interactive API documentation (Swagger UI)
            </div>
            <div class="endpoint">
                <strong>GET</strong> <a href="/api/health">/api/health</a> - Health check endpoint
            </div>
            <div class="endpoint">
                <strong>POST</strong> /api/collect - Collect papers from Semantic Scholar
            </div>
            <div class="endpoint">
                <strong>POST</strong> /api/extract/batch - Extract Q1-Q9 from one batch (10 papers)
            </div>
            <div class="endpoint">
                <strong>POST</strong> /api/extract - Legacy: Extract all papers at once (not recommended for >50 papers)
            </div>
            
            <h2>🚀 Quick Start:</h2>
            <ol>
                <li>Copy this URL: <code id="url"></code></li>
                <li>Paste in frontend "Backend URL" field</li>
                <li>Get free Groq API key: <a href="https://console.groq.com/keys" target="_blank">console.groq.com/keys</a></li>
                <li>Add queries and start analyzing!</li>
            </ol>
            
            <p><small>Version 1.0.0 | Powered by FastAPI + Groq LLM (Llama 3.1 70B) + Semantic Scholar API</small></p>
            
            <script>
                document.getElementById('url').textContent = window.location.origin;
            </script>
        </body>
    </html>
    """

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "message": "Aging Theory Analyzer Backend is running",
        "features": {
            "paper_collection": "Semantic Scholar API",
            "llm_extraction": "Groq (Llama 3.1 70B)",
            "batch_processing": "10 papers per batch"
        }
    }

@app.post("/api/collect")
async def collect_papers(request: CollectRequest):
    """Collect papers from Semantic Scholar"""
    logger.info(f"Collection: {len(request.queries)} queries, {request.year_from}-{request.year_to}")
    
    all_papers = []
    papers_per_query = request.max_papers // len(request.queries) if request.queries else request.max_papers
    
    api = SemanticScholarAPI()
    
    for query in request.queries:
        try:
            papers = await api.search_papers(query, request.year_from, request.year_to, papers_per_query)
            all_papers.extend(papers)
            logger.info(f"Collected {len(papers)} for '{query}'")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Deduplicate
    seen = set()
    unique_papers = []
    for paper in all_papers:
        key = paper.id or paper.title
        if key not in seen:
            seen.add(key)
            unique_papers.append(paper)
    
    logger.info(f"Collection complete: {len(unique_papers)} unique papers")
    
    return {
        "papers": [asdict(p) for p in unique_papers],
        "total": len(unique_papers),
        "api_calls": len(unique_papers) // 100 + 1,
        "recommended_batch_size": 10
    }

@app.post("/api/extract/batch")
async def extract_batch(request: BatchExtractRequest):
    """
    Extract Q1-Q9 from ONE batch of papers (recommended: 10 papers per batch)
    
    Frontend should call this endpoint multiple times for large datasets:
    - Split papers into batches of 10
    - Call this endpoint for each batch
    - Update progress bar after each batch
    """
    logger.info(f"Batch {request.batch_number}/{request.total_batches}: {len(request.papers)} papers with {request.llm_provider}")
    
    if not request.llm_api_key:
        raise HTTPException(status_code=400, detail="LLM API key required")
    
    extractor = GroqExtractor(request.llm_api_key)
    results = []
    failed = []
    
    for idx, paper in enumerate(request.papers, 1):
        try:
            result = await extractor.extract(paper)
            results.append(result)
            logger.info(f"Batch {request.batch_number}: [{idx}/{len(request.papers)}] ✓ {paper['title'][:50]}")
        except Exception as e:
            failed.append({
                "paper_id": paper.get('id', 'unknown'),
                "paper_title": paper.get('title', 'unknown'),
                "error": str(e)
            })
            logger.error(f"Batch {request.batch_number}: [{idx}/{len(request.papers)}] ✗ {str(e)[:100]}")
    
    return {
        "batch_number": request.batch_number,
        "total_batches": request.total_batches,
        "results": results,
        "processed": len(results),
        "failed": len(failed),
        "failed_papers": failed,
        "theories_found": sum(1 for r in results if r.get('q2') == 'Yes')
    }

@app.post("/api/extract")
async def extract_data(request: ExtractRequest):
    """
    Legacy endpoint: Extract all papers at once
    
     NOT RECOMMENDED for large datasets (>50 papers)
    Use /api/extract/batch instead for better reliability
    """
    logger.info(f"Extraction: {len(request.papers)} papers with {request.llm_provider}")
    
    if not request.llm_api_key:
        raise HTTPException(status_code=400, detail="API key required")
    
    # Warn if too many papers
    if len(request.papers) > 50:
        logger.warning(f"Large dataset detected ({len(request.papers)} papers). Consider using /api/extract/batch")
    
    extractor = GroqExtractor(request.llm_api_key)
    results = []
    failed_count = 0
    
    for idx, paper in enumerate(request.papers, 1):
        try:
            result = await extractor.extract(paper)
            results.append(result)
            logger.info(f"[{idx}/{len(request.papers)}] ✓")
        except Exception as e:
            failed_count += 1
            logger.error(f"[{idx}] Failed: {e}")
    
    return {
        "results": results,
        "total": len(results),
        "theories_found": sum(1 for r in results if r.get('q2') == 'Yes'),
        "failed": failed_count,
        "warning": "Consider using /api/extract/batch for large datasets" if len(request.papers) > 50 else None
    }
