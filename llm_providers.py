import aiohttp
import json
import logging
from typing import Dict, Optional
import time
import asyncio


logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, rate: float = 2.0):
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


class GroqExtractor:
    RESEARCH_QUESTIONS = {
        'q1': 'Does it suggest an aging biomarker (measurable entity)?',
        'q2': 'Does it suggest a molecular mechanism of aging?',
        'q3': 'Does it suggest a longevity intervention to test?',
        'q4': 'Does it claim that aging cannot be reversed?',
        'q5': 'Does it suggest biomarker predicting maximal lifespan differences?',
        'q6': 'Does it explain why naked mole rat lives 40+ years?',
        'q7': 'Does it explain why birds live longer than mammals?',
        'q8': 'Does it explain why large animals live longer?',
        'q9': 'Does it explain why caloric restriction increases lifespan?'
    }
    
    THEORY_KEYWORDS = {
       
        'Telomere Attrition': [
            'telomere', 'telomeric', 'telomerase', 'telomere shortening',
            'chromosome end', 'hayflick limit', 'telomere length'
        ],
        'Genomic Instability': [
            'genomic instability', 'genome instability', 'dna damage',
            'mutation', 'dna repair', 'chromosomal aberration', 'double strand break'
        ],
        'Epigenetic Alterations': [
            'epigenetic', 'dna methylation', 'methylation', 'histone',
            'chromatin', 'epigenetic clock', 'epigenome', 'epigenetic drift'
        ],
        'Loss of Proteostasis': [
            'proteostasis', 'protein folding', 'protein misfolding',
            'protein aggregation', 'unfolded protein response', 'chaperone',
            'proteasome', 'amyloid'
        ],
        'Mitochondrial Dysfunction': [
            'mitochondrial', 'mitochondria', 'mitochondrial dysfunction',
            'mitochondrial dna', 'oxidative phosphorylation', 'respiratory chain',
            'atp production', 'mitophagy'
        ],
        'Cellular Senescence': [
            'cellular senescence', 'senescence', 'senescent cells', 'sasp',
            'p16', 'p21', 'senolytic', 'senomorphic', 'p16ink4a'
        ],
        'Stem Cell Exhaustion': [
            'stem cell exhaustion', 'stem cell', 'stem cell pool',
            'stem cell decline', 'progenitor', 'hematopoietic stem cell',
            'muscle stem cell', 'regenerative capacity', 'niche aging'
        ],
        'Disabled Macroautophagy': [
            'autophagy', 'macroautophagy', 'autophagic', 'autophagosome',
            'lysosomal', 'lysosomal degradation', 'autophagic flux'
        ],
        'Deregulated Nutrient Sensing': [
            'nutrient sensing', 'mtor', 'igf-1', 'insulin signaling',
            'ampk', 'sirtuin', 'foxo', 'rapamycin', 'insulin resistance'
        ],
        'Altered Intercellular Communication': [
            'altered intercellular communication', 'inflammaging',
            'paracrine signaling', 'exosome', 'extracellular vesicle'
        ],
        
       
        'Free Radical Theory': [
            'free radical theory', 'free radical', 'harman',
            'radical damage', 'oxidative theory'
        ],
        'Oxidative Stress': [
            'oxidative stress', 'reactive oxygen species', 'ros',
            'antioxidant', 'superoxide', 'oxidative damage', 'redox'
        ],
        'Mitochondrial Free Radical Theory': [
            'mitochondrial free radical', 'mitochondrial ros', 'mtros',
            'mitochondria-generated ros'
        ],
      
        'Inflammaging': [
            'inflammaging', 'chronic inflammation', 'low-grade inflammation',
            'inflammatory aging'
        ],
        'Chronic Inflammation': [
            'inflammation', 'inflammatory', 'pro-inflammatory',
            'cytokine', 'il-6', 'tnf', 'tnf-alpha', 'nf-kb'
        ],
        'Immunosenescence': [
            'immunosenescence', 'immune aging', 'immune system decline',
            'thymic involution', 't cell exhaustion'
        ],
        
        
        'Antagonistic Pleiotropy': [
            'antagonistic pleiotropy', 'pleiotropic', 'trade-off',
            'williams theory', 'beneficial early detrimental late'
        ],
        'Mutation Accumulation': [
            'mutation accumulation', 'late-acting mutation',
            'late-acting deleterious', 'medawar'
        ],
        'Disposable Soma Theory': [
            'disposable soma', 'somatic maintenance', 'energy allocation',
            'kirkwood', 'germ line vs soma'
        ],
        'Rate of Living Theory': [
            'rate of living', 'metabolic rate', 'metabolism',
            'oxygen consumption', 'caloric turnover'
        ],
        
        # === СИСТЕМНЫЕ И ПРОГРАММНЫЕ ТЕОРИИ ===
        'Programmed Aging': [
            'programmed aging', 'genetic program', 'aging program',
            'quasi-program', 'developmental program'
        ],
        'Hyperfunction Theory': [
            'hyperfunction', 'mtor hyperfunction', 'geroconversion',
            'hallmarks as hyperfunction', 'blagosklonny'
        ],
        'Reliability Theory': [
            'reliability theory', 'system failure', 'redundancy',
            'engineering approach'
        ],
        'Information Theory of Aging': [
            'information theory', 'loss of information',
            'epigenetic information', 'entropy', 'information entropy'
        ],
        
        
        'Aging Clocks and Biomarkers': [
            'aging clock', 'epigenetic clock', 'dnam clock',
            'horvath clock', 'hannum clock', 'phenotypic age',
            'biological age', 'proteomic clock', 'dunedinpace'
        ],
        'Hallmarks of Aging Framework': [
            'hallmarks of aging', 'nine hallmarks', 'aging hallmarks',
            'geroscience', 'geroscience hypothesis'
        ],
        'Compression of Morbidity': [
            'compression of morbidity', 'fries hypothesis',
            'delayed morbidity', 'shortened disability'
        ],
        
        'Endocrine Theory': [
            'endocrine theory', 'hormonal aging', 'hormone decline',
            'neuroendocrine', 'hpa axis', 'growth hormone', 'dhea'
        ],
        'Reproductive-Cell Cycle Theory': [
            'reproductive-cell cycle', 'menopause', 'andropause',
            'reproductive senescence'
        ],
        
       
        'Cognitive Reserve Theory': [
            'cognitive reserve', 'brain reserve', 'neural reserve',
            'education buffer', 'resilience to neuropathology'
        ],
        'Socioemotional Selectivity Theory': [
            'socioemotional selectivity', 'time perspective',
            'emotion regulation', 'carstensen'
        ],
        'Successful Aging Theory': [
            'successful aging', 'rowe and kahn', 'healthy aging',
            'active aging', 'productive aging'
        ],
        'Selective Optimization with Compensation': [
            'selective optimization', 'compensation', 'baltes',
            'soc model', 'lifespan development'
        ],
        'Psychosocial Stress Theory': [
            'psychosocial stress', 'chronic stress', 'allostatic load',
            'stress-mediated aging', 'cortisol', 'psychological stress'
        ],
        
     
        'Disengagement Theory': [
            'disengagement theory', 'social withdrawal',
            'reduced social interaction', 'cumming and henry'
        ],
        'Activity Theory': [
            'activity theory', 'continued engagement',
            'social participation', 'stay active'
        ],
        'Continuity Theory': [
            'continuity theory', 'preserve identity',
            'consistency of roles', 'atchley'
        ],
        'Cumulative Disadvantage Theory': [
            'cumulative disadvantage', 'cumulative inequality',
            'life course inequality', 'divergent trajectories'
        ],
        'Life Course Perspective': [
            'life course', 'life course perspective', 'trajectory',
            'early life conditions', 'linked lives', 'critical periods'
        ],
        'Age Stratification Theory': [
            'age stratification', 'age cohort', 'age-based roles',
            'social structure by age'
        ],
        'Modernization Theory': [
            'modernization theory', 'status of elderly',
            'industrialization', 'cowgill'
        ],
        
        
        'Longevity Dividend': [
            'longevity dividend', 'economic benefit',
            'macro-economic impact', 'healthcare economics'
        ],
        'Demographic Transition': [
            'demographic transition', 'population aging',
            'aging society', 'old-age dependency', 'silver tsunami'
        ],
        
        'Cross-Linking Theory': [
            'cross-linking', 'glycation', 'advanced glycation end products',
            'age', 'protein cross-linking', 'collagen cross-linking'
        ],
        'Somatic Mutation Theory': [
            'somatic mutation', 'mutation accumulation in soma',
            'somatic dna damage'
        ],
        'Error Catastrophe Theory': [
            'error catastrophe', 'error accumulation',
            'orgel hypothesis', 'protein synthesis error'
        ],
        'Neuroendocrine Theory': [
            'neuroendocrine theory', 'hypothalamus aging',
            'pituitary aging', 'hormonal cascade'
        ],
        'Membrane Theory': [
            'membrane theory', 'membrane fluidity', 'lipid peroxidation',
            'membrane aging', 'phospholipid change'
        ],
        'Waste Accumulation Theory': [
            'waste accumulation', 'lipofuscin', 'cellular debris',
            'garbage catastrophe', 'lysosomal storage'
        ],
        
   
        'General Aging': [
            'aging', 'ageing', 'senescence', 'longevity', 'lifespan',
            'age-related', 'aging process'
        ]
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate=1.0)
    
    def classify_theory(self, paper: Dict) -> str:
        """
        Автоматическая классификация статьи по теории старения
        на основе ключевых слов в тексте
        """
        # Собираем весь доступный текст
        text_parts = [
            paper.get('title', ''),
            paper.get('abstract', ''),
            paper.get('full_text', '')
        ]
        full_text = ' '.join(text_parts).lower()
        
        # Подсчитываем совпадения для каждой теории
        theory_scores = {}
        for theory, keywords in self.THEORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in full_text)
            if score > 0:
                theory_scores[theory] = score
        
        # Выбираем теорию с максимальным score
        if theory_scores:
            best_theory = max(theory_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Classified as '{best_theory}' (score: {theory_scores[best_theory]})")
            return best_theory
        
        # Если ничего не найдено, возвращаем оригинальный theory_id или General Aging
        original_theory = paper.get('theory', 'General Aging')
        logger.info(f"No specific theory keywords found, using: {original_theory}")
        return original_theory
    
    async def extract(self, paper: Dict) -> Dict:
        """Extract Q1-Q9 answers from paper using Groq LLM"""
        try:
            await self.rate_limiter.acquire()
            
            # Проверяем наличие текста
            if not paper.get('title') and not paper.get('abstract') and not paper.get('full_text'):
                logger.error(f"Paper has no text content: {paper.get('id', 'unknown')}")
                raise ValueError("Paper has no text content")
            
            prompt = self._build_prompt(paper)
            
            # Логируем длину промпта для отладки
            logger.info(f"Prompt length: {len(prompt)} chars for paper: {paper.get('title', 'unknown')[:50]}")
            
            async with aiohttp.ClientSession() as session:
                response = await self._call_groq(session, prompt)
            
            result = self._parse_response(response, paper)
            
            logger.info(f"✓ Successfully extracted Q1-Q9 for: {paper.get('title', 'unknown')[:50]}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Extraction failed for {paper.get('title', 'unknown')[:50]}: {str(e)}")
            raise
    
    def _build_prompt(self, paper: Dict) -> str:
        """Build prompt using ALL available text from paper"""
        questions_text = '\n'.join([f'{k.upper()}: {v}' for k, v in self.RESEARCH_QUESTIONS.items()])
        
        # Собираем весь доступный текст
        full_text = ""
        
        # Приоритет 1: Если есть full_text - используем его
        if paper.get('full_text') and len(paper['full_text'].strip()) > 100:
            full_text = paper['full_text']
            logger.info(f"Using full_text field ({len(full_text)} chars)")
        else:
            # Приоритет 2: Собираем из доступных полей
            text_parts = []
            
            if paper.get('title'):
                text_parts.append(f"Title: {paper['title']}")
            
            if paper.get('abstract'):
                text_parts.append(f"Abstract: {paper['abstract']}")
            
            # Дополнительные поля если есть
            for field in ['introduction', 'methods', 'results', 'discussion', 'conclusion']:
                if paper.get(field):
                    text_parts.append(f"{field.title()}: {paper[field]}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Assembled text from {len(text_parts)} fields ({len(full_text)} chars)")
        
        # Проверяем что есть хоть какой-то текст
        if len(full_text.strip()) < 50:
            logger.warning(f"Very short text ({len(full_text)} chars) - results may be poor")
            full_text = f"Title: {paper.get('title', 'Unknown')}\nYear: {paper.get('year', 'Unknown')}\nAbstract: {paper.get('abstract', 'No abstract available')}"
        
        prompt = f"""Analyze this scientific paper about aging and answer 9 research questions.

Paper Year: {paper.get('year', 'Unknown')}

FULL TEXT:
{full_text}

QUESTIONS:
{questions_text}

IMPORTANT:
- For EACH question provide: answer (Yes/No), confidence (0.0-1.0), citation (exact quote)
- For Q1, answer options: "Yes, quantitatively shown" / "Yes, but not shown" / "No"
- Be specific and base answers ONLY on the text provided
- Use exact quotes from the text for citations

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
  "q1": {{"answer": "No", "confidence": 0.80, "citation": "exact quote from text"}},
  "q2": {{"answer": "Yes", "confidence": 0.90, "citation": "exact quote"}},
  "q3": {{"answer": "No", "confidence": 0.75, "citation": "quote"}},
  "q4": {{"answer": "No", "confidence": 0.80, "citation": "quote"}},
  "q5": {{"answer": "No", "confidence": 0.70, "citation": "quote"}},
  "q6": {{"answer": "No", "confidence": 0.65, "citation": "quote"}},
  "q7": {{"answer": "No", "confidence": 0.70, "citation": "quote"}},
  "q8": {{"answer": "No", "confidence": 0.75, "citation": "quote"}},
  "q9": {{"answer": "No", "confidence": 0.80, "citation": "quote"}}
}}"""
        return prompt
    
    async def _call_groq(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """Call Groq API with error handling and retries"""
        max_retries = 2
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Groq API (attempt {attempt + 1}/{max_retries})")
                
                async with session.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}'
                    },
                    json={
                        'model': 'llama-3.3-70b-versatile',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'response_format': {'type': 'json_object'},
                        'temperature': 0.1,
                        'max_tokens': 2048
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    response_text = await resp.text()
                    
                    if resp.status == 200:
                        data = json.loads(response_text)
                        content = data['choices'][0]['message']['content']
                        logger.info(f"✓ Groq API success (response length: {len(content)} chars)")
                        return content
                    elif resp.status == 429:
                        logger.warning(f"Rate limit hit, waiting {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Groq API error {resp.status}: {response_text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        raise Exception(f"Groq API error: {resp.status} - {response_text[:200]}")
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise Exception("Groq API timeout after retries")
            except Exception as e:
                logger.error(f"Call error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise
        
        raise Exception("Max retries exceeded")
    
    def _parse_response(self, llm_response: str, paper: Dict) -> Dict:
        """Parse LLM response and build result object"""
        try:
            # Удаляем возможные markdown code blocks
            llm_response = llm_response.strip()
            if llm_response.startswith('```'):
                llm_response = llm_response.split('```')[1]
                if llm_response.startswith('json'):
                    llm_response = llm_response[4:]
                llm_response = llm_response.strip()
            
            parsed = json.loads(llm_response)
            
            # Проверяем что все вопросы присутствуют
            for i in range(1, 10):
                if f'q{i}' not in parsed:
                    logger.error(f"Missing q{i} in response")
                    raise ValueError(f"Missing q{i} in response")
            
            # Calculate average confidence
            confidences = []
            for i in range(1, 10):
                conf = parsed[f'q{i}'].get('confidence', 0)
                try:
                    confidences.append(float(conf))
                except:
                    confidences.append(0.5)
            
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.5
            
            # Validate theory relevance
            text_for_validation = paper.get('full_text', '') or (paper.get('title', '') + ' ' + paper.get('abstract', ''))
            text_for_validation = text_for_validation.lower()
            
            aging_keywords = ['aging', 'ageing', 'senescence', 'longevity', 'lifespan']
            relevance = sum(1 for kw in aging_keywords if kw in text_for_validation) / len(aging_keywords)
            validity_score = round(relevance, 2)
            
            
            classified_theory = self.classify_theory(paper)
            
            result = {
                'theory_id': classified_theory,
                'paper_name': paper.get('title', 'Unknown'),
                'paper_url': paper.get('url', ''),
                'paper_year': paper.get('year', 0),
                'validity_score': validity_score,
                'avg_confidence': avg_confidence
            }
            
            # Add Q1-Q9 answers
            for i in range(1, 10):
                q = parsed[f'q{i}']
                result[f'q{i}'] = q.get('answer', 'No')
                result[f'q{i}_confidence'] = float(q.get('confidence', 0.5))
                result[f'q{i}_citation'] = q.get('citation', '')
            
            logger.info(f"Parsed result: theory={classified_theory}, avg_confidence={avg_confidence}, validity={validity_score}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response was: {llm_response[:500]}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.error(f"Response was: {llm_response[:500]}")
            raise
