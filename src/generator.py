"""
Módulo de generación de respuestas
Usa LLM local para generar respuestas con citas
"""

import re
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from loguru import logger

from .preprocessor import Chunk
from .prompt_loader import load_prompt
from .utils import timed, TimingContext, torch_memory_cleanup


class ResponseGenerator:
    """
    Genera respuestas usando un LLM local open source
    - Llama-3.2-3B-Instruct
    - Phi-4-mini
    - Qwen2.5-3B-Instruct
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 768,
        temperature: float = 0.3,
        device: str = "auto"
    ):
        """
        Args:
            model_name: Modelo de HuggingFace
            max_new_tokens: Tokens máximos a generar
            temperature: Temperatura de sampling (0.0-1.0)
            device: 'cuda', 'cpu', o 'auto'
        """
        if not 0 <= temperature <= 2.0:
            raise ValueError("temperature debe estar entre 0.0 y 2.0")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        logger.info(f"Cargando modelo LLM: {model_name}")
        
        # Determinar dispositivo
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Usando dispositivo: {device}")
        
        # Cargar tokenizer y modelo con timing
        with TimingContext("Carga de tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        with TimingContext("Carga de modelo LLM"):
            if device == "cuda":
                logger.info("Cargando modelo en GPU con float16...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                ).to(device)
            else:
                logger.info("Cargando modelo en CPU con cuantización 4-bit...")
                try:
                    # Intentar cuantización 4-bit con bitsandbytes (solo en CPU)
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    logger.success("✓ Modelo cuantizado a 4-bit (2-4x más rápido en CPU)")
                except Exception as e:
                    logger.warning(f"No se pudo cargar con cuantización 4-bit: {e}")
                    logger.info("Cargando modelo sin cuantización (más lento)...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    ).to(device)
        
        # Poner en modo evaluación
        self.model.eval()
        
        logger.success(f"✓ Modelo cargado en {device}")
    
    def _tokenize_spanish(self, text: str) -> list[str]:
        """
        Tokenizador robusto para español (reutilizado de BM25)
        Captura palabras + números sin stopwords innecesarias
        """
        stopwords_es = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no',
            'por', 'con', 'su', 'para', 'es', 'al', 'lo', 'como', 'o', 'más',
            'este', 'ese', 'si', 'ya', 'muy', 'sin', 'me', 'hay', 'sobre'
        }
        
        # Regex: palabras españolas (con acentos) + números
        pattern = r'[a-záéíóúüñü]+|\d+'
        tokens = re.findall(pattern, text.lower())
        
        # Filtrar stopwords pero mantener números y palabras importantes
        return [t for t in tokens if t not in stopwords_es or t.isdigit()]
    
    def _is_factual_query(self, query: str) -> bool:
        """Detecta si la query busca datos factuales (números, fechas, plazos)"""
        query_lower = query.lower()
        factual_patterns = [
            'cuántos', 'cuántas', 'cuánto', 'cuánta',
            'plazo', 'días', 'meses', 'años',
            'créditos', 'artículo', 'art.',
            'porcentaje', '%', 'número', 'cantidad',
            'fecha', 'cuando', 'quién', 'dónde'
        ]
        return any(pattern in query_lower for pattern in factual_patterns)
    
    def _extract_top_sentences(
        self,
        text: str,
        query: str,
        max_sentences: int = 5,
        max_chars: int = 1200
    ) -> str:
        """
        Extrae las mejores oraciones de un texto basándose en overlap con la query
        Mejoras:
        - Tokenización robusta (español + números)
        - Orden O(n) con índices guardados
        - Heurística factual (prioriza números y términos clave)
        
        Args:
            text: Texto del chunk
            query: Pregunta del usuario
            max_sentences: Número máximo de oraciones a retornar
            max_chars: Límite máximo de caracteres
        
        Returns:
            Extracto con las oraciones más relevantes
        """
        # Partir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return text[:max_chars]
        
        # Tokenizar query de forma robusta
        query_tokens = set(self._tokenize_spanish(query))
        is_factual = self._is_factual_query(query)
        
        # Puntuar oraciones con índice original guardado (O(n))
        scored_sentences = []
        for idx, sent in enumerate(sentences):
            if not sent.strip():
                continue
            
            sent_tokens = set(self._tokenize_spanish(sent))
            
            # Score base: overlap de tokens
            overlap = len(query_tokens & sent_tokens)
            score = overlap / (len(sent_tokens) + 1)
            
            # Boost para preguntas factuales
            if is_factual:
                # +0.3 si contiene números
                if any(c.isdigit() for c in sent):
                    score += 0.3
                
                # +0.2 si contiene palabras clave de normativa
                key_words = {'deberá', 'será', 'plazo', 'máximo', 'mínimo', 
                            'artículo', 'apartado', 'créditos', 'días', 'años'}
                if any(word in sent.lower() for word in key_words):
                    score += 0.2
            
            scored_sentences.append((idx, sent.strip(), score))
        
        # Ordenar por score y tomar top
        top_scored = sorted(scored_sentences, key=lambda x: x[2], reverse=True)[:max_sentences]
        
        # Para preguntas factuales: añadir contexto (1 frase antes/después)
        if is_factual and len(top_scored) > 0:
            context_indices = set()
            for idx, sent, score in top_scored:
                context_indices.add(idx)
                # Añadir frase anterior y posterior si existen
                if idx > 0:
                    context_indices.add(idx - 1)
                if idx < len(sentences) - 1:
                    context_indices.add(idx + 1)
            
            # Reconstruir con contexto en orden original
            context_sentences = [(i, sentences[i]) for i in sorted(context_indices)]
        else:
            # Orden original solo de las seleccionadas
            context_sentences = [(idx, sent) for idx, sent, score in sorted(top_scored, key=lambda x: x[0])]
        
        result = " ".join([s for _, s in context_sentences])
        
        # Límite de caracteres
        if len(result) > max_chars:
            result = result[:max_chars].rstrip() + "..."
        
        return result if result else text[:max_chars]
    
    def create_prompt(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> List[Dict[str, str]]:
        """
        Crea el prompt usando el formato de chat del tokenizer
        Separando system (instrucciones) y user (pregunta + docs)
        
        Args:
            query: Pregunta del usuario
            contexts: Lista de (Chunk, score) recuperados
        
        Returns:
            Lista de messages para apply_chat_template
        """
        if not contexts:
            raise ValueError("Se requieren contextos para generar el prompt")
        
        # Construir contextos con referencias (usando top-sentences)
        context_texts = []
        for i, (chunk, score) in enumerate(contexts, 1):
            metadata = chunk.metadata
            title = metadata.get('title', metadata.get('filename', 'Doc'))
            # Extraer top-sentences relevantes en lugar de [:400] fijo
            text = self._extract_top_sentences(
                chunk.text,
                query,
                max_sentences=5,
                max_chars=1200
            ).strip()
            context_texts.append(f"[{i}] {title}: {text}")
        
        contexts_str = "\n\n".join(context_texts)
        
        # Cargar prompts separados: system (instrucciones) y user (pregunta + docs)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_prompt", contexts=contexts_str, query=query)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def _is_suspicious_response(self, answer: str) -> bool:
        """
        Detecta respuestas sospechosas que indican un error de generación.
        
        Returns:
            True si la respuesta es sospechosa y debe reintentarse
        """
        clean = answer.strip()
        
        # Respuesta vacía o muy corta (1-2 caracteres)
        if len(clean) <= 2:
            return True
        
        # Solo un número (posible error de parsing)
        if clean.isdigit():
            return True
        
        # Respuestas que son solo "1", "1.", "1:" etc
        if re.match(r'^1\.?\s*:?\s*$', clean):
            return True
        
        # Respuestas que empiezan con numeración seguida de casi nada
        if re.match(r'^[1-9]\.\s*.{0,5}$', clean):
            return True
        
        return False
    
    def _generate_answer_internal(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]],
        temperature: float = None,
        do_sample: bool = None
    ) -> Tuple[str, 'TimingContext']:
        """
        Lógica interna de generación de respuesta.
        
        Returns:
            Tuple de (respuesta, timer)
        """
        # Crear mensajes para el chat
        messages = self.create_prompt(query, contexts)
        
        # Usar apply_chat_template para formato correcto
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenizar con presupuesto de contexto aumentado
        model_max_length = getattr(self.tokenizer, 'model_max_length', 4096)
        max_input_length = min(model_max_length, 4096)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=False
        ).to(self.device)
        
        logger.debug(f"Tokenización: {inputs['input_ids'].shape[1]} tokens (máx: {max_input_length})")
        
        # Obtener longitud del prompt para extraer respuesta
        prompt_length = inputs['input_ids'].shape[1]
        
        # Usar eos_token_id como pad_token_id
        eos_id = self.tokenizer.eos_token_id
        
        # Parámetros de generación
        use_temp = temperature if temperature is not None else self.temperature
        use_sample = do_sample if do_sample is not None else (use_temp > 0)
        
        logger.debug(f"Generando respuesta (max_new_tokens={self.max_new_tokens}, temp={use_temp})")
        
        with torch.no_grad():
            with TimingContext("Generación LLM", log=False) as timer:
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=use_sample,
                    temperature=use_temp if use_sample else None,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                    use_cache=True
                )
        
        logger.debug(f"Generación LLM: {timer.elapsed:.2f}s")
        
        # Decodificar solo los tokens nuevos generados
        generated_tokens = outputs[0][prompt_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer, timer
    
    def _generate_answer_with_retry(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Tuple[str, 'TimingContext']:
        """
        Reintenta generación con parámetros más conservadores.
        Si aún falla, intenta extracción directa para preguntas factuales.
        """
        # Retry 1: Sin sampling (determinístico)
        answer, timer = self._generate_answer_internal(
            query, contexts, temperature=0, do_sample=False
        )
        
        if not self._is_suspicious_response(answer):
            logger.info("Retry exitoso con temperature=0")
            return answer, timer
        
        # Retry 2: Modo extractivo para preguntas factuales
        if self._is_factual_query(query):
            logger.info("Intentando extracción directa para query factual")
            extracted = self._extract_factual_answer(query, contexts)
            if extracted:
                return extracted, timer
        
        # Si todo falla, devolver lo que tenemos
        logger.warning(f"Fallback final: '{answer[:50]}'")
        return answer, timer
    
    def _extract_factual_answer(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> str:
        """
        Extrae respuesta directamente del contexto para preguntas factuales.
        Busca oraciones que contengan números/fechas relevantes.
        """
        query_tokens = set(self._tokenize_spanish(query))
        
        best_sentence = None
        best_score = 0
        
        for chunk, score in contexts[:3]:  # Solo top-3 chunks
            sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:  # Muy corta
                    continue
                
                sent_tokens = set(self._tokenize_spanish(sent))
                overlap = len(query_tokens & sent_tokens)
                
                # Boost si contiene números
                has_numbers = bool(re.search(r'\d+', sent))
                if has_numbers:
                    overlap += 2
                
                # Boost si contiene palabras clave de normativa
                if any(kw in sent.lower() for kw in ['plazo', 'crédito', 'día', 'mes', 'año', 'máximo', 'mínimo']):
                    overlap += 1
                
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sent
        
        if best_sentence and best_score >= 2:
            # Limpiar y limitar longitud
            if len(best_sentence) > 300:
                best_sentence = best_sentence[:300].rsplit(' ', 1)[0] + '...'
            return best_sentence
        
        return None
    
    @timed
    def generate(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict[str, Any]:
        """
        Genera respuesta usando el LLM
        
        Args:
            query: Pregunta del usuario
            contexts: Contextos recuperados
        
        Returns:
            Dict con 'answer', 'sources', y metadata
        """
        if not query or not query.strip():
            raise ValueError("Query no puede estar vacía")
        
        if not contexts:
            logger.warning("No hay contextos para generar respuesta")
            return {
                'answer': "No encuentro información relevante en la normativa disponible. Te recomiendo contactar con secretaría de tu facultad.",
                'sources': [],
                'metadata': {'warning': 'no_contexts'}
            }
        
        logger.info("Generando respuesta...")
        
        # Generar respuesta inicial
        answer, timer = self._generate_answer_internal(query, contexts)
        
        # FALLBACK: Detectar respuestas sospechosas y reintentar
        if self._is_suspicious_response(answer):
            logger.warning(f"Respuesta sospechosa detectada: '{answer[:50]}'. Reintentando...")
            answer, timer = self._generate_answer_with_retry(query, contexts)
        
        # Preparar fuentes
        sources = self._format_sources(contexts)
        
        logger.success("✓ Respuesta generada")
        
        return {
            'answer': answer,
            'sources': sources,
            'contexts': contexts,
            'metadata': {
                'model': self.model_name,
                'temperature': self.temperature,
                'num_contexts': len(contexts),
                'generation_time': timer.elapsed
            }
        }
    
    def _format_sources(
        self,
        contexts: List[Tuple[Chunk, float]]
    ) -> List[Dict[str, Any]]:
        """
        Formatea las fuentes para mostrar al usuario
        """
        sources = []
        
        for i, (chunk, score) in enumerate(contexts, 1):
            metadata = chunk.metadata
            
            # Extraer doc_id del chunk
            doc_id = chunk.doc_id if hasattr(chunk, 'doc_id') else metadata.get('filename', 'unknown')
            
            source = {
                'id': doc_id,  # ID real del documento (para métricas precisas)
                'rank': i,     # Posición en el ranking
                'title': metadata.get('title', metadata.get('filename', 'Documento')),
                'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'score': round(score, 3),
                'chunk_id': chunk.chunk_id,  # ID del chunk (para métricas precisas)
                'metadata': {
                    'faculty': metadata.get('faculty', ''),
                    'year': metadata.get('year', ''),
                    'url': metadata.get('url', '')
                }
            }
            
            sources.append(source)
        
        return sources
    
    def cleanup(self) -> None:
        """Libera recursos de memoria"""
        logger.info("Liberando recursos del generador...")
        
        with torch_memory_cleanup():
            self.model = None
            self.tokenizer = None
        
        logger.success("✓ Recursos del generador liberados")


if __name__ == "__main__":
    print("✓ Módulo generator listo para usar")
