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
from .config import get_config


class ResponseGenerator:
    """
    Genera respuestas usando un LLM local open source
    Modelo por defecto: Qwen2.5-3B-Instruct
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens: int = 100,
        temperature: float = 0.1,
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
        
        # Cargar config para opciones de generación
        config = get_config()
        self.use_sentence_extraction = config.generation.use_sentence_extraction  # type: ignore
        self.max_context_chars = config.generation.max_context_chars_per_chunk  # type: ignore
        self.retry_on_abstention = config.generation.retry_on_abstention  # type: ignore
        
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
            # Siempre usar cuantización 4-bit para reducir uso de VRAM
            logger.info(f"Cargando modelo con cuantización 4-bit en {device}...")
            try:
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
                logger.success("✓ Modelo cuantizado a 4-bit (ahorro ~50% VRAM)")
            except Exception as e:
                logger.warning(f"No se pudo cargar con cuantización 4-bit: {e}")
                logger.info("Cargando modelo sin cuantización...")
                if device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    ).to(device)
                else:
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
        
        # Construir contextos con referencias
        context_texts = []
        total_context_chars = 0
        
        for i, (chunk, _score) in enumerate(contexts, 1):
            metadata = chunk.metadata
            title = metadata.get('title', metadata.get('filename', 'Doc'))
            
            # FASE 3: Usar chunk completo o recortado según config
            if self.use_sentence_extraction:
                # Método antiguo: extraer oraciones (puede perder respuesta)
                text = self._extract_top_sentences(
                    chunk.text,
                    query,
                    max_sentences=5,
                    max_chars=1200
                ).strip()
            else:
                # Método nuevo: chunk completo o recortado conservadoramente
                text = chunk.text.strip()
                if len(text) > self.max_context_chars:
                    # Recorte conservador: mantener texto contiguo
                    text = text[:self.max_context_chars].rsplit(' ', 1)[0] + "..."
            
            context_texts.append(f"[{i}] {title}: {text}")
            total_context_chars += len(text)
        
        contexts_str = "\n\n".join(context_texts)
        
        # Log diagnóstico de longitud de contexto
        logger.debug(
            f"Contexto para LLM: {len(contexts)} chunks, "
            f"{total_context_chars} chars total, "
            f"extraction={'sentences' if self.use_sentence_extraction else 'full'}"
        )
        
        # Cargar prompts separados: system (instrucciones) y user (pregunta + docs)
        system_prompt = load_prompt("system_prompt")
        user_prompt = load_prompt("user_prompt", contexts=contexts_str, query=query)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def _is_abstention_response(self, answer: str) -> bool:
        """
        Detecta si la respuesta es una abstención ("No dispongo de información...").
        
        Returns:
            True si es una abstención explícita
        """
        clean = answer.strip().lower()
        
        abstention_patterns = [
            'no dispongo',
            'no encuentro información',
            'no tengo información',
            'no hay información',
            'no se menciona',
            'no aparece',
            'no está especificado',
            'la normativa no especifica',
            'los fragmentos no contienen',
            'no puedo proporcionar',
            'no es posible determinar'
        ]
        
        return any(pattern in clean for pattern in abstention_patterns)
    
    def _context_has_answer_signals(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detecta si el contexto tiene señales de que contiene la respuesta.
        Útil para decidir si reintentar cuando el modelo abstiene.
        
        Returns:
            (tiene_señales, diagnostico_dict)
        """
        query_tokens = set(self._tokenize_spanish(query))
        
        # Keywords de normativa que indican respuesta factual
        factual_keywords = {
            'días', 'meses', 'años', 'plazo', 'créditos', 'ects',
            'máximo', 'mínimo', 'porcentaje', '%', 'euros', 'fecha',
            'deberá', 'podrá', 'será', 'tendrá', 'estará',
            'artículo', 'apartado', 'capítulo'
        }
        
        diagnostics = {
            'has_numbers': False,
            'has_factual_keywords': False,
            'high_overlap': False,
            'overlap_score': 0.0,
            'context_chars': 0
        }
        
        total_overlap = 0
        total_chars = 0
        
        for chunk, score in contexts[:3]:  # Top-3 chunks
            text = chunk.text.lower()
            total_chars += len(chunk.text)
            
            # Chequear números
            if re.search(r'\d+', text):
                diagnostics['has_numbers'] = True
            
            # Chequear keywords factuales
            if any(kw in text for kw in factual_keywords):
                diagnostics['has_factual_keywords'] = True
            
            # Calcular overlap
            chunk_tokens = set(self._tokenize_spanish(text))
            overlap = len(query_tokens & chunk_tokens)
            total_overlap += overlap
        
        diagnostics['context_chars'] = total_chars
        diagnostics['overlap_score'] = total_overlap / (len(query_tokens) + 1)
        diagnostics['high_overlap'] = diagnostics['overlap_score'] >= 2.0
        
        # Tiene señales si: (números + keywords) o (alto overlap + keywords)
        has_signals = (
            (diagnostics['has_numbers'] and diagnostics['has_factual_keywords']) or
            (diagnostics['high_overlap'] and diagnostics['has_factual_keywords'])
        )
        
        return has_signals, diagnostics
    
    def _is_suspicious_response(
        self,
        answer: str,
        query: str = None,
        contexts: List[Tuple[Chunk, float]] = None
    ) -> bool:
        """
        Detecta respuestas sospechosas que indican un error de generación.
        Ahora incluye abstenciones cuando hay señales de respuesta en el contexto.
        
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
        
        # NUEVO: Abstención cuando hay señales de respuesta en el contexto
        if self.retry_on_abstention and self._is_abstention_response(answer):
            if query and contexts:
                has_signals, diag = self._context_has_answer_signals(query, contexts)
                if has_signals:
                    logger.warning(
                        f"Abstención sospechosa detectada. "
                        f"Context signals: numbers={diag['has_numbers']}, "
                        f"keywords={diag['has_factual_keywords']}, "
                        f"overlap={diag['overlap_score']:.2f}"
                    )
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
    
    def _generate_with_focused_prompt(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Tuple[str, 'TimingContext']:
        """
        Genera respuesta con prompt más directo y enfocado.
        Usado en retry cuando el prompt normal falla.
        """
        chunk, score = contexts[0]
        
        # Prompt ultra-directo
        focused_system = """Eres un asistente que responde preguntas sobre normativa universitaria.
        
REGLA CRÍTICA: Si el documento contiene información relacionada con la pregunta, responde con esa información específica. 
NO digas "No dispongo" si hay algún dato relevante en el texto.

Responde de forma concisa y directa."""
        
        focused_user = f"""Documento:
{chunk.text}

Pregunta: {query}

Responde SOLO con la información que responda a la pregunta. Si hay nombres, fechas, números o datos específicos, menciónalos."""
        
        messages = [
            {"role": "system", "content": focused_system},
            {"role": "user", "content": focused_user}
        ]
        
        # Generar con temperatura 0 (determinístico)
        with TimingContext("Generación LLM focused", log=False) as timer:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            prompt_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=None,  # greedy
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated_tokens = outputs[0][prompt_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        logger.debug(f"Generación LLM focused: {timer.elapsed:.2f}s")
        return answer, timer
    
    def _extract_factual_answer(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> str:
        """
        Extrae respuesta directamente del contexto para preguntas factuales.
        Prioriza oraciones que contengan términos específicos de la pregunta.
        """
        # Extraer palabras clave IMPORTANTES de la pregunta (no stopwords)
        stopwords = {'qué', 'cuál', 'cuáles', 'cuándo', 'cuánto', 'cuántos', 'cuánta', 'cuántas',
                     'cómo', 'dónde', 'quién', 'quiénes', 'por', 'para', 'que', 'cual', 'como',
                     'cuando', 'donde', 'quien', 'los', 'las', 'del', 'de', 'la', 'el', 'un', 'una',
                     'unos', 'unas', 'es', 'son', 'fue', 'fueron', 'ser', 'estar', 'ha', 'han',
                     'debe', 'puede', 'se', 'en', 'con', 'sin', 'sobre', 'entre', 'tras', 'según',
                     'hay', 'al', 'a', 'y', 'o', 'pero', 'si', 'no', 'más', 'menos', 'muy', 'poco'}
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-záéíóúñü]{3,}\b', query_lower))
        query_keywords = query_words - stopwords
        
        # Detectar tipo de pregunta para priorizar
        is_when_question = any(w in query_lower for w in ['cuándo', 'cuando', 'fecha', 'día', 'mes'])
        is_how_many_question = any(w in query_lower for w in ['cuántos', 'cuántas', 'cuánto', 'número', 'cantidad'])
        
        best_sentence = None
        best_score = 0
        best_chunk_sentences = []
        best_sentence_idx = -1
        
        for chunk, chunk_score in contexts[:3]:  # Solo top-3 chunks
            # Mejor segmentación: considerar también saltos de línea como separadores
            text = chunk.text.replace('\n', ' ').replace('  ', ' ')
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                
                sent_lower = sent.lower()
                sent_words = set(re.findall(r'\b[a-záéíóúñü]{3,}\b', sent_lower))
                
                # CRÍTICO: Contar palabras clave de la PREGUNTA que aparecen en la oración
                keyword_overlap = len(query_keywords & sent_words)
                
                # Score base: overlap de keywords específicas (más importante)
                score = keyword_overlap * 3
                
                # Boost específico según tipo de pregunta
                if is_when_question:
                    # Para preguntas "cuándo": priorizar oraciones con fechas
                    if re.search(r'\d+\s*de\s*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)', sent_lower):
                        score += 8  # Muy alto boost para fechas completas
                    elif re.search(r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b', sent_lower):
                        score += 5
                    if re.search(r'pasado|último|anterior|próximo', sent_lower):
                        score += 2
                
                if is_how_many_question:
                    # Para preguntas "cuántos": priorizar oraciones con números
                    if re.search(r'\d+', sent):
                        score += 4
                
                # Boost menor genérico si contiene números
                has_numbers = bool(re.search(r'\d+', sent))
                if has_numbers and not is_when_question and not is_how_many_question:
                    score += 1
                
                # Penalizar oraciones que son claramente de otro tema
                if keyword_overlap == 0:
                    score = 0  # No considerar si no tiene ninguna keyword de la pregunta
                
                if score > best_score:
                    best_score = score
                    best_sentence = sent
                    best_sentence_idx = idx
                    best_chunk_sentences = sentences
        
        if best_sentence and best_score >= 4:
            # Para preguntas temporales/numéricas: solo la oración con la respuesta
            # Para otras: incluir contexto
            if is_when_question or is_how_many_question:
                # Solo la oración principal para respuestas factuales precisas
                result = best_sentence
            else:
                # Incluir contexto: frase anterior y posterior
                result_sentences = []
                
                # Frase anterior (si existe y es sustancial)
                if best_sentence_idx > 0:
                    prev = best_chunk_sentences[best_sentence_idx - 1].strip()
                    if len(prev) > 30:
                        result_sentences.append(prev)
                
                # Frase principal
                result_sentences.append(best_sentence)
                
                # Frase posterior (si existe y es sustancial)
                if best_sentence_idx < len(best_chunk_sentences) - 1:
                    next_sent = best_chunk_sentences[best_sentence_idx + 1].strip()
                    if len(next_sent) > 30:
                        result_sentences.append(next_sent)
                
                result = " ".join(result_sentences)
            
            # Limpiar y limitar longitud
            if len(result) > 500:
                result = result[:500].rsplit(' ', 1)[0] + '...'
            
            logger.debug(f"Extracción factual: score={best_score}, keywords_matched={len(query_keywords)}, len={len(result)}")
            return result
        
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
        
        # Generar respuesta inicial con todos los contextos
        answer, timer = self._generate_answer_internal(query, contexts)
        
        # FALLBACK: Si abstiene, retry con solo el mejor chunk (menos ruido)
        if self._is_suspicious_response(answer, query=query, contexts=contexts):
            logger.warning(f"Respuesta sospechosa: '{answer[:50]}'. Retry con mejor chunk...")
            
            # Retry con solo el chunk de mayor score (menos confusión para el LLM)
            best_chunk = contexts[:1]  # Solo el #1
            logger.info(f"Retry con 1 chunk (score={best_chunk[0][1]:.3f}) vs {len(contexts)} anteriores")
            
            # Usar prompt más directo para el retry
            answer_retry, timer = self._generate_with_focused_prompt(query, best_chunk)
            
            # Si el retry con menos contexto funciona, usarlo
            if not self._is_abstention_response(answer_retry) and len(answer_retry.strip()) > 10:
                logger.success(f"✓ Retry exitoso con chunk único: '{answer_retry[:60]}...'")
                answer = answer_retry
            else:
                # Si sigue absteniéndose, intentar extracción
                logger.info("Retry falló, intentando extracción...")
                is_abstention = self._is_abstention_response(answer)
                if self._is_factual_query(query) or is_abstention:
                    extracted = self._extract_factual_answer(query, contexts)
                    if extracted:
                        logger.success(f"✓ Extracción exitosa: '{extracted[:60]}...'")
                        answer = extracted
                    else:
                        logger.warning("Extracción falló, usando respuesta original")
        
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
