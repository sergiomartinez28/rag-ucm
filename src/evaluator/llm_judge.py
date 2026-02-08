"""
LLM Juez para evaluar calidad de respuestas RAG
Compara respuestas del RAG contra respuestas de referencia
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from loguru import logger
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from ..prompt_loader import load_prompt


@dataclass
class JudgeScore:
    """Puntuaciones del LLM Juez - 3 métricas simples"""
    id: int
    question: str
    
    # Métricas de calidad (0-1)
    relevancia: float
    fidelidad: float
    precision: float   # esta será 0 o 1
    overall_score: float

    # Agregados
    explanation: str
    judge_raw_output: str  # C1: Raw output del LLM juez para auditoría
    
    # Referencias
    question_type: str
    category: str


class LLMJudge:
    """
    LLM Juez que evalúa la calidad de las respuestas del RAG
    comparándolas con respuestas de referencia
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-medium-4k-instruct",
        device: str = "auto",
        use_4bit: bool = True
    ):
        """
        Args:
            model_name: Modelo para evaluación
            device: Dispositivo
            use_4bit: Usar cuantización 4-bit para modelos grandes
        """
        self.model_name = model_name
        
        logger.info(f"Cargando LLM Juez: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Cuantización 4-bit segura para ejecución local
        if use_4bit:
            logger.info("Cargando modelo con cuantización 4-bit (nf4, cpu offload)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_fp32_cpu_offload=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            logger.info("Cargando modelo en float16...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()
        logger.success(f"✓ LLM Juez cargado: {model_name}")

    def _generate_response(self, prompt: str, max_tokens: int = 600, use_messages: bool = True) -> str:
        """
        Genera respuesta del LLM
        
        Args:
            prompt: Texto del prompt (puede ser texto plano o ya formateado)
            max_tokens: Tokens máximos a generar
            use_messages: Si True, envuelve el prompt en messages. Si False, usa prompt directamente.
        """
        if use_messages:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Prompt ya viene formateado con apply_chat_template
            text = prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()
    
    def _is_abstention_response(self, answer: str) -> bool:
        """Detecta si la respuesta es una abstención (no responde)"""
        abstention_patterns = [
            "no dispongo de información",
            "no dispongo de la información",
            "no tengo información",
            "no encuentro información",
            "no hay información",
            "no puedo responder",
            "no es posible responder",
            "información no disponible",
            "sin información disponible",
            "no se especifica",
            "no se menciona",
            "no aparece en",
            "no está disponible"
        ]
        answer_lower = answer.lower().strip()
        return any(pattern in answer_lower for pattern in abstention_patterns)
    
    def _is_suspicious_short_answer(self, answer: str) -> bool:
        """Detecta respuestas sospechosamente cortas (posible error de parsing)"""
        clean = answer.strip()
        # Respuestas de 1-2 caracteres o solo números/símbolos
        if len(clean) <= 2:
            return True
        # Respuestas que son solo un número
        if clean.isdigit():
            return True
        return False
    
    def _deterministic_fallback_scores(
        self, 
        rag_answer: str, 
        reference_answer: str
    ) -> Dict:
        """
        Fallback determinista cuando el LLM juez falla.
        Penaliza fuertemente abstención y respuestas sospechosas.
        """
        is_abstention = self._is_abstention_response(rag_answer)
        is_suspicious = self._is_suspicious_short_answer(rag_answer)
        has_reference = len(reference_answer.strip()) > 10
        
        if is_abstention and has_reference:
            # El RAG no respondió pero había información → penalizar duramente
            return {
                'relevancia': 0.1,
                'fidelidad': 0.3,  # Al menos no inventó
                'precision': 0.0,
                'explanation': "Abstención: el RAG no respondió pero la referencia tiene información"
            }
        elif is_suspicious:
            # Respuesta sospechosa (ej: "1", muy corta)
            return {
                'relevancia': 0.2,
                'fidelidad': 0.2,
                'precision': 0.1,
                'explanation': f"Respuesta sospechosamente corta: '{rag_answer[:20]}'"
            }
        else:
            # Fallback genérico
            return {
                'relevancia': 0.3,
                'fidelidad': 0.3,
                'precision': 0.2,
                'explanation': "Error al extraer puntuaciones del LLM Juez - usando fallback"
            }
    
    def _extract_scores(self, response: str) -> Optional[Dict]:
        """
        Extrae puntuaciones de la respuesta del LLM.
        Prioriza JSON, luego patrones de texto.
        NO normaliza valores >1 (asume que el LLM sigue el prompt 0-1).
        """
        scores = {}
        
        # PRIORIDAD 1: Intentar extraer JSON primero (formato preferido)
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                parsed = json.loads(json_str)
                # Validar que tiene las 3 métricas y son números en rango
                for key in ['relevancia', 'fidelidad', 'precision']:
                    if key in parsed:
                        value = float(parsed[key])
                        # Si el valor es >1, probablemente está en escala 1-5
                        # pero en vez de normalizar, lo rechazamos para forzar retry
                        if value > 1.0:
                            logger.warning(f"Score {key}={value} fuera de rango 0-1, rechazando")
                            break
                        scores[key] = max(0.0, min(1.0, value))
                if len(scores) == 3:
                    return scores
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"No se pudo parsear JSON: {e}")
        
        # PRIORIDAD 2: Patrones de texto "key: value"
        patterns = [
            (r'relevancia\s*[:\=]\s*([0-9]*\.?[0-9]+)', 'relevancia'),
            (r'fidelidad\s*[:\=]\s*([0-9]*\.?[0-9]+)', 'fidelidad'),
            (r'precision\s*[:\=]\s*([0-9]*\.?[0-9]+)', 'precision'),
            (r'precisión\s*[:\=]\s*([0-9]*\.?[0-9]+)', 'precision'),  # con tilde
        ]
        
        scores = {}
        for pattern, key in patterns:
            if key in scores:  # Ya lo tenemos
                continue
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Rechazar valores fuera de rango (no normalizar)
                    if value > 1.0:
                        logger.warning(f"Score {key}={value} fuera de rango 0-1")
                        continue
                    scores[key] = max(0.0, min(1.0, value))
                except ValueError:
                    pass
        
        if len(scores) >= 2:
            return scores
        
        # No intentar extraer números sueltos - muy propenso a errores
        return None
    
    def evaluate_answer(
        self,
        question: str,
        reference_answer: str,
        rag_answer: str,
        question_id: int,
        question_type: str,
        category: str,
        sources: List[Dict]
    ) -> JudgeScore:
        """
        Evalúa una respuesta del RAG
        
        Args:
            question: Pregunta original
            reference_answer: Respuesta de referencia (gold standard)
            rag_answer: Respuesta generada por el RAG
            question_id: ID de la pregunta
            question_type: Tipo de pregunta
            category: Categoría,
            sources: List[Dict]
        Returns:
            JudgeScore con todas las métricas
        """
        retrieved_context = "\n\n".join(
            [f"[{s['id']}] {s['title']}\n{s['text_preview']}" for s in sources]
        )
                
        # Cargar prompts separados: system (instrucciones) y user (datos a evaluar)
        system_prompt = load_prompt("judge_evaluation")
        user_prompt = load_prompt(
            "judge_user_prompt",
            question=question,
            reference_answer=reference_answer,
            rag_answer=rag_answer,
            retrieved_context=retrieved_context
        )
        
        # Construir messages con roles separados
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generar texto completo para el modelo
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = self._generate_response(text, use_messages=False)
        logger.debug(f"LLM Juez response (Q#{question_id}): {response[:200]}...")
        
        # C1: Guardar raw response para auditoría
        raw_output = response
        
        scores = self._extract_scores(response)
        
        # RETRY: Si no se pudieron extraer scores, reintentar con prompt más estricto
        if scores is None:
            logger.warning(f"Retry: scores inválidos para pregunta {question_id}")
            retry_prompt = (
                f"RESPONDE SOLO JSON. Evalúa esta respuesta RAG:\n"
                f"Pregunta: {question[:200]}\n"
                f"Referencia: {reference_answer[:200]}\n"
                f"RAG: {rag_answer[:200]}\n\n"
                f"JSON obligatorio: {{\"relevancia\": X.X, \"fidelidad\": X.X, \"precision\": X.X}}\n"
                f"Todos los valores deben ser entre 0.0 y 1.0."
            )
            retry_response = self._generate_response(retry_prompt, use_messages=True, max_tokens=100)
            scores = self._extract_scores(retry_response)
        
        # FALLBACK DETERMINISTA: Si aún falla, usar heurísticas
        if scores is None:
            logger.warning(f"Fallback determinista para pregunta {question_id}")
            scores = self._deterministic_fallback_scores(rag_answer, reference_answer)
        else:
            # Aplicar penalización por abstención incluso si el juez no la detectó
            if self._is_abstention_response(rag_answer) and len(reference_answer.strip()) > 10:
                logger.debug(f"Penalizando abstención en pregunta {question_id}")
                scores['relevancia'] = min(scores.get('relevancia', 0), 0.2)
                scores['precision'] = 0.0
                scores['explanation'] = "Penalizado: abstención cuando hay referencia"
            
            # Penalizar respuestas sospechosamente cortas
            if self._is_suspicious_short_answer(rag_answer):
                logger.debug(f"Penalizando respuesta corta en pregunta {question_id}: '{rag_answer}'")
                scores['precision'] = min(scores.get('precision', 0), 0.3)
                scores['relevancia'] = min(scores.get('relevancia', 0), 0.3)
        
        # Calcular overall_score
        if 'overall_score' not in scores:
            metrics = ['relevancia', 'fidelidad', 'precision']
            avg = sum(scores.get(m, 0.0) for m in metrics) / 3
            scores['overall_score'] = avg
        
        return JudgeScore(
            id=question_id,
            question=question,
            relevancia=scores.get('relevancia', 0.0),
            fidelidad=scores.get('fidelidad', 0.0),
            precision=scores.get('precision', 0.0),
            overall_score=scores.get('overall_score', 0.0),
            explanation=scores.get('explicacion', scores.get('explanation', '')),
            judge_raw_output=raw_output,
            question_type=question_type,
            category=category
        )
    
    def evaluate_all(
        self,
        results_path: Path,
        output_path: Path,
        max_workers: int = 3
    ) -> List[JudgeScore]:
        """
        Evalúa todos los resultados del RAG
        
        Args:
            results_path: Ruta a los resultados de evaluación
            output_path: Ruta para guardar métricas
            max_workers: Número de workers paralelos (default: 3)
        
        Returns:
            Lista de JudgeScore
        """
        logger.info(f"Cargando resultados desde {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if isinstance(results, dict):
            results_list = [results]
        else:
            results_list = list(results)

        logger.info(f"Evaluando {len(results_list)} respuestas con LLM Juez usando {max_workers} workers...")
        
        scores = []
        
        # Función para procesar un item
        def process_item(item):
            return self.evaluate_answer(
                question=item['question'],
                reference_answer=item.get('reference_answer', ''),
                rag_answer=item['rag_answer'],
                question_id=item['id'],
                question_type=item['question_type'],
                category=item['category'],
                sources=item['sources']
            )
        
        # Procesamiento paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): item['id'] for item in results_list}
            
            with tqdm(total=len(results_list), desc="Evaluando con LLM Juez") as pbar:
                for future in as_completed(futures):
                    try:
                        score = future.result()
                        scores.append(score)
                    except Exception as e:
                        logger.error(f"Error evaluando pregunta {futures[future]}: {e}")
                    finally:
                        pbar.update(1)
        
        # Ordenar por ID para mantener consistencia
        scores.sort(key=lambda x: x.id)
        
        # Guardar métricas
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scores_data = [asdict(s) for s in scores]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, ensure_ascii=False, indent=2)
        
        # Calcular estadísticas
        if scores:
            avg_relevancia = sum(s.relevancia for s in scores) / len(scores)
            avg_fidelidad = sum(s.fidelidad for s in scores) / len(scores)
            avg_precision = sum(s.precision for s in scores) / len(scores)
            avg_overall = sum(s.overall_score for s in scores) / len(scores)
        else:
            avg_relevancia = 0.0
            avg_fidelidad = 0.0
            avg_precision = 0.0
            avg_overall = 0.0
        
        logger.success("? Evaluaci?n con LLM Juez completada")
        logger.info(f"  Relevancia:   {avg_relevancia:.2f}/1")
        logger.info(f"  Fidelidad:    {avg_fidelidad:.2f}/1")
        logger.info(f"  Precision:    {avg_precision:.2f}/1")
        logger.info(f"  Overall:      {avg_overall:.2f}/1")
        
        return scores


if __name__ == "__main__":
    judge = LLMJudge()
    
    judge_scores = judge.evaluate_all(
        results_path=Path("data/evaluation/results_test.json"),
        output_path=Path("data/evaluation/metrics_test.json")
    )
    
    print(f"\nMétricas: {len(judge_scores)} evaluaciones")
