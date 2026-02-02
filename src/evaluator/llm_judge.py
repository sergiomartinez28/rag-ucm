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
    
    
    def _extract_scores(self, response: str) -> Optional[Dict]:
        """Extrae puntuaciones de la respuesta del LLM - muy robusto"""
        
        scores = {}
        
        # Buscar patrones simples: "relevancia: 0.8" o "relevancia: 0.8,"
        patterns = [
            (r'relevancia\s*:\s*([0-9]*\.?[0-9]+)', 'relevancia'),
            (r'fidelidad\s*:\s*([0-9]*\.?[0-9]+)', 'fidelidad'),
            (r'precision\s*:\s*([0-9]*\.?[0-9]+)', 'precision'),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Normalizar a rango [0, 1]
                    if value > 1:
                        value = min(1.0, value / 5.0)
                    scores[key] = max(0.0, min(1.0, value))
                except ValueError:
                    logger.debug(f"Error converting {match.group(1)} to float")
                    pass
        
        # Si encontramos 3 scores, retornarlos
        if len(scores) == 3:
            return scores
        
        # Si encontramos 2+ scores, retornar lo que tengamos
        if len(scores) >= 2:
            logger.debug(f"Solo se encontraron {len(scores)}/3 scores: {scores}")
            return scores if len(scores) >= 2 else None
        
        # Si no hay patrones simple, intentar con JSON como último recurso
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except:
            pass
        
        # Si aún no tenemos nada, extraer números sueltos
        if len(scores) == 0:
            numbers = re.findall(r'([0-9]*\.?[0-9]+)', response)
            if len(numbers) >= 3:
                try:
                    return {
                        'relevancia': max(0.0, min(1.0, float(numbers[0]))),
                        'fidelidad': max(0.0, min(1.0, float(numbers[1]))),
                        'precision': max(0.0, min(1.0, float(numbers[2])))
                    }
                except ValueError:
                    pass
        
        return scores if len(scores) >= 2 else None
    
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
        scores = self._extract_scores(response)
        
        if scores is None:
            logger.warning(f"No se pudieron extraer scores para pregunta {question_id}")
            # Usar heurística simple basada en la respuesta
            is_empty_answer = len(rag_answer.strip()) < 10
            
            scores = {
                'relevancia': 0.0 if is_empty_answer else 0.3,
                'fidelidad': 0.0 if is_empty_answer else 0.3,
                'precision': 0.0 if is_empty_answer else 0.2,
                'overall_score': 0.0 if is_empty_answer else 0.3,
                'explicacion': "Error al extraer puntuaciones del LLM Juez"
            }
            logger.warning(f"Usando valores heurísticos: {scores}")
        else:
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
