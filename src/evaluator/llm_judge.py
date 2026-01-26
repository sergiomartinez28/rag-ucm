"""
LLM Juez para evaluar calidad de respuestas RAG
Compara respuestas del RAG contra respuestas de referencia
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    overall_score: float
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

    def _generate_response(self, prompt: str, max_tokens: int = 400) -> str:
        """Genera respuesta del LLM"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()
    
    
    def _extract_scores(self, response: str) -> Optional[Dict]:
        """Extrae puntuaciones del JSON de respuesta"""
        try:
            # Buscar JSON
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback: extraer números con regex
        scores = {}
        patterns = [
            (r'relevancia["\s:]+([\d]+(?:\.\d+)?)', 'relevancia'),
            (r'fidelidad["\s:]+([\d]+(?:\.\d+)?)', 'fidelidad'),
            (r'precision["\s:]+([\d]+(?:\.\d+)?)', 'precision'),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, response.lower())
            if match:
                scores[key] = float(match.group(1))

        return scores if len(scores) >= 3 else None
    
    def evaluate_answer(
        self,
        question: str,
        rag_answer: str,
        question_id: int,
        question_type: str,
        category: str,
        sources: List[Dict],
        precision: float
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
                
        # Cargar prompt desde archivo externo
        prompt = load_prompt(
            "judge_evaluation",
            question=question,
            rag_answer=rag_answer,
            retrieved_context=retrieved_context
        )

        response = self._generate_response(prompt)
        scores = self._extract_scores(response)
        
        if scores is None:
            logger.warning(f"No se pudieron extraer scores para pregunta {question_id}")
            scores = {
                'relevancia': 0.0,
                'fidelidad': 0.0,
                'precision': 0.0,
                'overall_score': 0.0,
                'explicacion': "Error al extraer puntuaciones"
            }
        
        # Calcular overall si no está
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
        output_path: Path
    ) -> List[JudgeScore]:
        """
        Evalúa todos los resultados del RAG
        
        Args:
            results_path: Ruta a los resultados de evaluación
            output_path: Ruta para guardar métricas
        
        Returns:
            Lista de JudgeScore
        """
        logger.info(f"Cargando resultados desde {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Evaluando {len(results)} respuestas con LLM Juez...")
        
        scores = []
    
        precision = 1.0 if results.get("correct_doc_in_top_k", False) else 0.0

        score = self.evaluate_answer(
            question=results['question'],
            rag_answer=results['rag_answer'],
            question_id=results['id'],
            question_type=results['question_type'],
            category=results['category'],
            sources=results['sources'],
            precision=precision      # 👈 SE INYECTA AQUÍ
        )
        
        scores.append(score)

        
        # Guardar métricas
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scores_data = [asdict(s) for s in scores]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, ensure_ascii=False, indent=2)
        
        # Calcular estadísticas
        avg_relevancia = sum(s.relevancia for s in scores) / len(scores)
        avg_fidelidad = sum(s.fidelidad for s in scores) / len(scores)
        avg_precision = sum(s.precision for s in scores) / len(scores)
        avg_overall = sum(s.overall_score for s in scores) / len(scores)
        
        logger.success("✓ Evaluación con LLM Juez completada")
        logger.info(f"  Relevancia:   {avg_relevancia:.2f}/5")
        logger.info(f"  Fidelidad:    {avg_fidelidad:.2f}/5")
        logger.info(f"  Precisión:    {avg_precision:.2f}/5")
        logger.info(f"  Overall:      {avg_overall:.2f}/5")
        
        return scores


if __name__ == "__main__":
    judge = LLMJudge()
    
    judge_scores = judge.evaluate_all(
        results_path=Path("data/evaluation/results_test.json"),
        output_path=Path("data/evaluation/metrics_test.json")
    )
    
    print(f"\nMétricas: {len(judge_scores)} evaluaciones")
