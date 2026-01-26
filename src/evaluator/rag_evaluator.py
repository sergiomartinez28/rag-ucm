"""
Ejecutor de evaluación RAG
Ejecuta el RAG con las preguntas del dataset y registra respuestas y tiempos
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from loguru import logger
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Resultado de una evaluación individual"""
    id: int
    question: str
    reference_answer: str
    rag_answer: str
    sources: List[Dict]
    question_type: str
    category: str
    source_doc: str
    chunk_id: int
    
    # Métricas de tiempo (en segundos)
    retrieval_time: float
    generation_time: float
    total_time: float
    
    # Métricas de retrieval
    correct_doc_in_top_k: bool  # ¿El doc fuente está en los resultados?
    correct_doc_rank: int  # Posición del doc fuente (0 si no está)


class RAGEvaluator:
    """
    Ejecuta el sistema RAG con las preguntas del dataset de evaluación
    y registra las respuestas junto con métricas de tiempo y retrieval
    """
    
    def __init__(self, rag_pipeline=None):
        """
        Args:
            rag_pipeline: Instancia de RAGPipeline (se crea si no se proporciona)
        """
        if rag_pipeline is None:
            logger.info("Cargando RAG Pipeline...")
            from src.pipeline import RAGPipeline
            self.rag = RAGPipeline()
        else:
            self.rag = rag_pipeline
        
        logger.success("✓ RAGEvaluator inicializado")
    
    def evaluate_question(
        self,
        question: str,
        reference_answer: str,
        expected_source: str,
        question_id: int,
        question_type: str,
        category: str,
        chunk_id: int,
        top_k: int = 5
    ) -> EvaluationResult:
        """
        Evalúa una pregunta individual
        
        Args:
            question: La pregunta a evaluar
            reference_answer: Respuesta de referencia
            expected_source: Documento fuente esperado
            question_id: ID de la pregunta
            question_type: Tipo de pregunta
            category: Categoría
            chunk_id: ID del chunk original
            top_k: Número de documentos a recuperar
        
        Returns:
            EvaluationResult con todos los datos
        """
        start_total = time.time()
        
        # Ejecutar RAG
        result = self.rag.query(
            question=question,
            top_k=top_k,
            include_verification=False
        )
        
        total_time = time.time() - start_total
        
        # Extraer tiempos del resultado si están disponibles
        retrieval_time = result.get('retrieval_time', 0)
        generation_time = result.get('generation_time', 0)
        
        # Si no hay tiempos detallados, estimar
        if retrieval_time == 0 and generation_time == 0:
            # Asumir 20% retrieval, 80% generation
            retrieval_time = total_time * 0.2
            generation_time = total_time * 0.8
        
        # Verificar si el documento correcto está en los resultados
        sources = result.get('sources', [])
        correct_doc_in_top_k = False
        correct_doc_rank = 0
        
        expected_source_lower = expected_source.lower()
        for i, source in enumerate(sources, 1):
            source_title = source.get('title', '').lower()
            if expected_source_lower in source_title or source_title in expected_source_lower:
                correct_doc_in_top_k = True
                correct_doc_rank = i
                break
        
        return EvaluationResult(
            id=question_id,
            question=question,
            reference_answer=reference_answer,
            rag_answer=result.get('answer', ''),
            sources=sources,
            question_type=question_type,
            category=category,
            source_doc=expected_source,
            chunk_id=chunk_id,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            correct_doc_in_top_k=correct_doc_in_top_k,
            correct_doc_rank=correct_doc_rank
        )
    
    def run_evaluation(
        self,
        dataset_path: Path,
        output_path: Path,
        top_k: int = 5
    ) -> List[EvaluationResult]:
        """
        Ejecuta la evaluación completa del dataset
        
        Args:
            dataset_path: Ruta al dataset JSON
            output_path: Ruta para guardar resultados
            top_k: Número de documentos a recuperar
        
        Returns:
            Lista de EvaluationResult
        """
        logger.info(f"Cargando dataset desde {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Evaluando {len(dataset)} preguntas...")
        
        results = []
        
        for qa in tqdm(dataset, desc="Evaluando"):
            result = self.evaluate_question(
                question=qa['question'],
                reference_answer=qa['reference_answer'],
                expected_source=qa['source_doc'],
                question_id=qa['id'],
                question_type=qa['question_type'],
                category=qa['category'],
                chunk_id=qa['chunk_id'],
                top_k=top_k
            )
            results.append(result)
        
        # Guardar resultados
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = [asdict(r) for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Calcular métricas básicas de retrieval
        precision_at_k = sum(1 for r in results if r.correct_doc_in_top_k) / len(results)
        
        mrr_sum = sum(1/r.correct_doc_rank for r in results if r.correct_doc_rank > 0)
        mrr = mrr_sum / len(results) if results else 0
        
        avg_time = sum(r.total_time for r in results) / len(results)
        
        logger.success(f"✓ Evaluación completada: {len(results)} preguntas")
        logger.info(f"  Precision@{top_k}: {precision_at_k:.2%}")
        logger.info(f"  MRR: {mrr:.3f}")
        logger.info(f"  Tiempo promedio: {avg_time:.2f}s")
        
        return results


if __name__ == "__main__":
    from pathlib import Path
    
    evaluator = RAGEvaluator()
    
    results = evaluator.run_evaluation(
        dataset_path=Path("data/evaluation/dataset_test.json"),
        output_path=Path("data/evaluation/results_test.json"),
        top_k=5
    )
    
    print(f"\nResultados: {len(results)} evaluaciones")
    for r in results[:3]:
        print(f"\nQ: {r.question[:80]}...")
        print(f"A: {r.rag_answer[:100]}...")
        print(f"Time: {r.total_time:.2f}s, Doc found: {r.correct_doc_in_top_k}")
