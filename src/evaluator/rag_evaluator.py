"""
Ejecutor de evaluación RAG
Ejecuta el RAG con las preguntas del dataset y registra respuestas y tiempos
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    source_doc: str  # Nombre legible del documento
    doc_id: str      # ID real del documento en el indexador
    chunk_id: str    # ID real del chunk original (formato: "doc_id_chunk_N")
    
    # Métricas de tiempo (en segundos)
    retrieval_time: float
    generation_time: float
    total_time: float
    
    # Métricas de retrieval
    correct_doc_in_top_k: bool   # ¿El doc fuente está en los resultados?
    correct_chunk_in_top_k: bool # ¿El chunk exacto está en los resultados?
    correct_doc_rank: int        # Posición del doc fuente (0 si no está)
    correct_chunk_rank: int      # Posición del chunk exacto (0 si no está)


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
        expected_doc_id: str,
        question_id: int,
        question_type: str,
        category: str,
        chunk_id: int,
        top_k: int = 3
    ) -> EvaluationResult:
        """
        Evalúa una pregunta individual
        
        Args:
            question: La pregunta a evaluar
            reference_answer: Respuesta de referencia
            expected_source: Nombre legible del documento fuente
            expected_doc_id: ID real del documento en el indexador
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
        timing = result.get('timing', {})
        retrieval_time = timing.get('retrieval', 0)
        generation_time = timing.get('generation', 0)
        
        # Si no hay tiempos detallados, estimar
        if retrieval_time == 0 and generation_time == 0:
            # Asumir 20% retrieval, 80% generation
            retrieval_time = total_time * 0.2
            generation_time = total_time * 0.8
        
        # Verificar si el documento/chunk correcto está en los resultados
        sources = result.get('sources', [])
        correct_doc_in_top_k = False
        correct_chunk_in_top_k = False
        correct_doc_rank = 0
        correct_chunk_rank = 0
        
        # Extraer doc_id esperado (sin _chunk_X si existe)
        expected_doc_id_clean = expected_doc_id
        if '_chunk_' in expected_doc_id_clean:
            expected_doc_id_clean = expected_doc_id_clean.split('_chunk_')[0]
        
        for i, source in enumerate(sources, 1):
            # Obtener chunk_id del source (viene del RAG) - ahora es string
            source_chunk_id = source.get('chunk_id', '')
            
            # Obtener doc_id del source
            source_id = source.get('id', '')
            
            # Limpiar source_id (quitar _chunk_X)
            source_doc_id_clean = source_id
            if '_chunk_' in source_doc_id_clean:
                source_doc_id_clean = source_doc_id_clean.split('_chunk_')[0]
            
            # Verificar coincidencia exacta de documento (por doc_id)
            if source_doc_id_clean == expected_doc_id_clean and correct_doc_rank == 0:
                correct_doc_in_top_k = True
                correct_doc_rank = i
            
            # Verificar coincidencia exacta de chunk (por chunk_id string)
            if source_chunk_id and source_chunk_id == chunk_id and correct_chunk_rank == 0:
                correct_chunk_in_top_k = True
                correct_chunk_rank = i
        
        return EvaluationResult(
            id=question_id,
            question=question,
            reference_answer=reference_answer,
            rag_answer=result.get('answer', ''),
            sources=sources,
            question_type=question_type,
            category=category,
            source_doc=expected_source,
            doc_id=expected_doc_id,
            chunk_id=chunk_id,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            correct_doc_in_top_k=correct_doc_in_top_k,
            correct_chunk_in_top_k=correct_chunk_in_top_k,
            correct_doc_rank=correct_doc_rank,
            correct_chunk_rank=correct_chunk_rank
        )
    
    def run_evaluation(
        self,
        dataset_path: Path,
        output_path: Path,
        top_k: int = 3,
        max_workers: int = 2
    ) -> List[EvaluationResult]:
        """
        Ejecuta la evaluación completa del dataset
        
        Args:
            dataset_path: Ruta al dataset JSON
            output_path: Ruta para guardar resultados
            top_k: Número de documentos a recuperar
            max_workers: Número de workers paralelos (default: 2, cuidado con memoria GPU)
        
        Returns:
            Lista de EvaluationResult
        """
        logger.info(f"Cargando dataset desde {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Evaluando {len(dataset)} preguntas con {max_workers} workers...")
        
        results = []
        
        # Función para procesar una pregunta
        def process_question(qa):
            return self.evaluate_question(
                question=qa['question'],
                reference_answer=qa['reference_answer'],
                expected_source=qa['source_doc'],
                expected_doc_id=qa.get('doc_id', qa['source_doc']),  # Fallback a source_doc si no hay doc_id
                question_id=qa['id'],
                question_type=qa['question_type'],
                category=qa['category'],
                chunk_id=qa['chunk_id'],
                top_k=top_k
            )
        
        # Procesamiento paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_question, qa): qa['id'] for qa in dataset}
            
            with tqdm(total=len(dataset), desc="Evaluando") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluando pregunta {futures[future]}: {e}")
                    finally:
                        pbar.update(1)
        
        # Ordenar por ID para mantener consistencia
        results.sort(key=lambda x: x.id)
        
        # Guardar resultados
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = [asdict(r) for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Calcular métricas básicas de retrieval
        if results:
            precision_at_k_doc = sum(1 for r in results if r.correct_doc_in_top_k) / len(results)
            precision_at_k_chunk = sum(1 for r in results if r.correct_chunk_in_top_k) / len(results)
            mrr_sum = sum(1 / r.correct_doc_rank for r in results if r.correct_doc_rank > 0)
            mrr = mrr_sum / len(results)
            avg_time = sum(r.total_time for r in results) / len(results)
        else:
            precision_at_k_doc = 0
            precision_at_k_chunk = 0
            mrr = 0
            avg_time = 0
        
        logger.success(f"✓ Evaluación completada: {len(results)} preguntas")
        logger.info(f"  Precision@{top_k} (documento): {precision_at_k_doc:.2%}")
        logger.info(f"  Precision@{top_k} (chunk exacto): {precision_at_k_chunk:.2%}")
        logger.info(f"  MRR: {mrr:.3f}")
        logger.info(f"  Tiempo promedio: {avg_time:.2f}s")
        
        return results

