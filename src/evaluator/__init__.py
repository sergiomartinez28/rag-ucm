"""
Módulo de evaluación para RAG-UCM
Permite generar datasets de evaluación, ejecutar tests y calcular métricas
"""

from .dataset_generator import DatasetGenerator
from .rag_evaluator import RAGEvaluator
from .llm_judge import LLMJudge
from .metrics import MetricsCalculator

__all__ = [
    'DatasetGenerator',
    'RAGEvaluator', 
    'LLMJudge',
    'MetricsCalculator'
]
