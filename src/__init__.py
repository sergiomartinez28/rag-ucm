"""
RAG-UCM: Sistema RAG para asistencia académica UCM
"""

__version__ = "0.1.0"
__author__ = "Sergio Martín"

from .pipeline import RAGPipeline
from .preprocessor import DocumentPreprocessor
from .indexer import DocumentIndexer
from .retrieval import HybridRetriever
from .generator import ResponseGenerator
from .verifier import FidelityVerifier

__all__ = [
    "RAGPipeline",
    "DocumentPreprocessor",
    "DocumentIndexer",
    "HybridRetriever",
    "ResponseGenerator",
    "FidelityVerifier",
]
