"""
Módulo de verificación de fidelidad
Detecta posibles alucinaciones en la respuesta generada
"""

from typing import Any, Dict, List, Optional, Tuple
from transformers import pipeline
import re
from loguru import logger

from .preprocessor import Chunk
from .utils import timed, TimingContext


class FidelityVerifier:
    """
    Verifica que las afirmaciones en la respuesta generada
    estén respaldadas por los fragmentos recuperados
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: float = 0.7
    ):
        """
        Args:
            model_name: Modelo para NLI (Natural Language Inference)
                       Si es None, usa heurísticas simples
            threshold: Umbral de confianza para considerar una afirmación verificada
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold debe estar entre 0 y 1")
        
        self.threshold = threshold
        self.model_name = model_name
        
        if model_name:
            with TimingContext("Carga de modelo NLI"):
                logger.info(f"Cargando modelo NLI: {model_name}")
                self.nli_pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    device=-1
                )
                logger.success("✓ Modelo NLI cargado")
        else:
            self.nli_pipeline = None
            logger.info("✓ Verificador inicializado con heurísticas")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Divide el texto en oraciones"""
        if not text or not text.strip():
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @timed
    def verify_with_heuristics(
        self,
        answer: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict[str, Any]:
        """Verificación usando heurísticas simples"""
        if not answer or not answer.strip():
            logger.warning("Respuesta vacía para verificar")
            return {
                'fidelity_score': 0.0,
                'is_faithful': False,
                'verifications': [],
                'unsupported_sentences': [],
                'num_sentences': 0,
                'num_unsupported': 0
            }
        
        if not contexts:
            logger.warning("No hay contextos para verificar")
            return {
                'fidelity_score': 0.0,
                'is_faithful': False,
                'verifications': [],
                'unsupported_sentences': [],
                'num_sentences': 0,
                'num_unsupported': 0
            }
        
        sentences = self.split_into_sentences(answer)
        context_texts = [chunk.text.lower() for chunk, _ in contexts]
        combined_context = " ".join(context_texts)
        
        verifications = []
        unsupported_sentences = []
        
        for sentence in sentences:
            words = sentence.lower().split()
            keywords = [
                w for w in words 
                if len(w) > 4 and w not in [
                    'según', 'normativa', 'documento', 'establece',
                    'indica', 'señala', 'especifica', 'determina'
                ]
            ]
            
            if not keywords:
                continue
            
            matches = sum(1 for kw in keywords if kw in combined_context)
            coverage = matches / len(keywords) if keywords else 0
            is_supported = coverage >= 0.5
            
            verifications.append({
                'sentence': sentence,
                'is_supported': is_supported,
                'coverage': coverage,
                'method': 'heuristic'
            })
            
            if not is_supported:
                unsupported_sentences.append(sentence)
        
        if verifications:
            fidelity_score = sum(
                1 for v in verifications if v['is_supported']
            ) / len(verifications)
        else:
            fidelity_score = 1.0
        
        return {
            'fidelity_score': fidelity_score,
            'is_faithful': fidelity_score >= self.threshold,
            'verifications': verifications,
            'unsupported_sentences': unsupported_sentences,
            'num_sentences': len(sentences),
            'num_unsupported': len(unsupported_sentences)
        }
    
    @timed
    def verify(
        self,
        answer: str,
        contexts: List[Tuple[Chunk, float]],
        method: str = "auto"
    ) -> Dict[str, Any]:
        """Verifica la fidelidad de una respuesta"""
        logger.info("Verificando fidelidad de la respuesta...")
        
        if method == "auto":
            method = "heuristic"  # Por ahora solo heurísticas
        
        result = self.verify_with_heuristics(answer, contexts)
        
        if not result['is_faithful']:
            result['warning'] = (
                f"⚠️ Atención: {result['num_unsupported']} de {result['num_sentences']} "
                f"afirmaciones no están claramente respaldadas por la normativa. "
                f"Te recomendamos verificar esta información con secretaría."
            )
        else:
            result['warning'] = None
        
        logger.info(
            f"✓ Verificación completa: fidelidad={result['fidelity_score']:.2f}, "
            f"método={method}"
        )
        
        return result
    
    def add_citation_check(
        self,
        answer: str,
        num_sources: int
    ) -> Dict[str, Any]:
        """Verifica que la respuesta incluya citas a las fuentes"""
        if not answer or not answer.strip():
            return {
                'has_citations': False,
                'num_citations': 0,
                'num_unique_citations': 0,
                'citation_coverage': 0.0,
                'missing_citations': True
            }
        
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(int(c) for c in citations)
        
        has_citations = len(unique_citations) > 0
        citation_coverage = len(unique_citations) / num_sources if num_sources > 0 else 0
        
        return {
            'has_citations': has_citations,
            'num_citations': len(citations),
            'num_unique_citations': len(unique_citations),
            'citation_coverage': citation_coverage,
            'missing_citations': citation_coverage < 0.5
        }


if __name__ == "__main__":
    verifier = FidelityVerifier(threshold=0.7)
    print("✓ Módulo verifier listo para usar")
