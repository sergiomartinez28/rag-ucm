"""
Módulo de verificación de fidelidad
Detecta posibles alucinaciones en la respuesta generada
"""

from typing import List, Dict, Tuple, Optional
from transformers import pipeline
import re
from loguru import logger

from .preprocessor import Chunk


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
        self.threshold = threshold
        self.model_name = model_name
        
        if model_name:
            logger.info(f"Cargando modelo NLI: {model_name}")
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=-1  # CPU por defecto
            )
            logger.success("✓ Modelo NLI cargado")
        else:
            self.nli_pipeline = None
            logger.info("✓ Verificador inicializado con heurísticas")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divide el texto en oraciones
        """
        # Simple split por puntos, signos de interrogación y exclamación
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def verify_with_heuristics(
        self,
        answer: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict[str, any]:
        """
        Verificación usando heurísticas simples
        Busca si las palabras clave de cada oración aparecen en los contextos
        """
        sentences = self.split_into_sentences(answer)
        context_texts = [chunk.text.lower() for chunk, _ in contexts]
        combined_context = " ".join(context_texts)
        
        verifications = []
        unsupported_sentences = []
        
        for sentence in sentences:
            # Extraer palabras clave (palabras de >4 letras, excluyendo comunes)
            words = sentence.lower().split()
            keywords = [
                w for w in words 
                if len(w) > 4 and w not in [
                    'según', 'normativa', 'documento', 'establece',
                    'indica', 'señala', 'especifica', 'determina'
                ]
            ]
            
            # Verificar si las keywords aparecen en el contexto
            if not keywords:
                continue
            
            matches = sum(1 for kw in keywords if kw in combined_context)
            coverage = matches / len(keywords) if keywords else 0
            
            is_supported = coverage >= 0.5  # Al menos 50% de keywords presentes
            
            verifications.append({
                'sentence': sentence,
                'is_supported': is_supported,
                'coverage': coverage,
                'method': 'heuristic'
            })
            
            if not is_supported:
                unsupported_sentences.append(sentence)
        
        # Calcular score global
        if verifications:
            fidelity_score = sum(
                1 for v in verifications if v['is_supported']
            ) / len(verifications)
        else:
            fidelity_score = 1.0  # Si no hay oraciones, asumir OK
        
        return {
            'fidelity_score': fidelity_score,
            'is_faithful': fidelity_score >= self.threshold,
            'verifications': verifications,
            'unsupported_sentences': unsupported_sentences,
            'num_sentences': len(sentences),
            'num_unsupported': len(unsupported_sentences)
        }
    
    def verify_with_nli(
        self,
        answer: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict[str, any]:
        """
        Verificación usando modelo NLI (Natural Language Inference)
        Más preciso pero más lento
        """
        if not self.nli_pipeline:
            logger.warning("No hay modelo NLI cargado, usando heurísticas")
            return self.verify_with_heuristics(answer, contexts)
        
        sentences = self.split_into_sentences(answer)
        context_texts = [chunk.text for chunk, _ in contexts]
        
        verifications = []
        unsupported_sentences = []
        
        for sentence in sentences:
            # Para cada oración, verificar contra cada contexto
            max_entailment_score = 0.0
            
            for context in context_texts:
                # Verificar si el contexto implica (entails) la oración
                result = self.nli_pipeline(
                    f"{context} [SEP] {sentence}",
                    truncation=True
                )
                
                # Buscar label "entailment" o "neutral"
                for pred in result:
                    if pred['label'].lower() in ['entailment', 'neutral']:
                        max_entailment_score = max(
                            max_entailment_score,
                            pred['score']
                        )
            
            is_supported = max_entailment_score >= self.threshold
            
            verifications.append({
                'sentence': sentence,
                'is_supported': is_supported,
                'score': max_entailment_score,
                'method': 'nli'
            })
            
            if not is_supported:
                unsupported_sentences.append(sentence)
        
        # Score global
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
    
    def verify(
        self,
        answer: str,
        contexts: List[Tuple[Chunk, float]],
        method: str = "auto"
    ) -> Dict[str, any]:
        """
        Verifica la fidelidad de una respuesta
        
        Args:
            answer: Respuesta generada por el LLM
            contexts: Contextos usados para generar la respuesta
            method: 'heuristic', 'nli', o 'auto'
        
        Returns:
            Dict con métricas de verificación
        """
        logger.info("Verificando fidelidad de la respuesta...")
        
        if method == "auto":
            method = "nli" if self.nli_pipeline else "heuristic"
        
        if method == "nli":
            result = self.verify_with_nli(answer, contexts)
        else:
            result = self.verify_with_heuristics(answer, contexts)
        
        # Añadir recomendación
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
    ) -> Dict[str, any]:
        """
        Verifica que la respuesta incluya citas a las fuentes
        """
        # Buscar referencias del tipo [1], [2], etc.
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
    # Ejemplo de uso
    verifier = FidelityVerifier(threshold=0.7)
    
    print("✓ Módulo verifier listo para usar")
    print("Implementa verificación de fidelidad con heurísticas y NLI (opcional)")
