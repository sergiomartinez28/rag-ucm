"""
Módulo de recuperación híbrida
Combina BM25 + embeddings + re-ranking
"""

from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from loguru import logger

from .indexer import DocumentIndexer
from .preprocessor import Chunk
from .utils import timed, TimingContext, torch_memory_cleanup


class HybridRetriever:
    """
    Recuperador híbrido que combina:
    1. Búsqueda BM25 (léxica)
    2. Búsqueda semántica (FAISS)
    3. Fusión con Reciprocal Rank Fusion
    4. Re-ranking con cross-encoder
    """
    
    def __init__(
        self,
        indexer: DocumentIndexer,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_type: str = "cross-encoder",
        alpha: float = 0.5
    ):
        """
        Args:
            indexer: Indexador con índices FAISS y BM25 ya construidos
            reranker_model: Modelo cross-encoder para re-ranking
            reranker_type: Tipo de re-ranking ('cross-encoder' o 'simple')
            alpha: Balance entre BM25 y semántico (0=solo BM25, 1=solo semántico)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha debe estar entre 0 y 1")
        
        self.indexer = indexer
        self.alpha = alpha
        self.reranker_type = reranker_type.lower().strip()
        self.reranker = None
        
        if self.reranker_type == "cross-encoder":
            with TimingContext("Carga de re-ranker"):
                logger.info(f"Cargando modelo de re-ranking: {reranker_model}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reranker = CrossEncoder(
                    reranker_model, 
                    device=device,
                    max_length=256
                )
                logger.info(f"Re-ranker cargado en: {device}")
        else:
            logger.info("Re-ranker simple activado (sin cross-encoder)")
        
        logger.success(f"✓ HybridRetriever inicializado (alpha={alpha})")
    
    @timed
    def search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Búsqueda BM25
        
        Returns:
            Lista de (índice_chunk, score) ordenados por relevancia
        """
        if not query or not query.strip():
            logger.warning("Query vacía para BM25")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.indexer.bm25_index.get_scores(tokenized_query)
        
        # Obtener top-k índices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        logger.debug(f"BM25: {len(results)} resultados")
        return results
    
    @timed
    def search_semantic(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Búsqueda semántica con FAISS
        
        Returns:
            Lista de (índice_chunk, score) ordenados por relevancia
        """
        if not query or not query.strip():
            logger.warning("Query vacía para búsqueda semántica")
            return []
        
        # Generar embedding de la query
        query_embedding = self.indexer.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Buscar en FAISS
        scores, indices = self.indexer.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        results = [
            (int(indices[0][i]), float(scores[0][i]))
            for i in range(len(indices[0]))
        ]
        
        logger.debug(f"Semántico: {len(results)} resultados")
        return results
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Fusión de resultados usando Reciprocal Rank Fusion (RRF)
        
        RRF score = sum(1 / (k + rank_i)) para cada lista
        
        Args:
            bm25_results: Resultados BM25
            semantic_results: Resultados semánticos
            k: Constante RRF (típicamente 60)
        
        Returns:
            Lista fusionada ordenada por score RRF
        """
        # Crear diccionario de scores RRF
        rrf_scores = {}
        
        # Añadir scores BM25
        for rank, (idx, _) in enumerate(bm25_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.alpha) / (k + rank)
        
        # Añadir scores semánticos
        for rank, (idx, _) in enumerate(semantic_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.alpha / (k + rank)
        
        # Ordenar por score RRF
        fused_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.debug(f"RRF: {len(fused_results)} resultados únicos fusionados")
        return fused_results
    
    @timed
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-ranking con cross-encoder
        
        Args:
            query: Query del usuario
            candidates: Lista de (índice_chunk, score_previo)
            top_k: Número de resultados finales
        
        Returns:
            Lista de (Chunk, score_reranking) ordenados
        """
        if not candidates:
            logger.warning("No hay candidatos para re-ranking")
            return []

        if self.reranker is None:
            return self.rerank_simple(candidates, top_k=top_k)
        
        # Preparar pares (query, documento) para el cross-encoder
        pairs = [
            [query, self.indexer.chunks[idx].text]
            for idx, _ in candidates
        ]
        
        # Calcular scores de re-ranking con batch processing
        rerank_scores = self.reranker.predict(
            pairs, 
            batch_size=min(32, len(pairs)), 
            show_progress_bar=False
        )
        
        # Los cross-encoders devuelven logits
        raw_scores = np.array(rerank_scores)
        
        logger.debug(
            f"Re-ranker raw scores: min={raw_scores.min():.3f}, "
            f"max={raw_scores.max():.3f}, mean={raw_scores.mean():.3f}"
        )
        
        # Normalización por ranking (mejor=1.0, peor=0.0)
        ranks = np.argsort(-raw_scores)
        normalized_scores = np.zeros_like(raw_scores, dtype=float)
        
        if len(raw_scores) > 1:
            normalized_scores[ranks] = 1 - (ranks / (len(raw_scores) - 1))
        else:
            normalized_scores[0] = 1.0
        
        # Combinar con chunks y ordenar
        results = [
            (self.indexer.chunks[candidates[i][0]], float(raw_scores[i]), float(normalized_scores[i]))
            for i in range(len(candidates))
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver solo (chunk, normalized_score)
        final_results = [(chunk, norm_score) for chunk, _, norm_score in results[:top_k]]
        
        logger.debug(f"Re-ranking: top {top_k} de {len(results)} candidatos")
        return final_results
    
    def rerank_simple(
        self,
        candidates: List[Tuple[int, float]],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-ranking simple basado en el score de fusion (RRF)
        """
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        n = len(ordered)
        results = []
        
        for i, (idx, _) in enumerate(ordered):
            if n > 1:
                score = 1 - (i / (n - 1))
            else:
                score = 1.0
            results.append((self.indexer.chunks[idx], float(score)))

        return results

    @timed
    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = 20,
        top_k_final: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Pipeline completo de recuperación híbrida
        
        Args:
            query: Pregunta del usuario
            top_k_retrieval: Candidatos a recuperar en cada búsqueda
            top_k_final: Resultados finales después de re-ranking
        
        Returns:
            Lista de (Chunk, score) con los fragmentos más relevantes
        """
        if not query or not query.strip():
            raise ValueError("Query no puede estar vacía")
        
        logger.info(f"Recuperando documentos para: '{query[:50]}...'")
        
        # 1. Búsqueda BM25
        with TimingContext("BM25", log=False) as timer:
            bm25_results = self.search_bm25(query, top_k=top_k_retrieval)
        logger.debug(f"  BM25: {timer.elapsed:.3f}s")
        
        # 2. Búsqueda semántica
        with TimingContext("Semántico", log=False) as timer:
            semantic_results = self.search_semantic(query, top_k=top_k_retrieval)
        logger.debug(f"  Semántico: {timer.elapsed:.3f}s")
        
        # 3. Fusión RRF
        with TimingContext("Fusión RRF", log=False) as timer:
            fused_results = self.reciprocal_rank_fusion(
                bm25_results,
                semantic_results
            )
        logger.debug(f"  Fusión: {timer.elapsed:.3f}s")
        
        # Limitar candidatos para re-ranking
        candidates = fused_results[:top_k_retrieval]
        
        # 4. Re-ranking
        with TimingContext("Re-ranking", log=False) as timer:
            if self.reranker_type == "cross-encoder":
                final_results = self.rerank(query, candidates, top_k=top_k_final)
            else:
                final_results = self.rerank_simple(candidates, top_k=top_k_final)
        logger.debug(f"  Rerank: {timer.elapsed:.3f}s")
        
        logger.success(f"✓ Recuperados {len(final_results)} documentos relevantes")
        
        return final_results
    
    def retrieve_with_threshold(
        self,
        query: str,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
        min_score: float = 0.3
    ) -> List[Tuple[Chunk, float]]:
        """
        Recuperación con umbral mínimo de relevancia
        Útil para evitar resultados irrelevantes
        """
        results = self.retrieve(query, top_k_retrieval, top_k_final)
        
        # Filtrar por score mínimo
        filtered_results = [
            (chunk, score) for chunk, score in results
            if score >= min_score
        ]
        
        if len(filtered_results) < len(results):
            logger.warning(
                f"Filtrados {len(results) - len(filtered_results)} "
                f"resultados con score < {min_score}"
            )
        
        return filtered_results
    
    def cleanup(self) -> None:
        """Libera recursos de memoria"""
        logger.info("Liberando recursos del retriever...")
        
        with torch_memory_cleanup():
            self.reranker = None
        
        logger.success("✓ Recursos del retriever liberados")


if __name__ == "__main__":
    print("✓ Módulo retrieval listo para usar")
    print("Implementa búsqueda híbrida BM25 + embeddings + re-ranking")
