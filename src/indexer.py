"""
Módulo de indexación de documentos
Crea índices FAISS (vectorial) y BM25 (léxico)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger

from .preprocessor import Chunk
from .utils import timed, TimingContext, torch_memory_cleanup, ProgressTracker, ensure_dir


class DocumentIndexer:
    """
    Indexa chunks de documentos en:
    1. FAISS para búsqueda semántica por embeddings
    2. BM25 para búsqueda léxica exacta
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        faiss_index_path: Optional[Path] = None,
        bm25_index_path: Optional[Path] = None
    ):
        """
        Args:
            embedding_model: Modelo de HuggingFace para embeddings
            faiss_index_path: Ruta para guardar índice FAISS
            bm25_index_path: Ruta para guardar índice BM25
        """
        self.embedding_model_name = embedding_model
        self.faiss_index_path = Path(faiss_index_path) if faiss_index_path else None
        self.bm25_index_path = Path(bm25_index_path) if bm25_index_path else None
        
        logger.info(f"Cargando modelo de embeddings: {embedding_model}")
        
        # Detectar dispositivo y cargar modelo optimizado
        with TimingContext("Carga de modelo de embeddings"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Índices
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        
        logger.success(f"✓ DocumentIndexer inicializado (dim={self.embedding_dim}, device={device})")
    
    @timed
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos a embebir
            batch_size: Tamaño de batch para procesamiento
        
        Returns:
            Array numpy con embeddings (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Lista de textos vacía")
        
        logger.info(f"Generando embeddings para {len(texts)} textos...")
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Importante para búsqueda por similitud coseno
        )
        
        logger.success(f"✓ Embeddings generados: shape={embeddings.shape}")
        return embeddings
    
    @timed
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Construye índice FAISS para búsqueda de similitud
        Usa Inner Product (IP) que es equivalente a coseno con embeddings normalizados
        
        Args:
            embeddings: Matriz de embeddings normalizados
        
        Returns:
            Índice FAISS construido
        """
        if embeddings.size == 0:
            raise ValueError("Array de embeddings vacío")
        
        logger.info("Construyendo índice FAISS...")
        
        # IndexFlatIP: búsqueda exacta por producto interno (=coseno si normalizados)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        
        logger.success(f"✓ Índice FAISS construido: {index.ntotal} vectores")
        return index
    
    @timed
    def build_bm25_index(self, texts: List[str]) -> BM25Okapi:
        """
        Construye índice BM25 para búsqueda léxica
        
        Args:
            texts: Lista de textos a indexar
        
        Returns:
            Índice BM25
        """
        if not texts:
            raise ValueError("Lista de textos vacía")
        
        logger.info("Construyendo índice BM25...")
        
        # Tokenizar (simple split por espacios)
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        logger.success(f"✓ Índice BM25 construido: {len(tokenized_corpus)} documentos")
        return bm25
    
    @timed
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Indexa una lista de chunks en ambos índices (FAISS + BM25)
        
        Args:
            chunks: Lista de chunks procesados
        
        Raises:
            ValueError: Si la lista de chunks está vacía
        """
        if not chunks:
            raise ValueError("No hay chunks para indexar")
        
        logger.info(f"Indexando {len(chunks)} chunks...")
        
        with torch_memory_cleanup():
            # Guardar chunks y textos
            self.chunks = chunks
            self.chunk_texts = [chunk.text for chunk in chunks]
            
            # Crear embeddings
            embeddings = self.create_embeddings(self.chunk_texts)
            
            # Construir índices
            self.faiss_index = self.build_faiss_index(embeddings)
            self.bm25_index = self.build_bm25_index(self.chunk_texts)
        
        logger.success(f"✓ Indexación completa: {len(chunks)} chunks indexados")
    
    def save_indices(self) -> None:
        """
        Guarda ambos índices a disco
        
        Raises:
            RuntimeError: Si no hay índices para guardar
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise RuntimeError("No hay índices para guardar. Ejecuta index_chunks() primero.")
        
        if not self.faiss_index_path or not self.bm25_index_path:
            raise RuntimeError("Rutas de índices no configuradas")
        
        # Crear directorios si no existen
        ensure_dir(str(self.faiss_index_path))
        ensure_dir(str(self.bm25_index_path))
        
        # Guardar FAISS
        with TimingContext("Guardado de índice FAISS"):
            logger.info(f"Guardando índice FAISS en {self.faiss_index_path}")
            faiss.write_index(self.faiss_index, str(self.faiss_index_path / "index.faiss"))
            
            # Guardar metadatos (chunks)
            with open(self.faiss_index_path / "chunks.pkl", 'wb') as f:
                pickle.dump(self.chunks, f)
        
        # Guardar BM25
        with TimingContext("Guardado de índice BM25"):
            logger.info(f"Guardando índice BM25 en {self.bm25_index_path}")
            with open(self.bm25_index_path / "bm25.pkl", 'wb') as f:
                pickle.dump({
                    'index': self.bm25_index,
                    'texts': self.chunk_texts
                }, f)
        
        logger.success("✓ Índices guardados correctamente")
    
    def load_indices(self) -> None:
        """
        Carga índices desde disco
        
        Raises:
            FileNotFoundError: Si no se encuentran los archivos de índices
        """
        if not self.faiss_index_path or not self.bm25_index_path:
            raise RuntimeError("Rutas de índices no configuradas")
        
        logger.info("Cargando índices desde disco...")
        
        # Cargar FAISS
        with TimingContext("Carga de índice FAISS"):
            faiss_path = self.faiss_index_path / "index.faiss"
            chunks_path = self.faiss_index_path / "chunks.pkl"
            
            if not faiss_path.exists():
                raise FileNotFoundError(f"No se encontró índice FAISS en {faiss_path}")
            
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
        
        # Cargar BM25
        with TimingContext("Carga de índice BM25"):
            bm25_path = self.bm25_index_path / "bm25.pkl"
            
            if not bm25_path.exists():
                raise FileNotFoundError(f"No se encontró índice BM25 en {bm25_path}")
            
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25_index = bm25_data['index']
                self.chunk_texts = bm25_data['texts']
        
        logger.success(f"✓ Índices cargados: {len(self.chunks)} chunks")
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Búsqueda semántica usando FAISS
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a retornar
        
        Returns:
            Lista de tuplas (chunk, score)
        """
        if self.faiss_index is None:
            raise RuntimeError("Índice FAISS no inicializado")
        
        # Crear embedding de la query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Buscar en FAISS
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
        
        return results
    
    def search_lexical(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Búsqueda léxica usando BM25
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a retornar
        
        Returns:
            Lista de tuplas (chunk, score)
        """
        if self.bm25_index is None:
            raise RuntimeError("Índice BM25 no inicializado")
        
        # Tokenizar query
        query_tokens = query.lower().split()
        
        # Obtener scores BM25
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Obtener top-k índices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.chunks[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de los índices"""
        if self.faiss_index is None:
            return {"status": "No indexado"}
        
        return {
            "total_chunks": len(self.chunks),
            "faiss_vectors": self.faiss_index.ntotal,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "avg_chunk_length": float(np.mean([len(t.split()) for t in self.chunk_texts])),
        }
    
    def cleanup(self) -> None:
        """Libera recursos de memoria"""
        logger.info("Liberando recursos del indexer...")
        
        with torch_memory_cleanup():
            self.embedding_model = None
        
        logger.success("✓ Recursos del indexer liberados")


if __name__ == "__main__":
    # Ejemplo de uso
    from pathlib import Path
    
    indexer = DocumentIndexer(
        embedding_model="BAAI/bge-m3",
        faiss_index_path=Path("data/processed/faiss_index"),
        bm25_index_path=Path("data/processed/bm25_index")
    )
    
    print("✓ Módulo indexer listo para usar")
    print(f"Dimensión de embeddings: {indexer.embedding_dim}")
