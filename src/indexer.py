"""
Módulo de indexación de documentos
Crea índices FAISS (vectorial) y BM25 (léxico)
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger
from tqdm import tqdm

from .preprocessor import Chunk


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
        self.faiss_index_path = faiss_index_path
        self.bm25_index_path = bm25_index_path
        
        logger.info(f"Cargando modelo de embeddings: {embedding_model}")
        # Detectar dispositivo y cargar modelo optimizado
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Índices
        self.faiss_index = None
        self.bm25_index = None
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        
        logger.success(f"✓ DocumentIndexer inicializado (dim={self.embedding_dim}, device={device})")
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos a embebir
            batch_size: Tamaño de batch para procesamiento
        
        Returns:
            Array numpy con embeddings (n_texts, embedding_dim)
        """
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
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Construye índice FAISS para búsqueda de similitud
        Usa Inner Product (IP) que es equivalente a coseno con embeddings normalizados
        
        Args:
            embeddings: Matriz de embeddings normalizados
        
        Returns:
            Índice FAISS construido
        """
        logger.info("Construyendo índice FAISS...")
        
        # IndexFlatIP: búsqueda exacta por producto interno (=coseno si normalizados)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        
        logger.success(f"✓ Índice FAISS construido: {index.ntotal} vectores")
        return index
    
    def build_bm25_index(self, texts: List[str]) -> BM25Okapi:
        """
        Construye índice BM25 para búsqueda léxica
        
        Args:
            texts: Lista de textos a indexar
        
        Returns:
            Índice BM25
        """
        logger.info("Construyendo índice BM25...")
        
        # Tokenizar (simple split por espacios)
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        logger.success(f"✓ Índice BM25 construido: {len(tokenized_corpus)} documentos")
        return bm25
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Indexa una lista de chunks en ambos índices (FAISS + BM25)
        
        Args:
            chunks: Lista de chunks procesados
        """
        if not chunks:
            logger.warning("No hay chunks para indexar")
            return
        
        logger.info(f"Indexando {len(chunks)} chunks...")
        
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
        """Guarda ambos índices a disco"""
        if self.faiss_index is None or self.bm25_index is None:
            logger.error("No hay índices para guardar. Ejecuta index_chunks() primero.")
            return
        
        # Crear directorios si no existen
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar FAISS
        logger.info(f"Guardando índice FAISS en {self.faiss_index_path}")
        faiss.write_index(self.faiss_index, str(self.faiss_index_path / "index.faiss"))
        
        # Guardar metadatos (chunks)
        with open(self.faiss_index_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Guardar BM25
        logger.info(f"Guardando índice BM25 en {self.bm25_index_path}")
        with open(self.bm25_index_path / "bm25.pkl", 'wb') as f:
            pickle.dump({
                'index': self.bm25_index,
                'texts': self.chunk_texts
            }, f)
        
        logger.success("✓ Índices guardados correctamente")
    
    def load_indices(self) -> None:
        """Carga índices desde disco"""
        logger.info("Cargando índices desde disco...")
        
        # Cargar FAISS
        faiss_path = self.faiss_index_path / "index.faiss"
        chunks_path = self.faiss_index_path / "chunks.pkl"
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"No se encontró índice FAISS en {faiss_path}")
        
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Cargar BM25
        bm25_path = self.bm25_index_path / "bm25.pkl"
        
        if not bm25_path.exists():
            raise FileNotFoundError(f"No se encontró índice BM25 en {bm25_path}")
        
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25_index = bm25_data['index']
            self.chunk_texts = bm25_data['texts']
        
        logger.success(f"✓ Índices cargados: {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict[str, any]:
        """Retorna estadísticas de los índices"""
        if self.faiss_index is None:
            return {"status": "No indexado"}
        
        return {
            "total_chunks": len(self.chunks),
            "faiss_vectors": self.faiss_index.ntotal,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "avg_chunk_length": np.mean([len(t.split()) for t in self.chunk_texts]),
        }


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
