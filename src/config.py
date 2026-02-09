"""
Configuración centralizada del sistema RAG-UCM

Todos los parámetros del sistema están definidos aquí con valores optimizados
tras la fase de experimentación. Usa Pydantic para validación de tipos.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuración de modelos"""
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Modelo de embeddings"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Modelo de re-ranking"
    )
    reranker_type: str = Field(
        default="cross-encoder",
        description="Tipo de re-ranking ('cross-encoder' o 'simple')"
    )
    use_cross_encoder: bool = Field(
        default=True,
        description="Usar cross-encoder para re-ranking (desactivar en CPU si es muy lento)"
    )
    llm_model: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Modelo de lenguaje para generación"
    )
    
    @field_validator('reranker_type')
    @classmethod
    def validate_reranker_type(cls, v: str) -> str:
        allowed = ['cross-encoder', 'simple']
        if v.lower() not in allowed:
            raise ValueError(f"reranker_type debe ser uno de {allowed}")
        return v.lower()
    
    @field_validator('use_cross_encoder')
    @classmethod
    def validate_use_cross_encoder(cls, v: bool, info) -> bool:
        reranker_type = info.data.get('reranker_type', 'cross-encoder')
        if reranker_type == 'simple' and v:
            # Si reranker_type es 'simple', desactivar cross-encoder automáticamente
            return False
        return v


class ChunkingConfig(BaseModel):
    """Configuración de chunking"""
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="Tamaño de chunks en palabras"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Solapamiento entre chunks"
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        return v


class RetrievalConfig(BaseModel):
    """Configuración de recuperación"""
    top_k_retrieval: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Candidatos a recuperar en la búsqueda inicial"
    )
    top_k_rerank: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Documentos finales después de re-ranking"
    )
    hybrid_alpha: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Balance BM25/Semántico (0=solo BM25, 1=solo semántico)"
    )
    min_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de score del reranker para filtrar chunks irrelevantes"
    )


class GenerationConfig(BaseModel):
    """Configuración de generación"""
    max_new_tokens: int = Field(
        default=100,
        ge=50,
        le=1024,
        description="Tokens máximos a generar"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperatura de sampling"
    )
    use_sentence_extraction: bool = Field(
        default=False,
        description="Usar extracción de oraciones (desactivado para evitar perder respuestas)"
    )
    max_context_chars_per_chunk: int = Field(
        default=1000,
        ge=400,
        le=3000,
        description="Máximo de caracteres por chunk en el contexto del prompt"
    )
    # Retry en abstenciones
    retry_on_abstention: bool = Field(
        default=True,
        description="Reintentar cuando el modelo abstiene pero hay señales de respuesta en contexto"
    )


class VerificationConfig(BaseModel):
    """Configuración de verificación"""
    enable_verification: bool = Field(
        default=False,
        description="Activar verificación de fidelidad"
    )
    verification_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Umbral de fidelidad"
    )


class PathsConfig(BaseModel):
    """Configuración de rutas"""
    data_raw_path: Path = Field(
        default=Path("./data/raw"),
        description="Ruta a documentos originales"
    )
    data_processed_path: Path = Field(
        default=Path("./data/processed"),
        description="Ruta a datos procesados"
    )
    faiss_index_path: Path = Field(
        default=Path("./data/processed/faiss_index"),
        description="Ruta a índice FAISS"
    )
    bm25_index_path: Path = Field(
        default=Path("./data/processed/bm25_index"),
        description="Ruta a índice BM25"
    )
    
    @field_validator('data_raw_path', 'data_processed_path', 'faiss_index_path', 'bm25_index_path')
    @classmethod
    def ensure_path(cls, v: Path) -> Path:
        """Convierte a Path si es string"""
        return Path(v) if not isinstance(v, Path) else v


class RAGConfig(BaseModel):
    """Configuración completa del sistema RAG"""
    models: ModelConfig = Field(default_factory=ModelConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para compatibilidad"""
        return self.model_dump()


# Instancia global de configuración (singleton)
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Obtiene la configuración global del sistema"""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reset_config() -> None:
    """Resetea la configuración (útil para tests)"""
    global _config
    _config = None
