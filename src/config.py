"""
Configuración centralizada del sistema RAG-UCM
Usa Pydantic para validación de tipos y valores por defecto
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()


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
    """Configuración de recuperación - Fase 2: Optimizada para Precision@k"""
    top_k_retrieval: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Candidatos a recuperar (Fase 2: aumentado 8→20)"
    )
    top_k_rerank: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Documentos finales después de re-ranking (Fase 2: aumentado 3→5)"
    )
    hybrid_alpha: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Balance BM25/Semántico (Fase 2: 0.6→0.45 para factual)"
    )
    min_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de score del reranker para filtrar ruido (Fase 2)"
    )


class GenerationConfig(BaseModel):
    """Configuración de generación"""
    max_new_tokens: int = Field(
        default=120,
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
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Carga configuración desde variables de entorno"""
        # Crear instancias default para obtener los valores por defecto
        default_retrieval = RetrievalConfig()
        
        return cls(
            models=ModelConfig(
                embedding_model=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'),
                reranker_model=os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base'),
                reranker_type=os.getenv('RERANKER_TYPE', 'cross-encoder'),
                use_cross_encoder=os.getenv('USE_CROSS_ENCODER', 'true').lower() == 'true',
                llm_model=os.getenv('LLM_MODEL', 'Qwen/Qwen2.5-3B-Instruct'),
            ),
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
            ),
            retrieval=RetrievalConfig(
                top_k_retrieval=int(os.getenv('TOP_K_RETRIEVAL', default_retrieval.top_k_retrieval)),
                top_k_rerank=int(os.getenv('TOP_K_RERANK', default_retrieval.top_k_rerank)),
                hybrid_alpha=float(os.getenv('HYBRID_ALPHA', default_retrieval.hybrid_alpha)),
                min_score_threshold=float(os.getenv('MIN_SCORE_THRESHOLD', default_retrieval.min_score_threshold)),
            ),
            generation=GenerationConfig(
                max_new_tokens=int(os.getenv('MAX_NEW_TOKENS', 120)),
                temperature=float(os.getenv('TEMPERATURE', 0.1)),
            ),
            verification=VerificationConfig(
                enable_verification=os.getenv('ENABLE_VERIFICATION', 'false').lower() == 'true',
                verification_threshold=float(os.getenv('VERIFICATION_THRESHOLD', 0.7)),
            ),
            paths=PathsConfig(
                data_raw_path=Path(os.getenv('DATA_RAW_PATH', './data/raw')),
                data_processed_path=Path(os.getenv('DATA_PROCESSED_PATH', './data/processed')),
                faiss_index_path=Path(os.getenv('FAISS_INDEX_PATH', './data/processed/faiss_index')),
                bm25_index_path=Path(os.getenv('BM25_INDEX_PATH', './data/processed/bm25_index')),
            ),
        )
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para compatibilidad"""
        return self.model_dump()


# Instancia global de configuración (singleton)
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Obtiene la configuración global del sistema"""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def reload_config() -> RAGConfig:
    """Recarga la configuración desde variables de entorno"""
    global _config
    load_dotenv(override=True)
    _config = RAGConfig.from_env()
    return _config
