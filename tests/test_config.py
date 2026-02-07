"""
Tests para el módulo de configuración
"""

import pytest
from src.config import (
    RAGConfig,
    ModelConfig,
    ChunkingConfig,
    RetrievalConfig,
    GenerationConfig,
    get_config,
    reset_config
)


class TestModelConfig:
    """Tests para configuración de modelos"""
    
    def test_default_values(self):
        """Verifica valores por defecto"""
        config = ModelConfig()
        
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.reranker_model == "BAAI/bge-reranker-base"
        assert config.llm_model == "Qwen/Qwen2.5-3B-Instruct"
        assert config.use_cross_encoder is True
    
    def test_reranker_type_validation(self):
        """Verifica validación de reranker_type"""
        # Válidos
        config = ModelConfig(reranker_type="cross-encoder")
        assert config.reranker_type == "cross-encoder"
        
        config = ModelConfig(reranker_type="simple")
        assert config.reranker_type == "simple"
        
        # Inválido
        with pytest.raises(ValueError):
            ModelConfig(reranker_type="invalid")
    
    def test_cross_encoder_auto_disable(self):
        """Verifica que cross-encoder se desactiva con reranker_type='simple'"""
        config = ModelConfig(reranker_type="simple", use_cross_encoder=True)
        assert config.use_cross_encoder is False


class TestChunkingConfig:
    """Tests para configuración de chunking"""
    
    def test_default_values(self):
        """Verifica valores por defecto"""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_chunk_size_bounds(self):
        """Verifica límites de chunk_size"""
        # Muy pequeño
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=50)
        
        # Muy grande
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=3000)
        
        # Válido
        config = ChunkingConfig(chunk_size=500)
        assert config.chunk_size == 500
    
    def test_overlap_must_be_less_than_size(self):
        """Verifica que overlap < chunk_size"""
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=500, chunk_overlap=500)
        
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=500, chunk_overlap=600)


class TestRetrievalConfig:
    """Tests para configuración de recuperación"""
    
    def test_default_values(self):
        """Verifica valores por defecto optimizados"""
        config = RetrievalConfig()
        
        assert config.top_k_retrieval == 10
        assert config.top_k_rerank == 3
        assert config.hybrid_alpha == 0.45
        assert config.min_score_threshold == 0.5
    
    def test_hybrid_alpha_bounds(self):
        """Verifica que alpha está entre 0 y 1"""
        with pytest.raises(ValueError):
            RetrievalConfig(hybrid_alpha=-0.1)
        
        with pytest.raises(ValueError):
            RetrievalConfig(hybrid_alpha=1.5)
        
        # Extremos válidos
        config = RetrievalConfig(hybrid_alpha=0.0)
        assert config.hybrid_alpha == 0.0
        
        config = RetrievalConfig(hybrid_alpha=1.0)
        assert config.hybrid_alpha == 1.0


class TestGenerationConfig:
    """Tests para configuración de generación"""
    
    def test_default_values(self):
        """Verifica valores por defecto"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.temperature == 0.1
        assert config.retry_on_abstention is True
    
    def test_temperature_bounds(self):
        """Verifica límites de temperatura"""
        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            GenerationConfig(temperature=2.5)


class TestRAGConfig:
    """Tests para configuración completa"""
    
    def test_default_config(self):
        """Verifica que se crea configuración completa por defecto"""
        config = RAGConfig()
        
        assert config.models is not None
        assert config.chunking is not None
        assert config.retrieval is not None
        assert config.generation is not None
        assert config.paths is not None
    
    def test_to_dict(self):
        """Verifica conversión a diccionario"""
        config = RAGConfig()
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert "models" in d
        assert "chunking" in d
        assert "retrieval" in d
    
    def test_get_config_singleton(self):
        """Verifica patrón singleton"""
        reset_config()
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reset_config(self):
        """Verifica reset de configuración"""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        # Son objetos diferentes después del reset
        assert config1 is not config2
        # Pero con los mismos valores
        assert config1.models.llm_model == config2.models.llm_model
