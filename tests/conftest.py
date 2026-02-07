"""
Fixtures compartidos para tests de RAG-UCM
"""

import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Añadir src al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MockChunk:
    """Chunk simulado para tests"""
    text: str
    metadata: Dict[str, str]
    chunk_id: str
    doc_id: str


@pytest.fixture
def sample_chunks() -> List[MockChunk]:
    """Chunks de ejemplo para tests"""
    return [
        MockChunk(
            text="El plazo para presentar el TFM es de 30 días naturales desde la defensa.",
            metadata={"title": "Normativa TFM", "faculty": "Informática"},
            chunk_id="tfm_001_chunk_0",
            doc_id="tfm_001"
        ),
        MockChunk(
            text="Los estudiantes pueden solicitar reconocimiento de hasta 36 créditos ECTS.",
            metadata={"title": "Reconocimiento de créditos", "faculty": "General"},
            chunk_id="cred_002_chunk_0",
            doc_id="cred_002"
        ),
        MockChunk(
            text="La matrícula ordinaria se realiza en septiembre según calendario académico.",
            metadata={"title": "Calendario académico", "faculty": "General"},
            chunk_id="cal_003_chunk_0",
            doc_id="cal_003"
        ),
    ]


@pytest.fixture
def sample_text() -> str:
    """Texto de ejemplo para tests de preprocesamiento"""
    return """
    Artículo 1. Objeto y ámbito de aplicación.
    
    El presente reglamento tiene por objeto regular los Trabajos Fin de Máster 
    en la Universidad Complutense de Madrid. Se aplicará a todos los estudiantes 
    matriculados en másteres oficiales.
    
    Artículo 2. Características del TFM.
    
    El Trabajo Fin de Máster tendrá una extensión mínima de 6 créditos ECTS y 
    máxima de 30 créditos ECTS, según lo establecido en cada plan de estudios.
    El estudiante deberá defender públicamente su trabajo ante un tribunal.
    
    Artículo 3. Plazos de presentación.
    
    Los plazos de presentación serán fijados por cada facultad, respetando 
    un mínimo de 30 días naturales entre la entrega y la defensa.
    """


@pytest.fixture
def sample_queries() -> List[str]:
    """Preguntas de ejemplo para tests"""
    return [
        "¿Cuál es el plazo para presentar el TFM?",
        "¿Cuántos créditos puedo reconocer?",
        "¿Cuándo es la matrícula?",
    ]


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Directorio temporal para datos de test"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    return data_dir
