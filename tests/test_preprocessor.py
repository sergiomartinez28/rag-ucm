"""
Tests para el módulo de preprocesamiento
"""

import pytest
from pathlib import Path
from src.preprocessor import DocumentPreprocessor, Chunk


class TestDocumentPreprocessor:
    """Tests para DocumentPreprocessor"""
    
    def test_init_default_values(self):
        """Verifica inicialización con valores por defecto"""
        preprocessor = DocumentPreprocessor()
        
        assert preprocessor.chunk_size == 1000
        assert preprocessor.chunk_overlap == 200
    
    def test_init_custom_values(self):
        """Verifica inicialización con valores personalizados"""
        preprocessor = DocumentPreprocessor(chunk_size=500, chunk_overlap=50)
        
        assert preprocessor.chunk_size == 500
        assert preprocessor.chunk_overlap == 50
    
    def test_overlap_must_be_less_than_size(self):
        """Verifica validación de overlap"""
        with pytest.raises(ValueError, match="chunk_overlap debe ser menor"):
            DocumentPreprocessor(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError):
            DocumentPreprocessor(chunk_size=100, chunk_overlap=150)


class TestTextCleaning:
    """Tests para limpieza de texto"""
    
    @pytest.fixture
    def preprocessor(self):
        return DocumentPreprocessor(chunk_size=500, chunk_overlap=50)
    
    def test_clean_text_removes_extra_whitespace(self, preprocessor):
        """Verifica que elimina espacios extra"""
        dirty = "Texto   con    muchos     espacios"
        clean = preprocessor.clean_text(dirty)
        
        assert "   " not in clean
        assert "  " not in clean
    
    def test_clean_text_normalizes_newlines(self, preprocessor):
        """Verifica normalización de saltos de línea"""
        dirty = "Línea 1\n\n\n\n\nLínea 2"
        clean = preprocessor.clean_text(dirty)
        
        # No más de 2 saltos consecutivos
        assert "\n\n\n" not in clean
    
    def test_clean_text_removes_headers_footers(self, preprocessor):
        """Verifica que elimina headers/footers comunes"""
        dirty = "Página 15 de 30\n\nContenido real del documento"
        clean = preprocessor.clean_text(dirty)
        
        # El contenido real se mantiene
        assert "Contenido real" in clean


class TestChunking:
    """Tests para división en chunks"""
    
    @pytest.fixture
    def preprocessor(self):
        return DocumentPreprocessor(chunk_size=100, chunk_overlap=20)
    
    def test_create_chunks_returns_list(self, preprocessor, sample_text):
        """Verifica que devuelve lista de chunks"""
        metadata = {"title": "Test", "filename": "test.pdf"}
        chunks = preprocessor.create_chunks(sample_text, metadata)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_chunk_has_required_fields(self, preprocessor, sample_text):
        """Verifica que cada chunk tiene campos requeridos"""
        metadata = {"title": "Test", "filename": "test.pdf"}
        chunks = preprocessor.create_chunks(sample_text, metadata)
        
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk, 'chunk_id')
            assert hasattr(chunk, 'doc_id')
            assert len(chunk.text) > 0
    
    def test_chunks_preserve_metadata(self, preprocessor, sample_text):
        """Verifica que chunks preservan metadata"""
        metadata = {"title": "Normativa TFM", "faculty": "Informática"}
        chunks = preprocessor.create_chunks(sample_text, metadata)
        
        for chunk in chunks:
            assert chunk.metadata["title"] == "Normativa TFM"
            assert chunk.metadata["faculty"] == "Informática"
    
    def test_chunk_ids_are_unique(self, preprocessor, sample_text):
        """Verifica que IDs de chunks son únicos"""
        metadata = {"title": "Test", "filename": "test.pdf"}
        chunks = preprocessor.create_chunks(sample_text, metadata)
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
    
    def test_empty_text_returns_empty_list(self, preprocessor):
        """Verifica manejo de texto vacío"""
        metadata = {"title": "Test"}
        chunks = preprocessor.create_chunks("", metadata)
        
        assert chunks == []
    
    def test_short_text_returns_single_chunk(self, preprocessor):
        """Verifica que texto corto genera un solo chunk"""
        short_text = "Este es un texto muy corto."
        metadata = {"title": "Test"}
        chunks = preprocessor.create_chunks(short_text, metadata)
        
        assert len(chunks) == 1
        assert "texto muy corto" in chunks[0].text


class TestHTMLExtraction:
    """Tests para extracción de HTML"""
    
    @pytest.fixture
    def preprocessor(self):
        return DocumentPreprocessor()
    
    def test_extract_from_html_basic(self, preprocessor, tmp_path):
        """Verifica extracción básica de HTML"""
        html_content = """
        <html>
        <body>
            <h1>Título del documento</h1>
            <p>Este es el contenido principal.</p>
            <p>Segundo párrafo con información.</p>
        </body>
        </html>
        """
        
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")
        
        text = preprocessor.extract_from_html(html_file)
        
        assert "Título del documento" in text
        assert "contenido principal" in text
    
    def test_extract_from_html_removes_scripts(self, preprocessor, tmp_path):
        """Verifica que elimina scripts y estilos"""
        html_content = """
        <html>
        <head>
            <script>alert('malicious');</script>
            <style>.hidden { display: none; }</style>
        </head>
        <body>
            <p>Contenido visible</p>
        </body>
        </html>
        """
        
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")
        
        text = preprocessor.extract_from_html(html_file)
        
        assert "alert" not in text
        assert "display" not in text
        assert "Contenido visible" in text
