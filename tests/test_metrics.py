"""
Tests para el calculador de métricas
"""

import pytest
from src.evaluator.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests para MetricsCalculator"""
    
    @pytest.fixture
    def calculator(self):
        return MetricsCalculator()
    
    def test_is_abstention_positive(self, calculator):
        """Verifica detección de abstenciones"""
        abstentions = [
            "No dispongo de información sobre este tema.",
            "No tengo información disponible.",
            "No encuentro información relevante.",
            "No puedo responder a esta pregunta.",
        ]
        
        for answer in abstentions:
            assert calculator._is_abstention(answer) is True
    
    def test_is_abstention_negative(self, calculator):
        """Verifica que respuestas normales no son abstenciones"""
        normal_answers = [
            "El plazo es de 30 días naturales.",
            "Los estudiantes deben matricularse en septiembre.",
            "Se pueden reconocer hasta 36 créditos ECTS.",
        ]
        
        for answer in normal_answers:
            assert calculator._is_abstention(answer) is False
    
    def test_is_suspicious_short(self, calculator):
        """Verifica detección de respuestas sospechosamente cortas"""
        suspicious = ["", "1", "42", "OK"]
        
        for answer in suspicious:
            assert calculator._is_suspicious_short(answer) is True
    
    def test_is_suspicious_short_normal(self, calculator):
        """Verifica que respuestas normales no son sospechosas"""
        normal = [
            "El plazo es de 30 días.",
            "Sí, se puede solicitar.",
        ]
        
        for answer in normal:
            assert calculator._is_suspicious_short(answer) is False
    
    def test_extract_factual_data_numbers(self, calculator):
        """Verifica extracción de números"""
        text = "El plazo es de 30 días y se requieren 6 créditos."
        facts = calculator._extract_factual_data(text)
        
        assert "30" in facts
        assert "6" in facts
    
    def test_extract_factual_data_percentages(self, calculator):
        """Verifica extracción de porcentajes"""
        text = "Se aplica un 15% de descuento."
        facts = calculator._extract_factual_data(text)
        
        # La regex captura "15%" como un token
        assert "15" in facts or "15%" in facts
    
    def test_context_has_numbers(self, calculator):
        """Verifica detección de números en contexto"""
        sources_with_numbers = [
            {"text_preview": "El plazo es de 30 días naturales."}
        ]
        sources_without_numbers = [
            {"text_preview": "Los estudiantes deben matricularse."}
        ]
        
        assert calculator._context_has_numbers(sources_with_numbers) is True
        assert calculator._context_has_numbers(sources_without_numbers) is False
    
    def test_context_has_keywords(self, calculator):
        """Verifica detección de keywords normativas"""
        sources_with_keywords = [
            {"text_preview": "El estudiante deberá presentar la solicitud."}
        ]
        sources_without_keywords = [
            {"text_preview": "Texto sin palabras clave especiales."}
        ]
        
        assert calculator._context_has_keywords(sources_with_keywords) is True
        assert calculator._context_has_keywords(sources_without_keywords) is False
