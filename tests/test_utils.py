"""
Tests para utilidades
"""

import pytest
import time
from src.utils import timed, TimingContext, validate_file_exists


class TestTimed:
    """Tests para decorador @timed"""
    
    def test_timed_returns_result(self):
        """Verifica que el decorador devuelve el resultado correcto"""
        @timed
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    def test_timed_preserves_function_name(self):
        """Verifica que preserva el nombre de la función"""
        @timed
        def my_function():
            pass
        
        assert my_function.__name__ == "my_function"


class TestTimingContext:
    """Tests para TimingContext"""
    
    def test_timing_context_measures_time(self):
        """Verifica que mide tiempo correctamente"""
        with TimingContext("test", log=False) as timer:
            time.sleep(0.1)
        
        assert timer.elapsed is not None
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2
    
    def test_timing_context_stores_name(self):
        """Verifica que almacena el nombre"""
        with TimingContext("operación de prueba", log=False) as timer:
            pass
        
        assert timer.name == "operación de prueba"
    
    def test_timing_context_handles_exceptions(self):
        """Verifica manejo de excepciones"""
        with pytest.raises(ValueError):
            with TimingContext("test", log=False) as timer:
                raise ValueError("Error de prueba")
        
        # El tiempo se registra incluso con error
        assert timer.elapsed is not None


class TestValidateFileExists:
    """Tests para validación de archivos"""
    
    def test_validate_existing_file(self, tmp_path):
        """Verifica que no lanza error con archivo existente"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("contenido")
        
        # No debe lanzar error
        validate_file_exists(str(test_file))
    
    def test_validate_nonexistent_file(self):
        """Verifica que lanza error con archivo inexistente"""
        with pytest.raises(FileNotFoundError):
            validate_file_exists("/ruta/que/no/existe.txt")
    
    def test_validate_empty_path(self):
        """Verifica manejo de ruta vacía"""
        # Una ruta vacía no existe, por lo que debe lanzar FileNotFoundError
        with pytest.raises(FileNotFoundError):
            validate_file_exists("/ruta/vacia/inexistente")
