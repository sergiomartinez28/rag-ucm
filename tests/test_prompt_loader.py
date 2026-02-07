"""
Tests para el cargador de prompts
"""

import pytest
from pathlib import Path
from src.prompt_loader import load_prompt, list_prompts, PROMPTS_DIR


class TestLoadPrompt:
    """Tests para carga de prompts"""
    
    def test_load_system_prompt(self):
        """Verifica que carga el prompt de sistema"""
        prompt = load_prompt("system_prompt")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_load_user_prompt_with_variables(self):
        """Verifica que formatea variables correctamente"""
        prompt = load_prompt(
            "user_prompt",
            contexts="[1] Documento de prueba",
            query="¿Cuál es el plazo?"
        )
        
        assert "Documento de prueba" in prompt
        assert "plazo" in prompt
    
    def test_load_nonexistent_prompt_raises_error(self):
        """Verifica error con prompt inexistente"""
        with pytest.raises(FileNotFoundError):
            load_prompt("prompt_que_no_existe")
    
    def test_prompts_directory_exists(self):
        """Verifica que existe el directorio de prompts"""
        assert PROMPTS_DIR.exists()
        assert PROMPTS_DIR.is_dir()


class TestListPrompts:
    """Tests para listar prompts"""
    
    def test_list_prompts_returns_list(self):
        """Verifica que devuelve lista"""
        prompts = list_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
    
    def test_list_prompts_includes_required(self):
        """Verifica que incluye prompts requeridos"""
        prompts = list_prompts()
        
        assert "system_prompt" in prompts
        assert "user_prompt" in prompts
