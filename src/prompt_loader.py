"""
Utilidad para cargar prompts desde archivos externos
"""

from pathlib import Path
from typing import Optional
from loguru import logger


# Ruta base de prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """
    Carga un prompt desde archivo y formatea con variables
    
    Args:
        name: Nombre del archivo de prompt (sin extensión .txt)
        **kwargs: Variables para formatear el prompt
    
    Returns:
        Prompt formateado
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Formatear si hay variables
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Variable faltante en prompt {name}: {e}")
            return template
    
    return template


def get_prompt_path(name: str) -> Path:
    """Obtiene la ruta de un archivo de prompt"""
    return PROMPTS_DIR / f"{name}.txt"


def list_prompts() -> list:
    """Lista todos los prompts disponibles"""
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]
