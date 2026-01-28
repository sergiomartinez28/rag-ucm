"""
Utilidades comunes para el sistema RAG-UCM
Incluye decoradores, gestión de contexto y helpers
"""

import time
import functools
from contextlib import contextmanager
from typing import Callable, Any, Optional, Dict
from loguru import logger
import torch
import gc


def timed(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución de funciones
    
    Usage:
        @timed
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"⏱️  {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def timed_with_result(func: Callable) -> Callable:
    """
    Decorador que retorna resultado + tiempo de ejecución
    
    Usage:
        @timed_with_result
        def my_function():
            return data
        
        result, elapsed = my_function()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


class TimingContext:
    """
    Context manager para medir tiempos de bloques de código
    
    Usage:
        with TimingContext("Loading models") as timer:
            # código a medir
            pass
        print(f"Elapsed: {timer.elapsed:.2f}s")
    """
    
    def __init__(self, name: str, log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.log:
            logger.debug(f"⏱️  Iniciando: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.log:
            if exc_type is None:
                logger.debug(f"✓ {self.name}: {self.elapsed:.3f}s")
            else:
                logger.error(f"✗ {self.name}: falló después de {self.elapsed:.3f}s")


@contextmanager
def torch_memory_cleanup():
    """
    Context manager para limpiar memoria GPU/CPU después de operaciones pesadas
    
    Usage:
        with torch_memory_cleanup():
            # operaciones con modelos
            pass
        # memoria liberada automáticamente
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ResourceManager:
    """
    Gestor de recursos para modelos ML
    Asegura liberación correcta de memoria
    """
    
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.resource = None
    
    def __enter__(self):
        logger.debug(f"📦 Adquiriendo recurso: {self.resource_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"🗑️  Liberando recurso: {self.resource_name}")
        if self.resource is not None:
            del self.resource
            self.resource = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def set_resource(self, resource: Any):
        """Asigna el recurso a gestionar"""
        self.resource = resource
        return resource


def safe_execute(func: Callable, fallback: Any = None, log_errors: bool = True) -> Callable:
    """
    Decorador para ejecución segura con fallback
    
    Usage:
        @safe_execute(fallback=[])
        def risky_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"Error en {func.__name__}: {e}")
            return fallback
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorador para reintentar operaciones fallidas
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre reintentos (segundos)
        backoff: Factor multiplicador del delay
    
    Usage:
        @retry(max_attempts=3, delay=1.0)
        def unstable_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Intento {attempt}/{max_attempts} falló para {func.__name__}: {e}. "
                            f"Reintentando en {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"Todos los intentos fallaron para {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def batch_iterator(items: list, batch_size: int):
    """
    Generador para procesar items en batches
    
    Usage:
        for batch in batch_iterator(items, batch_size=32):
            process(batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def format_bytes(bytes_size: int) -> str:
    """Formatea tamaño en bytes a formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_gpu_memory_info() -> Dict[str, str]:
    """Obtiene información de memoria GPU si está disponible"""
    if not torch.cuda.is_available():
        return {"status": "No GPU available"}
    
    return {
        "device": torch.cuda.get_device_name(0),
        "allocated": format_bytes(torch.cuda.memory_allocated(0)),
        "cached": format_bytes(torch.cuda.memory_reserved(0)),
        "total": format_bytes(torch.cuda.get_device_properties(0).total_memory)
    }


class ProgressTracker:
    """
    Tracker simple de progreso para operaciones largas
    
    Usage:
        tracker = ProgressTracker(total=100, desc="Processing")
        for item in items:
            # procesar
            tracker.update()
        tracker.close()
    """
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Incrementa el contador"""
        self.current += n
        if self.current % max(1, self.total // 10) == 0:  # Log cada 10%
            progress = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
            logger.info(f"{self.desc}: {progress:.1f}% ({self.current}/{self.total}) - ETA: {eta:.1f}s")
    
    def close(self):
        """Finaliza el tracking"""
        elapsed = time.time() - self.start_time
        logger.success(f"✓ {self.desc} completado: {self.current}/{self.total} en {elapsed:.1f}s")


def validate_file_exists(filepath: str, error_msg: Optional[str] = None) -> bool:
    """Valida que un archivo exista"""
    from pathlib import Path
    path = Path(filepath)
    if not path.exists():
        msg = error_msg or f"Archivo no encontrado: {filepath}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    return True


def ensure_dir(dirpath: str) -> None:
    """Asegura que un directorio exista, creándolo si es necesario"""
    from pathlib import Path
    Path(dirpath).mkdir(parents=True, exist_ok=True)
