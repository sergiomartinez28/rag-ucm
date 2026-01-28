"""
Módulo de generación de respuestas
Usa LLM local para generar respuestas con citas
"""

from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger

from .preprocessor import Chunk
from .prompt_loader import load_prompt


class ResponseGenerator:
    """
    Genera respuestas usando un LLM local open source
    - Llama-3.2-3B-Instruct
    - Phi-4-mini
    - Qwen2.5-3B-Instruct
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device: str = "auto"
    ):
        """
        Args:
            model_name: Modelo de HuggingFace
            max_new_tokens: Tokens máximos a generar
            temperature: Temperatura de sampling (0.0-1.0)
            device: 'cuda', 'cpu', o 'auto'
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        logger.info(f"Cargando modelo LLM: {model_name}")
        
        # Determinar dispositivo
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Usando dispositivo: {device}")
        
        # Cargar tokenizer con optimizaciones
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configurar pad_token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        if device == "cuda":
            logger.info("Cargando modelo en GPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(device)
        else:
            logger.info("Cargando modelo en CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(device)
        
        # Poner en modo evaluación
        self.model.eval()
        
        logger.success(f"✓ Modelo cargado en {device}")
    
    def create_prompt(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict:
        """
        Crea el prompt usando el formato de chat del tokenizer
        
        Args:
            query: Pregunta del usuario
            contexts: Lista de (Chunk, score) recuperados
        
        Returns:
            Dict con messages para apply_chat_template
        """
        # Construir contextos con referencias - optimizado
        context_texts = []
        for i, (chunk, score) in enumerate(contexts, 1):
            metadata = chunk.metadata
            title = metadata.get('title', metadata.get('filename', 'Doc'))
            # Texto más corto para modelo 1.5B
            text = chunk.text[:400].strip()
            context_texts.append(f"[{i}] {title}: {text}")
        
        contexts_str = "\n\n".join(context_texts)
        
        # Cargar prompts desde archivos externos
        system_prompt = load_prompt("generator_system")
        user_prompt = load_prompt("generator_user", contexts=contexts_str, query=query)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def generate(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ) -> Dict[str, Any]:
        """
        Genera respuesta usando el LLM
        
        Args:
            query: Pregunta del usuario
            contexts: Contextos recuperados
        
        Returns:
            Dict con 'answer', 'sources', y metadata
        """
        if not contexts:
            logger.warning("No hay contextos para generar respuesta")
            return {
                'answer': "No encuentro información relevante en la normativa disponible. Te recomiendo contactar con secretaría de tu facultad.",
                'sources': [],
                'metadata': {'warning': 'no_contexts'}
            }
        
        logger.info("Generando respuesta...")
        
        # Crear mensajes para el chat
        messages = self.create_prompt(query, contexts)
        
        # Usar apply_chat_template para formato correcto
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenizar con límite ajustado
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1536,
            truncation=True,
            padding=False  # Sin padding para evitar problemas
        ).to(self.device)
        
        # Obtener longitud del prompt para extraer respuesta
        prompt_length = inputs['input_ids'].shape[1]
        
        # Usar eos_token_id como pad_token_id (más seguro)
        eos_id = self.tokenizer.eos_token_id
        
        # Generar respuesta - greedy decoding (más rápido y determinista)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
                use_cache=True
            )
        
        # Decodificar solo los tokens nuevos generados
        generated_tokens = outputs[0][prompt_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Preparar fuentes
        sources = self._format_sources(contexts)
        
        logger.success("✓ Respuesta generada")
        
        return {
            'answer': answer,
            'sources': sources,
            'contexts': contexts,
            'metadata': {
                'model': self.model_name,
                'temperature': self.temperature,
                'num_contexts': len(contexts)
            }
        }
    
    def _format_sources(
        self,
        contexts: List[Tuple[Chunk, float]]
    ) -> List[Dict[str, str]]:
        """
        Formatea las fuentes para mostrar al usuario
        """
        sources = []
        
        for i, (chunk, score) in enumerate(contexts, 1):
            metadata = chunk.metadata
            
            source = {
                'id': i,
                'title': metadata.get('title', metadata.get('filename', 'Documento')),
                'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'score': round(score, 3),
                'metadata': {
                    'faculty': metadata.get('faculty', ''),
                    'year': metadata.get('year', ''),
                    'url': metadata.get('url', ''),
                    'chunk_id': chunk.chunk_id
                }
            }
            
            sources.append(source)
        
        return sources
    
    def generate_stream(
        self,
        query: str,
        contexts: List[Tuple[Chunk, float]]
    ):
        """
        Genera respuesta en streaming (para interfaz más dinámica)
        TODO: Implementar streaming real
        """
        # Por ahora, retorna la respuesta completa
        # En futuras versiones, se puede implementar streaming real
        result = self.generate(query, contexts)
        yield result['answer']


if __name__ == "__main__":
    # Ejemplo de uso
    print("✓ Módulo generator listo para usar")
    print("Nota: La primera vez que uses un modelo, se descargará de HuggingFace")
