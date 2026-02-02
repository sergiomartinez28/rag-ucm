"""
Generador de datasets de evaluación
Usa un LLM local para generar preguntas y respuestas de referencia desde los chunks
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from tqdm import tqdm

from ..prompt_loader import load_prompt


@dataclass
class QAPair:
    """Par pregunta-respuesta para evaluación"""
    id: int
    question: str
    reference_answer: str
    question_type: str  # factual, procedural, conceptual
    source_doc: str  # Nombre del documento (legible)
    doc_id: str      # ID real del documento en el indexador
    chunk_id: int    # ID real del chunk en el indexador
    chunk_text: str
    category: str


class DatasetGenerator:
    """
    Genera preguntas y respuestas sintéticas desde chunks de documentos
    usando un LLM local potente
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto"
    ):
        """
        Args:
            model_name: Nombre del modelo HuggingFace para generación
            device: Dispositivo (cuda, cpu, auto)
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Cargando modelo generador (Ground Truth): {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Cargar modelo en float16 para GPU
        logger.info("Cargando modelo en float16...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Más compatible
        )
        
        logger.success(f"✓ Modelo generador (Ground Truth) cargado: {model_name}")
    
    @staticmethod
    def _normalize_question(question: str) -> str:
        """
        Normaliza una pregunta para detectar duplicados:
        - Lowercase
        - Elimina puntuación (excepto espacios)
        - Colapsa espacios múltiples
        - Strip de espacios
        
        Returns:
            Versión normalizada de la pregunta
        """
        # Lowercase
        normalized = question.lower()
        
        # Eliminar signos de puntuación pero preservar espacios y letras acentuadas
        normalized = re.sub(r'[^\w\sáéíóúüñ]', '', normalized)
        
        # Colapsar espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip
        normalized = normalized.strip()
        
        return normalized
    
    def _generate_response(self, prompt: str, max_tokens: int = 800) -> str:
        """Genera respuesta del LLM"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.5,  # Más consistente
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extrae JSON de la respuesta del LLM"""
        try:
            # Buscar JSON en la respuesta
            start = text.find('[')
            if start == -1:
                start = text.find('{')
            
            end = text.rfind(']')
            if end == -1:
                end = text.rfind('}')
            
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def generate_qa_from_chunk(
        self,
        chunk_text: str,
        source_doc: str,
        chunk_id: int,
        category: str
    ) -> List[Dict]:
        """
        Genera preguntas y respuestas desde un chunk
        
        Args:
            chunk_text: Texto del chunk
            source_doc: Nombre del documento fuente
            chunk_id: ID del chunk
            category: Categoría del documento
        
        Returns:
            Lista de diccionarios con Q&A
        """
        # Cargar prompt desde archivo externo
        prompt = load_prompt(
            "dataset_generator",
            source_doc=source_doc,
            chunk_text=chunk_text[:2000]
        )

        response = self._generate_response(prompt)
        qa_list = self._extract_json(response)
        
        if qa_list is None:
            logger.warning(f"No se pudo extraer JSON para chunk {chunk_id}")
            return []
        
        # Asegurar que es una lista
        if isinstance(qa_list, dict):
            qa_list = [qa_list]
        
        # Añadir metadatos y mapear question_text -> question
        for qa in qa_list:
            # Mapear question_text -> question si es necesario
            if 'question_text' in qa and 'question' not in qa:
                qa['question'] = qa.pop('question_text')
            
            qa['source_doc'] = source_doc
            qa['chunk_id'] = chunk_id
            qa['category'] = category
            qa['chunk_text'] = chunk_text[:500]  # Guardar parte del chunk para referencia
        
        return qa_list
    
    def generate_dataset(
        self,
        chunks_path: Path,
        output_path: Path,
        num_samples: int = 5,
        random_seed: int = 42,
        max_workers: int = 4
    ) -> List[QAPair]:
        """
        Genera un dataset de evaluación desde chunks guardados
        
        Args:
            chunks_path: Ruta al archivo chunks.pkl
            output_path: Ruta para guardar el dataset
            num_samples: Número de chunks a muestrear
            random_seed: Semilla para reproducibilidad
            max_workers: Número de workers paralelos (default: 4)
        
        Returns:
            Lista de QAPair
        """
        logger.info(f"Cargando chunks desde {chunks_path}")
        
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        logger.info(f"Total chunks disponibles: {len(chunks)}")
        
        # Filtrar chunks con contenido sustancial (mínimo 200 caracteres)
        # y que contengan información útil (artículos, números, procedimientos)
        quality_indices = []
        for i, chunk in enumerate(chunks):
            text = chunk.text
            # Chunks con buen contenido: suficientemente largos y con estructura
            if len(text) >= 300:
                # Preferir chunks con artículos, números o procedimientos
                has_structure = any(word in text.lower() for word in 
                    ['artículo', 'artculo', 'apartado', 'punto', 'será', 'deberá', 
                     'podrá', 'establecer', 'corresponde', 'función', 'plazo'])
                if has_structure:
                    quality_indices.append(i)
        
        logger.info(f"Chunks de calidad encontrados: {len(quality_indices)}")
        
        # Muestreo aleatorio de chunks de calidad
        random.seed(random_seed)
        if len(quality_indices) >= num_samples:
            sampled_indices = random.sample(quality_indices, num_samples)
        else:
            # Si no hay suficientes de calidad, completar con otros
            sampled_indices = quality_indices + random.sample(
                [i for i in range(len(chunks)) if i not in quality_indices],
                min(num_samples - len(quality_indices), len(chunks) - len(quality_indices))
            )
        
        all_qa_pairs = []
        seen_questions: Set[str] = set()  # Track de preguntas normalizadas para deduplicar
        qa_id = 0
        duplicates_count = 0
        
        # Función para procesar un chunk
        def process_chunk(idx):
            chunk = chunks[idx]
            
            # Obtener doc_id real del chunk
            doc_id = chunk.doc_id if hasattr(chunk, 'doc_id') else (
                chunk.source if hasattr(chunk, 'source') else "unknown"
            )
            
            # Nombre legible del documento (sin _chunk_X)
            source_name = doc_id
            if '_chunk_' in source_name:
                source_name = source_name.split('_chunk_')[0]
            
            category = self._infer_category(source_name)
            
            # Generar Q&A
            qa_list = self.generate_qa_from_chunk(
                chunk_text=chunk.text,
                source_doc=source_name,
                chunk_id=idx,
                category=category
            )
            
            return qa_list, source_name, doc_id, idx, category, chunk.text
        
        # Procesamiento paralelo
        logger.info(f"Procesando {len(sampled_indices)} chunks con {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, idx): idx for idx in sampled_indices}
            
            with tqdm(total=len(sampled_indices), desc="Generando Q&A") as pbar:
                for future in as_completed(futures):
                    try:
                        qa_list, source_name, doc_id, idx, category, chunk_text = future.result()
                        
                        # Convertir a QAPair y deduplicar
                        for qa in qa_list:
                            question_text = qa.get('question', '').strip()
                            
                            # Skip si pregunta vacía
                            if not question_text:
                                continue
                            
                            # Normalizar pregunta para detectar duplicados
                            normalized_q = self._normalize_question(question_text)
                            
                            # Deduplicar: skip si ya vimos esta pregunta
                            if normalized_q in seen_questions:
                                duplicates_count += 1
                                logger.debug(f"Duplicado detectado: '{question_text[:50]}...'")
                                continue
                            
                            # Registrar pregunta como vista
                            seen_questions.add(normalized_q)
                            
                            # Crear QAPair con doc_id y chunk_id reales
                            qa_pair = QAPair(
                                id=qa_id,
                                question=question_text,
                                reference_answer=qa.get('reference_answer', ''),
                                question_type=qa.get('question_type', 'unknown'),
                                source_doc=source_name,
                                doc_id=doc_id,
                                chunk_id=idx,
                                chunk_text=chunk_text[:500],
                                category=category
                            )
                            all_qa_pairs.append(qa_pair)
                            qa_id += 1
                    except Exception as e:
                        logger.error(f"Error procesando chunk {futures[future]}: {e}")
                    finally:
                        pbar.update(1)
        
        # Ordenar por ID para mantener consistencia
        all_qa_pairs.sort(key=lambda x: x.id)
        
        # Log de deduplicación
        if duplicates_count > 0:
            logger.info(f"Duplicados eliminados durante generación: {duplicates_count}")
        
        # Guardar dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = [asdict(qa) for qa in all_qa_pairs]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.success(
            f"✓ Dataset generado: {len(all_qa_pairs)} preguntas únicas "
            f"({duplicates_count} duplicados eliminados) en {output_path}"
        )
        
        return all_qa_pairs
    
    def _infer_category(self, source: str) -> str:
        """Infiere la categoría desde el nombre del documento"""
        source_lower = source.lower()
        
        if 'tfm' in source_lower or 'tfg' in source_lower:
            return 'trabajo_fin'
        elif 'master' in source_lower or 'máster' in source_lower:
            return 'master'
        elif 'gobierno' in source_lower or 'claustro' in source_lower:
            return 'gobierno'
        elif 'defensor' in source_lower or 'reclamacion' in source_lower:
            return 'defensoria'
        elif 'beca' in source_lower:
            return 'becas'
        elif 'estatuto' in source_lower:
            return 'estatutos'
        elif 'reglamento' in source_lower:
            return 'reglamentos'
        else:
            return 'general'


if __name__ == "__main__":
    # Test rápido
    chunks_path = Path("data/processed/faiss_index/chunks.pkl")
    output_path = Path("data/evaluation/dataset_test.json")
    
    generator = DatasetGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct"
    )
    
    qa_dataset = generator.generate_dataset(
        chunks_path=chunks_path,
        output_path=output_path,
        num_samples=3
    )
    
    print(f"\nGeneradas {len(qa_dataset)} preguntas:")
    for qa_item in qa_dataset:
        print(f"\n[{qa_item.question_type}] {qa_item.question}")
        print(f"  → {qa_item.reference_answer[:100]}...")
