"""
Módulo de preprocesamiento de documentos
Limpieza, normalización y chunking de documentos UCM
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import pypdf
import pdfplumber
from bs4 import BeautifulSoup
from loguru import logger
from unidecode import unidecode


@dataclass
class Document:
    """Representa un documento procesado"""
    text: str
    metadata: Dict[str, str]
    source: str


@dataclass
class Chunk:
    """Representa un fragmento (chunk) de documento"""
    text: str
    metadata: Dict[str, str]
    chunk_id: str
    doc_id: str


class DocumentPreprocessor:
    """
    Preprocesa documentos PDF/HTML de normativa UCM
    - Extrae texto
    - Limpia y normaliza
    - Divide en chunks semánticos
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Tamaño aproximado de cada chunk en tokens
            chunk_overlap: Solapamiento entre chunks consecutivos
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentPreprocessor inicializado: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_from_pdf(self, pdf_path: Path) -> str:
        """
        Extrae texto de un PDF
        Usa pdfplumber para mejor extracción de tablas y estructura
        """
        logger.info(f"Extrayendo texto de PDF: {pdf_path}")
        
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                        
            full_text = "\n\n".join(text_parts)
            logger.success(f"✓ Extraídas {len(text_parts)} páginas de {pdf_path.name}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extrayendo PDF {pdf_path}: {e}")
            raise
    
    def extract_from_html(self, html_path: Path) -> str:
        """
        Extrae texto limpio de HTML
        """
        logger.info(f"Extrayendo texto de HTML: {html_path}")
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
            
            # Eliminar scripts, estilos, etc.
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            logger.success(f"✓ Texto extraído de {html_path.name}")
            return text
            
        except Exception as e:
            logger.error(f"Error extrayendo HTML {html_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto
        - Elimina cabeceras/pies repetidos
        - Normaliza espacios
        - Mantiene estructura de párrafos
        """
        # Eliminar múltiples saltos de línea
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r' {2,}', ' ', text)
        
        # Eliminar líneas muy cortas que suelen ser headers/footers
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 15 or len(line.strip()) == 0]
        text = '\n'.join(lines)
        
        # Normalizar guiones y comillas
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def create_chunks(self, text: str, metadata: Dict[str, str]) -> List[Chunk]:
        """
        Divide el texto en chunks con solapamiento
        Intenta respetar límites de párrafos y secciones
        """
        # Dividir por párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_counter = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            # Si el párrafo solo es muy largo, dividirlo
            if para_length > self.chunk_size:
                # Guardar chunk actual si existe
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, metadata, chunk_counter
                    ))
                    chunk_counter += 1
                    current_chunk = []
                    current_length = 0
                
                # Dividir párrafo largo por oraciones
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_length = 0
                
                for sent in sentences:
                    sent_length = len(sent.split())
                    if temp_length + sent_length > self.chunk_size and temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text, metadata, chunk_counter
                        ))
                        chunk_counter += 1
                        # Mantener overlap
                        overlap_words = ' '.join(temp_chunk).split()[-self.chunk_overlap:]
                        temp_chunk = [' '.join(overlap_words), sent]
                        temp_length = len(' '.join(temp_chunk).split())
                    else:
                        temp_chunk.append(sent)
                        temp_length += sent_length
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length
                    
            # Párrafo normal
            elif current_length + para_length > self.chunk_size and current_chunk:
                # Guardar chunk actual
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, metadata, chunk_counter
                ))
                chunk_counter += 1
                
                # Comenzar nuevo chunk con overlap
                overlap_words = chunk_text.split()[-self.chunk_overlap:]
                current_chunk = [' '.join(overlap_words), para]
                current_length = len(' '.join(current_chunk).split())
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Último chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, metadata, chunk_counter
            ))
        
        logger.info(f"✓ Creados {len(chunks)} chunks del documento")
        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict[str, str], chunk_id: int) -> Chunk:
        """Crea un objeto Chunk con metadata"""
        doc_id = metadata.get('doc_id', 'unknown')
        
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_number'] = chunk_id
        chunk_metadata['char_count'] = len(text)
        chunk_metadata['word_count'] = len(text.split())
        
        return Chunk(
            text=text,
            metadata=chunk_metadata,
            chunk_id=f"{doc_id}_chunk_{chunk_id}",
            doc_id=doc_id
        )
    
    def process_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, str]] = None
    ) -> List[Chunk]:
        """
        Pipeline completo: extrae, limpia y divide en chunks
        
        Args:
            file_path: Ruta al documento
            metadata: Metadatos opcionales (título, facultad, fecha, etc.)
        
        Returns:
            Lista de chunks procesados
        """
        if metadata is None:
            metadata = {}
        
        # Añadir metadata básico
        metadata['filename'] = file_path.name
        metadata['doc_id'] = file_path.stem
        
        # Extraer según tipo
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.html', '.htm']:
            text = self.extract_from_html(file_path)
        else:
            raise ValueError(f"Formato no soportado: {file_path.suffix}")
        
        # Limpiar
        text = self.clean_text(text)
        
        # Crear chunks
        chunks = self.create_chunks(text, metadata)
        
        logger.success(f"✓ Documento procesado: {len(chunks)} chunks generados")
        return chunks


if __name__ == "__main__":
    # Ejemplo de uso
    from pathlib import Path
    
    preprocessor = DocumentPreprocessor(chunk_size=600, chunk_overlap=100)
    
    # Procesar un PDF de ejemplo
    # pdf_path = Path("data/raw/normativa_tfg_ejemplo.pdf")
    # chunks = preprocessor.process_document(
    #     pdf_path,
    #     metadata={
    #         'title': 'Normativa TFG Facultad de Informática',
    #         'faculty': 'Informática',
    #         'year': '2024',
    #         'url': 'https://...'
    #     }
    # )
    
    print("✓ Módulo preprocessor listo para usar")
