"""
Módulo de preprocesamiento de documentos
Limpieza, normalización y chunking de documentos UCM
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
from bs4 import BeautifulSoup
from loguru import logger

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Fase 1: Intentar usar PyMuPDF si está disponible
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .utils import timed, ProgressTracker, validate_file_exists


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
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentPreprocessor inicializado: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    @timed
    def extract_from_pdf(self, pdf_path: Path) -> str:
        """
        Extrae texto de un PDF
        Fase 1: Intenta múltiples métodos para máxima compatibilidad
        1. PyMuPDF (fitz) - Más robusto para PDFs problemáticos
        2. pdfplumber con tolerancias - Para PDFs estándar
        3. OCR - Para PDFs escaneados
        """
        validate_file_exists(str(pdf_path))
        logger.info(f"Extrayendo texto de PDF: {pdf_path.name}")
        
        text_parts = []
        
        try:
            # Intento 1: PyMuPDF si está disponible (más robusto)
            if PYMUPDF_AVAILABLE:
                logger.debug(f"Intentando extracción con PyMuPDF")
                text_parts = self._extract_with_pymupdf(pdf_path)
                
                if text_parts:
                    full_text = "\n\n".join(text_parts)
                    logger.success(f"✓ Extraídas {len(text_parts)} páginas (PyMuPDF) de {pdf_path.name}")
                    return full_text
            
            # Intento 2: pdfplumber con tolerancias ajustadas
            logger.debug(f"Intentando extracción con pdfplumber")
            text_parts = self._extract_with_pdfplumber_tuned(pdf_path)
            
            # Intento 3: OCR si no hay texto extraíble
            if not text_parts:
                if OCR_AVAILABLE:
                    logger.info(f"PDF sin texto extraíble, intentando OCR: {pdf_path.name}")
                    text_parts = self._extract_with_ocr(pdf_path)
                else:
                    logger.warning(f"PDF {pdf_path.name} sin texto extraíble y OCR no disponible")
                        
            full_text = "\n\n".join(text_parts)
            if text_parts:
                logger.success(f"✓ Extraídas {len(text_parts)} páginas de {pdf_path.name}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extrayendo PDF {pdf_path}: {e}")
            raise RuntimeError(f"Error extrayendo PDF: {e}") from e
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[str]:
        """
        Extrae texto usando PyMuPDF (fitz)
        Fase 1: Alternativa más robusta para PDFs problemáticos
        """
        if not PYMUPDF_AVAILABLE:
            return []
        
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            total_pages = len(doc)
            tracker = ProgressTracker(total=total_pages, desc=f"Extrayendo (PyMuPDF) {pdf_path.name}")
            
            for page_num in range(total_pages):
                page = doc[page_num]
                # Extrae con dictools para mejor formato
                text = page.get_text("text")
                if text and text.strip():
                    text_parts.append(text)
                tracker.update()
            
            tracker.close()
            doc.close()
            
            logger.debug(f"PyMuPDF: extraído {len(text_parts)} páginas")
            return text_parts
            
        except Exception as e:
            logger.warning(f"Error con PyMuPDF {pdf_path.name}: {e}")
            return []
    
    def _extract_with_pdfplumber_tuned(self, pdf_path: Path) -> List[str]:
        """
        Extrae texto usando pdfplumber con tolerancias ajustadas
        Fase 1: Parámetros optimizados para PDFs de normativa
        """
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                tracker = ProgressTracker(total=total_pages, desc=f"Extrayendo {pdf_path.name}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Intento 1: Extracción estándar
                    text = page.extract_text()
                    
                    # Intento 2: Si el texto es muy corto, probar layout_mode
                    if not text or len(text.strip()) < 100:
                        try:
                            # layout mode es más tolerante con PDFs con layout complejo
                            text = page.extract_text(layout=True)
                        except Exception:
                            pass
                    
                    # Intento 3: Si aún hay poco texto, probar con tolerancias
                    if not text or len(text.strip()) < 100:
                        try:
                            # Usar tabla extraction si hay tablas
                            tables = page.extract_tables()
                            if tables:
                                text_content = []
                                for table in tables:
                                    for row in table:
                                        text_content.append(" | ".join(str(cell) if cell else "" for cell in row))
                                if text_content:
                                    text = "\n".join(text_content)
                        except Exception:
                            pass
                    
                    if text and text.strip():
                        text_parts.append(text)
                    
                    tracker.update()
                
                tracker.close()
            
            logger.debug(f"pdfplumber: extraído {len(text_parts)} páginas")
            return text_parts
            
        except Exception as e:
            logger.warning(f"Error con pdfplumber {pdf_path.name}: {e}")
            return []
    
    @timed
    def _extract_with_ocr(self, pdf_path: Path) -> List[str]:
        """
        Extrae texto de PDFs escaneados usando OCR (Tesseract)
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR no disponible. Instala: pip install pytesseract pdf2image")
            return []
        
        text_parts = []
        try:
            # Convertir PDF a imágenes
            images = convert_from_path(str(pdf_path))
            total_pages = len(images)
            tracker = ProgressTracker(total=total_pages, desc=f"OCR en {pdf_path.name}")
            
            for page_num, image in enumerate(images, 1):
                # Aplicar OCR a cada página
                text = pytesseract.image_to_string(image, lang='spa+eng')
                if text and text.strip():
                    text_parts.append(text)
                tracker.update()
            
            tracker.close()
            
            if text_parts:
                logger.success(f"✓ OCR completado: {len(text_parts)} páginas de {pdf_path.name}")
            
            return text_parts
            
        except Exception as e:
            logger.error(f"Error en OCR para {pdf_path.name}: {e}")
            return []
    
    @timed
    def extract_from_html(self, html_path: Path) -> str:
        """
        Extrae texto limpio de HTML
        """
        validate_file_exists(str(html_path))
        logger.info(f"Extrayendo texto de HTML: {html_path.name}")
        
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
            raise RuntimeError(f"Error extrayendo HTML: {e}") from e
    
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto PDF
        Fase 1: Heurísticas avanzadas para legibilidad
        
        - Deshacer guiones de final de línea (palabra-\n → palabra)
        - Insertar espacios perdidos (minúscula+Mayúscula, letra+número, etc.)
        - Elimina cabeceras/pies repetidos
        - Normaliza espacios y puntuación
        - Mantiene estructura de párrafos
        """
        if not text or not text.strip():
            logger.warning("Texto vacío recibido para limpieza")
            return ""
        
        # 1. DESHACER GUIONES DE FIN DE LÍNEA
        # Patrón: palabra-\n continuación → palabracontinuación
        text = re.sub(r'(\w+)-\n\s*', r'\1', text)
        # También: palabra - \n continuación
        text = re.sub(r'(\w+)\s*-\s*\n\s*', r'\1', text)
        
        # 2. INSERTAR ESPACIOS PERDIDOS ENTRE MINÚSCULA Y MAYÚSCULA
        # Pero evitar siglas (ej: "UCM", "PDF")
        # Heurística: si sigue de número o es final de palabra larga, insertar espacio
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 3. INSERTAR ESPACIOS ENTRE LETRAS Y NÚMEROS
        # letra+número o número+letra
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # 4. INSERTAR ESPACIO DESPUÉS DE PUNTUACIÓN SI FALTA
        # Después de . ! ? : ; seguido de letra (no es decimal)
        text = re.sub(r'([.!?:;])([a-zA-Z])', r'\1 \2', text)
        
        # 5. ELIMINAR MÚLTIPLES SALTOS DE LÍNEA
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. ELIMINAR ESPACIOS MÚLTIPLES
        text = re.sub(r' {2,}', ' ', text)
        
        # 7. LIMPIAR ESPACIOS ALREDEDOR DE SALTOS DE LÍNEA
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        
        # 8. ELIMINAR LÍNEAS MUY CORTAS (headers/footers típicos)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 15 or len(line.strip()) == 0]
        text = '\n'.join(lines)
        
        # 9. NORMALIZAR GUIONES Y COMILLAS TIPOGRÁFICAS
        replacements = {
            "–": "-",
            "—": "-",
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        # 10. LIMPIAR ESPACIOS FINALES Y DUPLICADOS NUEVAMENTE
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    @timed
    def create_chunks(self, text: str, metadata: Dict[str, str]) -> List[Chunk]:
        """
        Divide el texto en chunks respetando unidades semánticas
        Fase 3: Normativa-aware - detecta Artículos, Disposiciones, Capítulos, etc.
        
        Detecta cabeceras y chunkea por fronteras estructurales
        Si una sección es muy grande, la divide internamente
        """
        if not text or not text.strip():
            logger.warning("No se puede crear chunks de texto vacío")
            return []
        
        # Fase 3: Detectar secciones normativa-aware
        sections = self._split_by_normative_structure(text)
        
        if not sections:
            logger.warning("No se encontraron secciones normativa en el texto")
            return []
        
        chunks = []
        chunk_counter = 0
        
        for section_title, section_text in sections:
            # Cada sección se convierte en uno o más chunks
            section_chunks = self._chunk_section(section_title, section_text, metadata, chunk_counter)
            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)
        
        logger.info(f"✓ Creados {len(chunks)} chunks (normativa-aware)")
        return chunks
    
    def _split_by_normative_structure(self, text: str) -> List[tuple]:
        """
        Fase 3: Detecta cabeceras de normativa y divide por ellas
        
        Detecta: Artículo, Disposición, Capítulo, Sección, Apartado, etc.
        Retorna lista de (título, contenido)
        """
        # Patrones de cabeceras normativa UCM
        # Artículo 1, Artículo 2.1, etc.
        # Disposición Adicional Primera, Segunda, etc.
        # Capítulo I, Capítulo II, etc.
        
        patterns = [
            (r'(?:^|\n)(Artículo\s+[\d.]+[^\n]*)', 'article'),
            (r'(?:^|\n)(Disposición\s+(?:Adicional|Transitoria|Derogatoria)\s+\w+[^\n]*)', 'disposition'),
            (r'(?:^|\n)(Capítulo\s+[IVX]+[^\n]*)', 'chapter'),
            (r'(?:^|\n)(Sección\s+[\d.]+[^\n]*)', 'section'),
            (r'(?:^|\n)(Apartado\s+[\d.]+[^\n]*)', 'apartado'),
            (r'(?:^|\n)(Párrafo\s+[\d.]+[^\n]*)', 'paragraph'),
        ]
        
        # Encontrar todas las cabeceras
        headers = []
        for pattern, header_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                headers.append({
                    'pos': match.start(),
                    'title': match.group(1).strip(),
                    'type': header_type
                })
        
        # Si no hay headers, dividir por párrafos simples
        if not headers:
            logger.debug("No se detectaron cabeceras normativa, usando paragraphs")
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return [(f"Párrafo {i+1}", para) for i, para in enumerate(paragraphs[:10])]
        
        # Ordenar por posición y extraer secciones
        headers.sort(key=lambda x: x['pos'])
        sections = []
        
        for i, header in enumerate(headers):
            title = header['title']
            start_pos = header['pos']
            
            # Encontrar fin de esta sección (principio de la siguiente o fin del texto)
            if i + 1 < len(headers):
                end_pos = headers[i + 1]['pos']
            else:
                end_pos = len(text)
            
            # Extraer contenido (sin duplicar título)
            content_start = start_pos + len(title)
            section_content = text[content_start:end_pos].strip()
            
            if section_content:
                sections.append((title, section_content))
        
        logger.debug(f"Detectadas {len(sections)} secciones normativa")
        return sections
    
    def _chunk_section(
        self, 
        section_title: str, 
        section_text: str, 
        metadata: Dict[str, str],
        start_chunk_id: int
    ) -> List[Chunk]:
        """
        Divide una sección en chunks
        Si cabe en un chunk, va completo
        Si no, divide por párrafos/oraciones manteniendo overlap
        """
        chunks = []
        section_length = len(section_text.split())
        
        # Si cabe en un chunk, devolver completo
        if section_length <= self.chunk_size:
            chunk_text = f"{section_title}\n\n{section_text}"
            chunk = self._create_chunk(chunk_text, metadata, start_chunk_id)
            chunk.metadata['section_title'] = section_title
            chunks.append(chunk)
            return chunks
        
        # Si es muy largo, dividir por párrafos pero respetando límites
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        
        # Reconstruir chunk con header + párrafos
        current_chunk = [f"{section_title}"]  # Empezar con el título
        current_length = len(section_title.split())
        chunk_counter = start_chunk_id
        
        for para in paragraphs:
            para_length = len(para.split())
            
            # Si párrafo es muy largo, dividirlo por oraciones
            if para_length > self.chunk_size:
                # Guardar chunk actual si existe
                if len(current_chunk) > 1:  # Más que solo el título
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, metadata, chunk_counter)
                    chunk.metadata['section_title'] = section_title
                    chunks.append(chunk)
                    chunk_counter += 1
                    
                    # Nuevamente empezar con título para overlap
                    current_chunk = [f"{section_title}"]
                    current_length = len(section_title.split())
                
                # Dividir párrafo largo por oraciones
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_length = 0
                
                for sent in sentences:
                    sent_length = len(sent.split())
                    if temp_length + sent_length > self.chunk_size and temp_chunk:
                        # Guardar este chunk
                        chunk_text = f"{section_title}\n\n" + ' '.join(temp_chunk)
                        chunk = self._create_chunk(chunk_text, metadata, chunk_counter)
                        chunk.metadata['section_title'] = section_title
                        chunks.append(chunk)
                        chunk_counter += 1
                        
                        # Mantener overlap - últimas palabras
                        overlap_words = ' '.join(temp_chunk).split()[-self.chunk_overlap:]
                        temp_chunk = overlap_words + [sent]
                        temp_length = len(temp_chunk)
                    else:
                        temp_chunk.append(sent)
                        temp_length += sent_length
                
                # Guardar resto de oraciones
                if temp_chunk:
                    current_chunk = [f"{section_title}"] + temp_chunk
                    current_length = len(' '.join(current_chunk).split())
                    
            # Párrafo normal - agregar si cabe
            elif current_length + para_length <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_length
            else:
                # No cabe - guardar chunk y comenzar nuevo
                if len(current_chunk) > 1:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, metadata, chunk_counter)
                    chunk.metadata['section_title'] = section_title
                    chunks.append(chunk)
                    chunk_counter += 1
                    
                    # Comenzar nuevo con overlap
                    overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 1 else current_chunk[0]
                    current_chunk = [f"{section_title}", overlap_text, para]
                    current_length = len(' '.join(current_chunk).split())
                else:
                    current_chunk.append(para)
                    current_length += para_length
        
        # Último chunk
        if len(current_chunk) > 1:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = self._create_chunk(chunk_text, metadata, chunk_counter)
            chunk.metadata['section_title'] = section_title
            chunks.append(chunk)
        
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
    
    @timed
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
        
        Raises:
            ValueError: Si el formato no es soportado
            RuntimeError: Si hay errores en extracción o procesamiento
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
        
        if not text:
            logger.warning(f"⚠️  Sin contenido extraíble: {file_path.name} (posiblemente PDF escaneado)")
            return []
        
        # Crear chunks
        chunks = self.create_chunks(text, metadata)
        
        if not chunks:
            logger.warning(f"⚠️  No se pudieron crear chunks de: {file_path.name}")
            return []
        
        logger.success(f"✓ Documento procesado: {len(chunks)} chunks generados")
        return chunks
    
    def process_batch(
        self,
        file_paths: List[Path],
        metadata_dict: Optional[Dict[str, Dict[str, str]]] = None
    ) -> List[Chunk]:
        """
        Procesa múltiples documentos en batch
        
        Args:
            file_paths: Lista de rutas a documentos
            metadata_dict: Diccionario con metadata por documento
        
        Returns:
            Lista combinada de todos los chunks
        """
        all_chunks = []
        failed_files = []
        
        tracker = ProgressTracker(total=len(file_paths), desc="Procesando documentos")
        
        for file_path in file_paths:
            try:
                metadata = metadata_dict.get(str(file_path), {}) if metadata_dict else {}
                chunks = self.process_document(file_path, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error procesando {file_path.name}: {e}")
                failed_files.append(file_path.name)
            finally:
                tracker.update()
        
        tracker.close()
        
        if failed_files:
            logger.warning(f"Archivos con errores ({len(failed_files)}): {', '.join(failed_files)}")
        
        logger.success(f"✓ Procesamiento batch completo: {len(all_chunks)} chunks totales")
        return all_chunks


if __name__ == "__main__":
    # Ejemplo de uso
    from pathlib import Path
    
    preprocessor = DocumentPreprocessor(chunk_size=600, chunk_overlap=100)
    
    print("✓ Módulo preprocessor listo para usar")
